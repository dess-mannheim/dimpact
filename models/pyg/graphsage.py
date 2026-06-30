import os
import numpy as np
import time
import torch
import torch.nn.functional as F
from torch_geometric import seed_everything
from torch_geometric.data import Data
from sklearn.linear_model import LogisticRegression
from torch_geometric.loader import LinkNeighborLoader, NeighborLoader
from torch_geometric.nn import GraphSAGE
from typing import Dict

from models.pyg.train_utils import prepare_link_prediction_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_num_neighbors(config: Dict, num_layers: int) -> list[int]:
    raw_num_neighbors = config["num_neighbors"]
    if isinstance(raw_num_neighbors, int):
        num_neighbors_list = [raw_num_neighbors] * num_layers
    else:
        num_neighbors_list = list(raw_num_neighbors)
        if len(num_neighbors_list) == 0:
            raise ValueError("config['num_neighbors'] must not be an empty list.")
        if len(num_neighbors_list) < num_layers:
            num_neighbors_list.extend([num_neighbors_list[-1]] * (num_layers - len(num_neighbors_list)))
        elif len(num_neighbors_list) > num_layers:
            num_neighbors_list = num_neighbors_list[:num_layers]

    max_sampled_nodes = config.get("max_sampled_nodes_per_batch", 200_000)
    fallback_full_neighbors = config.get("full_neighbor_fallback", 32)

    def estimate_nodes(neighbors: list[int]) -> float:
        frontier = 1.0
        total = 1.0
        for n in neighbors:
            if n < 0:
                return float("inf")
            frontier *= n
            total += frontier
        return total * config["batch_size"]

    original_neighbors = list(num_neighbors_list)
    num_neighbors_list = [fallback_full_neighbors if n < 0 else n for n in num_neighbors_list]
    if num_neighbors_list != original_neighbors:
        print(
            f"Adjusted num_neighbors from {original_neighbors} to {num_neighbors_list} "
            "to avoid full-neighborhood sampling with deep models."
        )

    if max_sampled_nodes is not None:
        while estimate_nodes(num_neighbors_list) > max_sampled_nodes and any(n > 1 for n in num_neighbors_list):
            max_neighbor_value = max(num_neighbors_list)
            idx = next(
                i
                for i in range(len(num_neighbors_list) - 1, -1, -1)
                if num_neighbors_list[i] == max_neighbor_value
            )
            num_neighbors_list[idx] = max(1, num_neighbors_list[idx] // 2)
        if num_neighbors_list != original_neighbors:
            print(
                f"Capped num_neighbors per layer from {original_neighbors} to {num_neighbors_list} "
                f"(estimated sampled nodes per batch <= {max_sampled_nodes})."
            )

    return num_neighbors_list


@torch.no_grad()
def infer_embeddings(
    model: GraphSAGE,
    data: Data,
    loader: NeighborLoader,
    embedding_dim: int,
) -> torch.Tensor:
    model.eval()
    z = torch.empty((data.num_nodes, embedding_dim), dtype=torch.float32)
    for batch in loader:
        batch = batch.to(device)
        pos_z = model(batch.x, batch.edge_index)[: batch.batch_size]
        z[batch.n_id[: batch.batch_size].cpu()] = pos_z.cpu()
    return z


def train_model(
    dataset: Data,
    embedding_dim: int,
    config: Dict,
    save_path: str,
    seed: int,
    embedding_path: str = None,
    keep_checkpoints: bool = False,
) -> float:

    seed_everything(seed)
    num_layers = config["num_layers"]
    config["num_neighbors_list"] = resolve_num_neighbors(config=config, num_layers=num_layers)
    data = dataset
    train_loader = LinkNeighborLoader(
        data,
        batch_size=config["batch_size"],
        shuffle=config["shuffle"],
        neg_sampling_ratio=config["neg_sampling_ratio"],
        num_neighbors=config["num_neighbors_list"],
    )
    test_loader = NeighborLoader(
        data,
        batch_size=config["batch_size"],
        num_neighbors=config["num_neighbors_list"],
        num_workers=config.get("num_workers", 0),
    )
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    data = data.to(device, "x", "edge_index")

    model = GraphSAGE(
        data.num_node_features,
        hidden_channels=embedding_dim,
        num_layers=config["num_layers"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    def train():
        model.train()

        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            h = model(batch.x, batch.edge_index)
            h_src = h[batch.edge_label_index[0]]
            h_dst = h[batch.edge_label_index[1]]
            pred = (h_src * h_dst).sum(dim=-1)
            loss = F.binary_cross_entropy_with_logits(pred, batch.edge_label)
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * pred.size(0)

        return total_loss / data.num_nodes

    @torch.no_grad()
    def test():
        test_start = time.perf_counter()
        z = infer_embeddings(model=model, data=data, loader=test_loader, embedding_dim=embedding_dim)
        eval_embed_time = time.perf_counter() - test_start
        prep_start = time.perf_counter()

        if data.y is None:
            X_train, y_train, X_val, y_val = prepare_link_prediction_data(
                data=data,
                z=z,
                seed=config.get("train_negative_seed", seed),
            )
        else:
            X_train, y_train, X_val, y_val = (
                z[data.train_mask],
                data.y[data.train_mask],
                z[data.val_mask],
                data.y[data.val_mask],
            )
        prep_time = time.perf_counter() - prep_start

        clf = LogisticRegression()
        test_clf_start = time.perf_counter()
        clf.fit(X_train, y_train)
        test_clf_time = time.perf_counter() - test_clf_start
        print(
            f"Test timing | embedding_eval: {eval_embed_time:.2f}s, "
            f"data_prep: {prep_time:.2f}s, classifier_fit_eval: {test_clf_time:.2f}s, "
            f"total_test: {eval_embed_time + prep_time + test_clf_time:.2f}s"
        )

        return clf.score(X_val, y_val)

    best_loss = np.inf
    patience = config.get("patience", 10)
    min_epochs = config.get("min_epochs", 10)
    rel_delta_factor = config.get("early_stopping_rel_delta", 1e-3)
    patience_counter = 0

    for epoch in range(1, config["epochs"] + 1):
        loss = train()

        prev_best_loss = best_loss
        if np.isinf(prev_best_loss):
            required_delta = 0.0
            meaningful_improvement = True
        else:
            required_delta = rel_delta_factor * abs(prev_best_loss)
            meaningful_improvement = (prev_best_loss - loss) >= required_delta

        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), save_path)

        if epoch <= min_epochs:
            patience_counter = 0
        elif meaningful_improvement:
            patience_counter = 0
        else:
            patience_counter += 1

        delta_actual = np.inf if np.isinf(prev_best_loss) else prev_best_loss - loss
        print(
            f"Epoch {epoch:02d}, Loss: {loss:.4f}, "
            f"Delta: {delta_actual:.6f}, Required Delta: {required_delta:.6f}, "
            f"Patience: {patience_counter}/{patience}"
        )

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch:02d}.")
            break

    model.load_state_dict(torch.load(save_path, weights_only=True))

    val_acc = test()
    print(f"Validation accuracy is {val_acc:.4f}.")

    if embedding_path is not None:
        model.load_state_dict(torch.load(save_path, map_location=device))
        embedding = infer_embeddings(model=model, data=data, loader=test_loader, embedding_dim=embedding_dim)
        np.save(embedding_path, embedding.numpy())

    if not keep_checkpoints:
        os.remove(save_path)

    return val_acc
