import os
import numpy as np
import time
import torch
from tqdm import tqdm
from torch_geometric import seed_everything
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import DeepGraphInfomax, SAGEConv
from typing import Dict

from models.pyg.train_utils import prepare_link_prediction_data

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers):
        super().__init__()
        layers = [SAGEConv(in_channels, hidden_channels)]
        for i in range(num_layers - 1):
            layers.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs = torch.nn.ModuleList(layers)
        self.activations = torch.nn.ModuleList([torch.nn.PReLU(hidden_channels)] * num_layers)

    def forward(self, x, edge_index, batch_size):
        for conv, act in zip(self.convs, self.activations):
            x = conv(x, edge_index)
            x = act(x)
        return x[:batch_size]


def corruption(x, edge_index, batch_size):
    return x[torch.randperm(x.size(0))], edge_index, batch_size


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
    model: DeepGraphInfomax,
    data: Data,
    loader: NeighborLoader,
    embedding_dim: int,
) -> torch.Tensor:
    model.eval()
    z = torch.empty((data.num_nodes, embedding_dim), dtype=torch.float32)
    for batch in tqdm(loader, desc="Evaluating"):
        batch = batch.to(device)
        pos_z, _, _ = model(batch.x, batch.edge_index, batch.batch_size)
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
    data = dataset
    num_layers = config["num_layers"]
    num_neighbors_list = resolve_num_neighbors(config=config, num_layers=num_layers)

    train_loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors_list,
        batch_size=config["batch_size"],
        shuffle=config["shuffle"],
        num_workers=config["num_workers"],
    )

    test_loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors_list,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
    )

    model = DeepGraphInfomax(
        hidden_channels=embedding_dim,
        encoder=Encoder(
            in_channels=dataset.num_features, hidden_channels=embedding_dim, num_layers=config["num_layers"]
        ),
        summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        corruption=corruption,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    eval_max_iter = config.get("eval_max_iter", 1000)

    def train():
        model.train()
        total_loss = total_examples = 0
        for batch in tqdm(train_loader):
            batch = batch.to(device)
            optimizer.zero_grad()
            pos_z, neg_z, summary = model(batch.x, batch.edge_index, batch.batch_size)
            loss = model.loss(pos_z, neg_z, summary)
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * pos_z.size(0)
            total_examples += pos_z.size(0)
        return total_loss / total_examples

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

        test_clf_start = time.perf_counter()
        acc = model.test(X_train, y_train, X_val, y_val, max_iter=eval_max_iter)
        test_clf_time = time.perf_counter() - test_clf_start
        print(
            f"Test timing | embedding_eval: {eval_embed_time:.2f}s, "
            f"data_prep: {prep_time:.2f}s, classifier_fit_eval: {test_clf_time:.2f}s, "
            f"total_test: {eval_embed_time + prep_time + test_clf_time:.2f}s"
        )
        return acc

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
            print(f"Early stopping at epoch {epoch:02d}")
            break

    val_acc = test()
    print(f"Validation Accuracy is {val_acc:.4f}")

    if embedding_path is not None:
        model.load_state_dict(torch.load(save_path, map_location=device))
        embedding = infer_embeddings(model=model, data=data, loader=test_loader, embedding_dim=embedding_dim)
        np.save(embedding_path, embedding.numpy())

    if not keep_checkpoints:
        os.remove(save_path)

    return val_acc
