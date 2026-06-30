import sys
import torch
import numpy as np
from torch_geometric import seed_everything
from torch_geometric.data import Data
from torch_geometric.nn import Node2Vec
from torch_geometric.utils import negative_sampling
from typing import Dict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(
    dataset: Data, embedding_dim: int, config: Dict, save_path: str, seed: int, embedding_path: str = None
) -> float:
    seed_everything(seed)
    data = dataset
    model = Node2Vec(
        data.edge_index,
        embedding_dim=embedding_dim,
        walk_length=config["walk_length"],
        context_size=config["context_size"],
        walks_per_node=config["walks_per_node"],
        num_negative_samples=config["num_negative_samples"],
        p=config["p"],
        q=config["q"],
        sparse=bool(config["sparse"]),
    ).to(device)

    num_workers = 1 if sys.platform == "linux" else 0
    loader = model.loader(batch_size=config["batch_size"], shuffle=config["shuffle"], num_workers=num_workers)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=config["learning_rate"])

    def train():
        model.train()
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    @torch.no_grad()
    def test():
        model.eval()
        z = model()

        if data.y is None:
            pos_train_edges = data.edge_index
            neg_train_edges = negative_sampling(edge_index=data.edge_index, num_neg_samples=data.edge_index.size()[1])
            pos_val_edges = torch.t(data.val_edges["edge"])
            neg_val_edges = torch.t(data.val_edges["edge_neg"])
            X_train = torch.concatenate(
                (
                    z[torch.cat((pos_train_edges[0], neg_train_edges[0]), dim=0)],
                    z[torch.cat((pos_train_edges[1], neg_train_edges[1]), dim=0)],
                ),
                dim=1,
            )
            y_train = torch.cat(
                (torch.ones(pos_train_edges.size()[1]), torch.zeros(neg_train_edges.size()[1])), dim=0
            )
            perm = torch.randperm(X_train.size()[0])
            X_train = X_train[perm]
            y_train = y_train[perm]
            X_val = torch.concatenate(
                (
                    z[torch.cat((pos_val_edges[0], neg_val_edges[0]), dim=0)],
                    z[torch.cat((pos_val_edges[1], neg_val_edges[1]), dim=0)],
                ),
                dim=1,
            )
            y_val = torch.cat((torch.ones(pos_val_edges.size()[1]), torch.zeros(neg_val_edges.size()[1])), dim=0)

        else:
            X_train, y_train, X_val, y_val = (
                z[data.train_mask],
                data.y[data.train_mask],
                z[data.val_mask],
                data.y[data.val_mask],
            )

        acc = model.test(
            train_z=X_train,
            train_y=y_train,
            test_z=X_val,
            test_y=y_val,
            max_iter=150,
        )
        return acc

    best_loss = np.inf
    patience = config.get("patience", 10)  # Default patience of 10 epochs
    patience_counter = 0

    for epoch in range(1, config["epochs"] + 1):
        loss = train()

        if loss < best_loss:
            best_loss = loss
            patience_counter = 0  # Reset patience counter
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1

        print(f"Epoch {epoch:02d}, Loss: {loss:.4f}")

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch:02d}.")
            break

    val_acc = test()
    print(f"Validation Accuracy is {val_acc:.4f}")

    if embedding_path is not None:
        model.load_state_dict(torch.load(save_path, map_location=device))

        # Set the model to evaluation mode
        model.eval()

        # Get the embedding vectors from the model
        with torch.no_grad():
            embedding = model()

        np.save(embedding_path, embedding.cpu().numpy())

    return val_acc
