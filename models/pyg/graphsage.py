import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric import seed_everything
from torch_geometric.data import Data
from sklearn.linear_model import LogisticRegression
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import GraphSAGE
from typing import Dict

from models.pyg.train_utils import prepare_link_prediction_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(
    dataset: Data, embedding_dim: int, config: Dict, save_path: str, seed: int, embedding_path: str = None
) -> float:

    seed_everything(seed)
    config["num_neighbors_list"] = [config["num_neighbors"]] * config["num_layers"]
    data = dataset
    train_loader = LinkNeighborLoader(
        data,
        batch_size=config["batch_size"],
        shuffle=config["shuffle"],
        neg_sampling_ratio=config["neg_sampling_ratio"],
        num_neighbors=config["num_neighbors_list"],
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
        model.eval()
        z = model(data.x, data.edge_index).cpu()

        if data.y is None:
            X_train, y_train, X_val, y_val = prepare_link_prediction_data(data, z)
        else:
            X_train, y_train, X_val, y_val = (
                z[data.train_mask],
                data.y[data.train_mask],
                z[data.val_mask],
                data.y[data.val_mask],
            )

        clf = LogisticRegression()
        clf.fit(X_train, y_train)

        return clf.score(X_val, y_val)

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

    model.load_state_dict(torch.load(save_path, weights_only=True))

    val_acc = test()
    print(f"Validation accuracy is {val_acc:.4f}.")

    if embedding_path is not None:
        model.load_state_dict(torch.load(save_path, map_location=device))

        # Set the model to evaluation mode
        model.eval()

        # Get the embedding vectors from the model
        with torch.no_grad():
            embedding = model(data.x.to(device), data.edge_index.to(device))

        np.save(embedding_path, embedding.cpu().detach().numpy())

    return val_acc
