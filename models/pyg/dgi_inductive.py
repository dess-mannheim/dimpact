import numpy as np
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


def train_model(
    dataset: Data, embedding_dim: int, config: Dict, save_path: str, seed: int, embedding_path: str = None
) -> float:
    seed_everything(seed)
    data = dataset.to(device, "x", "edge_index")

    train_loader = NeighborLoader(
        data,
        num_neighbors=config["num_neighbors"],
        batch_size=config["batch_size"],
        shuffle=config["shuffle"],
        num_workers=config["num_workers"],
    )

    test_loader = NeighborLoader(
        data,
        num_neighbors=config["num_neighbors"],
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

    def train():
        model.train()
        total_loss = total_examples = 0
        for batch in tqdm(train_loader):
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
        model.eval()
        zs = []
        for batch in tqdm(test_loader, desc="Evaluating"):
            pos_z, _, _ = model(batch.x, batch.edge_index, batch.batch_size)
            zs.append(pos_z.cpu())
        z = torch.cat(zs, dim=0)

        if data.y is None:
            X_train, y_train, X_val, y_val = prepare_link_prediction_data(data, z)
        else:
            X_train, y_train, X_val, y_val = (
                z[data.train_mask],
                data.y[data.train_mask],
                z[data.val_mask],
                data.y[data.val_mask],
            )

        acc = model.test(X_train, y_train, X_val, y_val, max_iter=10000)
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
            print(f"Early stopping at epoch {epoch:02d}")
            break

    val_acc = test()
    print(f"Validation Accuracy is {val_acc:.4f}")

    if embedding_path is not None:
        model.load_state_dict(torch.load(save_path, map_location=device))

        # Set the model to evaluation mode
        model.eval()

        # Get the embedding vectors from the model
        embedding, _, _ = model(data.x.to(device), data.edge_index.to(device), batch_size=data.num_nodes)
        np.save(embedding_path, embedding.cpu().detach().numpy())

    return val_acc
