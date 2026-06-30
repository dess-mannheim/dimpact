import torch
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling

from typing import Tuple


def _edges_to_2xn(edges: torch.Tensor) -> torch.Tensor:
    """Normalize edge tensors to shape (2, n_edges)."""
    if edges.dim() != 2:
        raise ValueError(f"Edge tensor must be 2D, got shape {tuple(edges.shape)}.")
    if edges.size(0) == 2:
        return edges
    if edges.size(1) == 2:
        return edges.t()
    raise ValueError(f"Unsupported edge tensor shape {tuple(edges.shape)}.")


def _sample_negative_edges(edge_index: torch.Tensor, num_neg_samples: int, seed: int | None = None) -> torch.Tensor:
    """Sample negative edges, optionally with a deterministic seed."""
    if seed is None:
        return negative_sampling(edge_index=edge_index, num_neg_samples=num_neg_samples).cpu()

    with torch.random.fork_rng():
        torch.manual_seed(seed)
        return negative_sampling(edge_index=edge_index, num_neg_samples=num_neg_samples).cpu()


def prepare_link_prediction_data(
    data: Data,
    z: torch.tensor,
    seed: int | None = None,
) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
    pos_train_edges = data.edge_index.cpu()

    if hasattr(data, "val_edges"):

        neg_train_edges = _sample_negative_edges(
            edge_index=data.edge_index,
            num_neg_samples=data.edge_index.size()[1],
            seed=seed,
        )
        pos_val_edges = torch.t(data.val_edges["edge"]).cpu()
        neg_val_edges = torch.t(data.val_edges["edge_neg"]).cpu()
        X_train = torch.concatenate(
            (
                z[torch.cat((pos_train_edges[0], neg_train_edges[0]), dim=0)],
                z[torch.cat((pos_train_edges[1], neg_train_edges[1]), dim=0)],
            ),
            dim=1,
        )
        y_train = torch.cat((torch.ones(pos_train_edges.size()[1]), torch.zeros(neg_train_edges.size()[1])), dim=0)
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
        pos_edges = data.edge_index.cpu()
        pos_train_edges = pos_edges[:, data.train_mask]
        pos_val_edges = pos_edges[:, data.val_mask]

        neg_train_edges = _sample_negative_edges(
            edge_index=data.edge_index,
            num_neg_samples=pos_train_edges.size()[1],
            seed=seed,
        )

        neg_val_edges = _edges_to_2xn(data.neg_edges["val"].cpu())
        X_train = torch.concatenate(
            (
                z[torch.cat((pos_train_edges[0], neg_train_edges[0]), dim=0)],
                z[torch.cat((pos_train_edges[1], neg_train_edges[1]), dim=0)],
            ),
            dim=1,
        )
        y_train = torch.cat((torch.ones(pos_train_edges.size()[1]), torch.zeros(neg_train_edges.size()[1])), dim=0)
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

    return X_train, y_train, X_val, y_val
