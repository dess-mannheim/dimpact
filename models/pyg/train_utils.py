import torch
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling

from typing import Tuple


def prepare_link_prediction_data(
    data: Data, z: torch.tensor
) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
    pos_train_edges = data.edge_index.cpu()

    if hasattr(data, "val_edges"):

        neg_train_edges = negative_sampling(
            edge_index=data.edge_index, num_neg_samples=data.edge_index.size()[1]
        ).cpu()
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

        neg_train_edges = negative_sampling(
            edge_index=data.edge_index, num_neg_samples=pos_train_edges.size()[1]
        ).cpu()

        neg_val_edges = torch.t(data.neg_edges["val"]).cpu()
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
