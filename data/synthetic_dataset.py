import os.path as osp

import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import negative_sampling, to_undirected
from paths_globals import (
    EXPERIMENTS_DEFAULT_SEED,
    GRAPH_RECONSTRUCTION_DEFAULT_TRAINING_RATIO,
    GRAPH_RECONSTRUCTION_DEFAULT_VALIDATION_RATIO,
    DATA_EDGE_LIST_DEFAULT_FILE_NAME,
)


class SyntheticDataset(InMemoryDataset):

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, force_reload=False):
        super().__init__(root, transform, pre_transform, pre_filter, force_reload=force_reload)
        self.load(self.processed_paths[0])
        # For PyG<2.4:
        # self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):

        edgelist_path = osp.join(self.root, DATA_EDGE_LIST_DEFAULT_FILE_NAME)

        srcs = []
        dests = []
        nodes = set()
        with open(edgelist_path, "r") as f:
            for l in f.readlines():
                u, v = map(int, l.split())
                srcs.append(u)
                dests.append(v)
                nodes.update({u, v})

        num_nodes = len(nodes)

        edge_index = torch.tensor([srcs, dests], dtype=torch.long)
        edge_index = to_undirected(edge_index)
        num_edges = edge_index.size(1)

        num_train = int(num_edges * GRAPH_RECONSTRUCTION_DEFAULT_TRAINING_RATIO)
        num_val = int(num_edges * GRAPH_RECONSTRUCTION_DEFAULT_VALIDATION_RATIO)
        num_test = num_edges - num_train - num_val

        torch.manual_seed(EXPERIMENTS_DEFAULT_SEED)

        # train/val/test index for existing edges
        indices = torch.randperm(num_edges)
        train_mask = torch.zeros(num_edges, dtype=torch.bool)
        val_mask = torch.zeros(num_edges, dtype=torch.bool)
        test_mask = torch.zeros(num_edges, dtype=torch.bool)

        train_mask[indices[:num_train]] = True
        val_mask[indices[num_train : num_train + num_val]] = True
        test_mask[indices[num_train + num_val :]] = True

        # negative edges for validation/testing
        neg_val_edges = negative_sampling(edge_index=edge_index, num_neg_samples=num_val)
        neg_test_edges = negative_sampling(edge_index=edge_index, num_neg_samples=num_test)

        neg_edges_dict = {"val": neg_val_edges, "test": neg_test_edges}

        data = Data(
            edge_index=edge_index,
            x=torch.eye(num_nodes),
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            neg_edges=neg_edges_dict,
        )

        # Read data into huge `Data` list.
        data_list = [data]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])
