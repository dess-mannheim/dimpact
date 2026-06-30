from scipy.io import loadmat

import torch
import torch_geometric
from torch_geometric import EdgeIndex
from torch_geometric.data import Data, InMemoryDataset, download_url
from torch_geometric.utils import negative_sampling
from paths_globals import *


def convert_coo_matrix(mat):
    indices = torch.tensor([mat.row, mat.col], dtype=torch.long)
    values = torch.tensor(mat.data, dtype=torch.float32)
    shape = torch.Size(mat.shape)

    return torch.sparse_coo_tensor(indices, values, shape)


class CoAuthor(InMemoryDataset):

    urls = ["http://tsitsul.in/pub/academic_coa_2014.mat", "http://tsitsul.in/pub/academic_coa_full.mat"]

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, force_reload=False):
        super().__init__(root, transform, pre_transform, pre_filter, force_reload=force_reload)
        self.load(self.processed_paths[0])
        # For PyG<2.4:
        # self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["academic_coa_2014.mat", "academic_coa_full.mat"]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        # Download to `self.raw_dir`.
        download_url(self.urls[0], self.raw_dir)
        download_url(self.urls[1], self.raw_dir)

    def process(self):

        adj_2014 = loadmat(self.raw_paths[0])["network"].tocoo()
        adj_2016 = loadmat(self.raw_paths[1])["network"].tocoo()

        adj_delta = adj_2016 - adj_2014
        n_nodes = adj_2016.shape[0]

        edge_index = EdgeIndex(convert_coo_matrix(adj_2014).coalesce()).as_tensor().to(torch.long)
        edges_2016 = EdgeIndex(convert_coo_matrix(adj_2016).coalesce())

        edges_delta = EdgeIndex(convert_coo_matrix(adj_delta.tocoo()).coalesce()).as_tensor()
        n_test_edges = edges_delta.size()[1]

        torch_geometric.seed_everything(EXPERIMENTS_DEFAULT_SEED)
        neg_edges = negative_sampling(edges_2016.as_tensor(), num_neg_samples=n_test_edges).t()
        perm = torch.randperm(n_test_edges)
        edges_delta = edges_delta.t()[perm]
        split_index = int(n_test_edges / 2)

        val_edges = {"edge": edges_delta[:split_index], "edge_neg": neg_edges[:split_index]}
        test_edges = {"edge": edges_delta[split_index:], "edge_neg": neg_edges[split_index:]}

        x = torch.eye(n_nodes)
        data = Data(x=x, edge_index=edge_index, val_edges=val_edges, test_edges=test_edges)

        # Read data into huge `Data` list.
        data_list = [data]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])
