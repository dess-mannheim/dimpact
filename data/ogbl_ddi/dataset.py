import os.path as osp
import fsspec

from ogb.utils.torch_util import replace_numpy_with_torchtensor
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data.data import BaseData
from typing import Type


class OGBL_DDI(InMemoryDataset):

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, force_reload=False):
        super().__init__(root, transform, pre_transform, pre_filter, force_reload=force_reload)
        self.load(self.processed_paths[0])
        # For PyG<2.4:
        # self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["geometric_data_processed.pt"]

    @property
    def processed_file_names(self):
        return ["geometric_data_processed.pt"]

    def load(self, path: str, data_cls: Type[BaseData] = Data) -> None:
        r"""Loads the dataset from the file path :obj:`path`."""
        with fsspec.open(path, "rb") as f:
            out = torch.load(f, weights_only=False)
        assert isinstance(out, tuple)
        assert len(out) == 2 or len(out) == 3
        if len(out) == 2:  # Backward compatibility.
            data, self.slices = out
        else:
            data, self.slices, data_cls = out

        if not isinstance(data, dict):  # Backward compatibility.
            self.data = data
        else:
            self.data = data_cls.from_dict(data)

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def get_edge_split(self):

        train = replace_numpy_with_torchtensor(
            torch.load(osp.join(self.processed_dir, "train.pt"), weights_only=False)
        )
        valid = replace_numpy_with_torchtensor(
            torch.load(osp.join(self.processed_dir, "valid.pt"), weights_only=False)
        )
        test = replace_numpy_with_torchtensor(torch.load(osp.join(self.processed_dir, "test.pt"), weights_only=False))

        return {"train": train, "valid": valid, "test": test}

    def process(self):

        self.load(self.raw_paths[0])

        # Read data into huge `Data` list.
        data_list = [self.data]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])
