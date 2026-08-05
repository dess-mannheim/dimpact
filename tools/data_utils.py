import random
import pickle
import json

import networkx as nx

import pandas as pd
from sklearn.preprocessing import LabelEncoder

import torch

from torch_geometric.data import Data
from torch_geometric.datasets import FacebookPagePage, Planetoid, AttributedGraphDataset
from torch_geometric.utils import to_networkx, negative_sampling

from data.ogbl_ddi.dataset import OGBL_DDI
from data.coauthor.dataset import CoAuthor
from data.synthetic_dataset import SyntheticDataset
from paths_globals import *


def transform_labels(data):
    # Extract the labels and convert to a NumPy array
    labels = data.y.numpy()

    # Convert each 121-length label array into a string to represent it uniquely
    labels_as_strings = np.array(["".join(map(str, label)) for label in labels])

    # Initialize LabelEncoder
    label_encoder = LabelEncoder()

    # Fit and transform the unique string labels
    integer_encoded_labels = label_encoder.fit_transform(labels_as_strings)

    # Convert the encoded labels back to a PyTorch tensor
    integer_encoded_labels = torch.tensor(integer_encoded_labels, dtype=torch.long)

    return Data(x=data.x, edge_index=data.edge_index, y=integer_encoded_labels)


def snowball_sampling(dataset, sampling_ratio, seed):
    """
    Samples a subgraph from the dataset using snowball sampling until the desired
    number of nodes is reached. If disconnected, selects a new random seed node.

    Parameters:
    - dataset: PyG dataset to sample from
    - sampling_ratio: float, ratio of nodes to sample (0 < sampling_ratio <= 1)
    - seed: int, random seed to use

    Returns:
    - sampled_data: PyG dataset, sampled subgraph
    """
    random.seed(seed)

    # Convert the PyG dataset to a NetworkX graph
    G = to_networkx(dataset, to_undirected=True)
    num_nodes = dataset.num_nodes
    print(sampling_ratio)
    print(num_nodes)
    num_sampled_nodes = int(sampling_ratio * num_nodes)

    sampled_nodes = set()
    visited_nodes = set()  # To avoid reselecting previously chosen disconnected nodes

    while len(sampled_nodes) < num_sampled_nodes:
        # Randomly choose a start node that hasn’t been fully processed
        unvisited_nodes = set(G.nodes) - visited_nodes
        if not unvisited_nodes:  # If no unvisited nodes are left
            break
        start_node = random.choice(list(unvisited_nodes))

        # Initialize a wave starting from the start_node
        current_wave_nodes = {start_node}
        visited_nodes.add(start_node)  # Mark start_node as visited

        while current_wave_nodes and len(sampled_nodes) < num_sampled_nodes:
            next_wave_nodes = set()
            for node in current_wave_nodes:
                neighbors = list(G.neighbors(node))
                next_wave_nodes.update(neighbors)
            sampled_nodes.update(current_wave_nodes)
            current_wave_nodes = next_wave_nodes - sampled_nodes  # Exclude already sampled nodes

    # Convert the set of sampled nodes to a tensor
    sampled_indices = torch.tensor(list(sampled_nodes), dtype=torch.long)

    # Create a mask for the sampled nodes
    node_mask = torch.zeros(num_nodes, dtype=torch.bool)
    node_mask[sampled_indices] = True

    # Filter the edge index to only include edges where both nodes are in the sampled subset
    edge_mask = node_mask[dataset.edge_index[0]] & node_mask[dataset.edge_index[1]]
    sampled_edge_index = dataset.edge_index[:, edge_mask]

    # Re-index the nodes in the edge_index to ensure consecutive indexing in the subgraph
    sampled_indices_map = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(sampled_indices)}
    reindexed_edge_index = torch.tensor(
        [[sampled_indices_map[old_idx.item()] for old_idx in edge] for edge in sampled_edge_index],
        dtype=torch.long,
    )

    # Filter edge attributes if they exist
    sampled_data = dataset.clone()
    sampled_data.x = dataset.x[sampled_indices]
    sampled_data.y = dataset.y[sampled_indices]
    sampled_data.edge_index = reindexed_edge_index
    sampled_data.num_nodes = len(sampled_nodes)  # Update num_nodes attribute to reflect the new size

    if dataset.edge_attr is not None:
        sampled_data.edge_attr = dataset.edge_attr[edge_mask]

    return sampled_data


def create_downstream_df(data, dataset_params: Dict[str, Any]):
    def _as_edge_pairs(edge_tensor) -> np.ndarray:
        """Normalize edge arrays to shape (n_edges, 2)."""
        edges = edge_tensor.numpy() if hasattr(edge_tensor, "numpy") else np.asarray(edge_tensor)
        if edges.ndim != 2:
            raise ValueError(f"Expected 2D edge array, got shape {edges.shape}.")
        if edges.shape[1] == 2:
            return edges
        if edges.shape[0] == 2:
            return edges.T
        raise ValueError(f"Unsupported edge array shape {edges.shape}; expected (n,2) or (2,n).")

    dataset_name = dataset_params[CONFIG_DATASET_NAME_KEY]
    task = DATASET_TASK_DICT[dataset_name]
    if task == NODE_CLASSIFICATION:
        y = data.y.numpy()
        class_labels = sorted(int(label) for label in np.unique(y).tolist())
        num_classes = len(class_labels)
        categories = pd.Series(
            pd.Categorical(
                [DOWNSTREAM_TASK_DATA_TRAIN_CATEGORY] * len(y),
                categories=[
                    DOWNSTREAM_TASK_DATA_TRAIN_CATEGORY,
                    DOWNSTREAM_TASK_DATA_VAL_CATEGORY,
                    DOWNSTREAM_TASK_DATA_TEST_CATEGORY,
                ],
                ordered=False,
            )
        )
        categories.loc[data.val_mask.numpy()] = DOWNSTREAM_TASK_DATA_VAL_CATEGORY
        categories.loc[data.test_mask.numpy()] = DOWNSTREAM_TASK_DATA_TEST_CATEGORY
        train_df = pd.DataFrame(
            np.vstack((categories, y)).T,
            columns=DOWNSTREAM_TASK_DATA_NC_COLUMN_NAMES,
            index=torch.unique(data.edge_index, sorted=True).numpy(),
        )
    elif task == LINK_PREDICTION:
        class_labels = [0, 1]
        num_classes = 2
        edge_data = []

        pos_val_edges = _as_edge_pairs(data.val_edges["edge"])
        for e in pos_val_edges:
            edge_data.append([e[0], e[1], DOWNSTREAM_TASK_DATA_VAL_CATEGORY, 1])
        neg_val_edges = _as_edge_pairs(data.val_edges["edge_neg"])
        for e in neg_val_edges:
            edge_data.append([e[0], e[1], DOWNSTREAM_TASK_DATA_VAL_CATEGORY, 0])
        pos_test_edges = _as_edge_pairs(data.test_edges["edge"])
        for e in pos_test_edges:
            edge_data.append([e[0], e[1], DOWNSTREAM_TASK_DATA_TEST_CATEGORY, 1])
        neg_test_edges = _as_edge_pairs(data.test_edges["edge_neg"])
        for e in neg_test_edges:
            edge_data.append([e[0], e[1], DOWNSTREAM_TASK_DATA_TEST_CATEGORY, 0])

        train_df = pd.DataFrame(
            edge_data, columns=DOWNSTREAM_TASK_DATA_LP_COLUMN_NAMES, index=np.arange(len(edge_data))
        )
        for split_name in [DOWNSTREAM_TASK_DATA_VAL_CATEGORY, DOWNSTREAM_TASK_DATA_TEST_CATEGORY]:
            split_labels = train_df.loc[train_df[DOWNSTREAM_TASK_DATA_SPLIT_COL_KEY] == split_name, DOWNSTREAM_TASK_DATA_LABEL_COL_KEY]
            if not ({0, 1}.issubset(set(split_labels.unique().tolist()))):
                raise RuntimeError(f"LP split '{split_name}' must contain both positive and negative edges.")
    else:
        class_labels = [0, 1]
        num_classes = 2
        edge_data = []
        pos_train_edges = data.edge_index[:, data.train_mask].numpy().T

        for e in pos_train_edges:
            edge_data.append([e[0], e[1], DOWNSTREAM_TASK_DATA_TRAIN_CATEGORY, 1])

        pos_val_edges = _as_edge_pairs(data.edge_index[:, data.val_mask])
        for e in pos_val_edges:
            edge_data.append([e[0], e[1], DOWNSTREAM_TASK_DATA_VAL_CATEGORY, 1])
        neg_val_edges = _as_edge_pairs(data.neg_edges["val"])
        for e in neg_val_edges:
            edge_data.append([e[0], e[1], DOWNSTREAM_TASK_DATA_VAL_CATEGORY, 0])

        pos_test_edges = _as_edge_pairs(data.edge_index[:, data.test_mask])
        for e in pos_test_edges:
            edge_data.append([e[0], e[1], DOWNSTREAM_TASK_DATA_TEST_CATEGORY, 1])
        neg_test_edges = _as_edge_pairs(data.neg_edges["test"])
        for e in neg_test_edges:
            edge_data.append([e[0], e[1], DOWNSTREAM_TASK_DATA_TEST_CATEGORY, 0])

        train_df = pd.DataFrame(
            edge_data, columns=DOWNSTREAM_TASK_DATA_LP_COLUMN_NAMES, index=np.arange(len(edge_data))
        )
        for split_name in [DOWNSTREAM_TASK_DATA_VAL_CATEGORY, DOWNSTREAM_TASK_DATA_TEST_CATEGORY]:
            split_labels = train_df.loc[train_df[DOWNSTREAM_TASK_DATA_SPLIT_COL_KEY] == split_name, DOWNSTREAM_TASK_DATA_LABEL_COL_KEY]
            if not ({0, 1}.issubset(set(split_labels.unique().tolist()))):
                raise RuntimeError(f"LP split '{split_name}' must contain both positive and negative edges.")

    dataset_src_dir = BUILD_DATASET_SRC_DIR(dataset_params)
    data_file_path = os.path.join(dataset_src_dir, DOWNSTREAM_TASK_DATA_FILE_NAME)
    train_df.to_csv(data_file_path, index=True)

    metadata = {
        CONFIG_DATASET_NAME_KEY: dataset_name,
        DATASET_METADATA_TASK_KEY: task,
        DATASET_METADATA_NUM_CLASSES_KEY: num_classes,
        DATASET_METADATA_CLASS_LABELS_KEY: class_labels,
    }
    metadata_file_path = os.path.join(dataset_src_dir, DOWNSTREAM_METADATA_JSON_FILE_NAME)
    with open(metadata_file_path, "w") as f:
        json.dump(metadata, f, indent=2)


def sample_dataset(dataset, sampling_ratio, dataset_name, seed):
    """
    Samples a dataset using a specified sampling strategy and saves it for reuse.

    Parameters:
    - dataset: PyG dataset to sample from
    - sampling_ratio: float, ratio of nodes to sample (0 < sampling_ratio <= 1)
    - dataset_name: str, name of the dataset
    - seed: int, sed to use when sampling dataset)

    Returns:
    - sampled_dataset: PyG dataset, sampled subgraph
    """
    # Define a path to save/reuse the sampled graph based on the sampling type and ratio
    dataset_dir = osp.join(DATA_DIR, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    sample_file_path = osp.join(dataset_dir, SAMPLED_DATA_FILE_NAME(sampling_ratio, seed))

    # Check if the sampled dataset already exists
    if osp.exists(sample_file_path):
        print(f"Loading sampled dataset from {sample_file_path}")
        with open(sample_file_path, "rb") as f:
            return pickle.load(f)

    # Apply snowball sampling
    sampled_dataset = snowball_sampling(dataset, sampling_ratio, seed)

    # Save the sampled dataset for future use
    with open(sample_file_path, "wb") as f:
        pickle.dump(sampled_dataset, f)
    print(f"Sampled dataset saved to {sample_file_path}")

    return sampled_dataset




def _write_edge_list_file(dataset: Data, edge_list_path: str) -> None:
    edge_list = dataset.edge_index.t().tolist()
    with open(edge_list_path, "w") as f:
        for edge in edge_list:
            f.write(f"{edge[0]} {edge[1]}\n")


def load_dataset(dataset_params: Dict[str, Any], reload: bool = False) -> Tuple[Data, str]:
    dataset_name = dataset_params[CONFIG_DATASET_NAME_KEY]

    if dataset_name in EMPIRICAL_DATASET_LIST:
        return load_empirical_dataset(dataset_name, dataset_params, reload)

    else:
        return load_synthetic_dataset(dataset_name, dataset_params, reload)


def load_empirical_dataset(dataset_name: DATASET, dataset_params: Dict[str, Any], reload: bool = False) -> Tuple[Data, str]:

    dataset_dir = osp.join(DATA_DIR, dataset_name)
    if dataset_name == FACEBOOK:
        dataset = FacebookPagePage(os.path.join(DATA_DIR, FACEBOOK), force_reload=reload)
    elif dataset_name == COAUTHOR:
        dataset = CoAuthor(os.path.join(DATA_DIR, dataset_name), force_reload=reload)
    elif dataset_name in PLANETOID_DATASETS:
        dataset = Planetoid(DATA_DIR, name=dataset_name, force_reload=reload)
    elif dataset_name == DDI:
        dataset = OGBL_DDI(root=osp.join(DATA_DIR, DDI), force_reload=reload)
    else:
        dataset = AttributedGraphDataset(root=DATA_DIR, name=dataset_name, force_reload=reload)

    if dataset_name == DDI:
        split_edge = dataset.get_edge_split()
        dataset = dataset[0]
        dataset.val_edges = split_edge["valid"]
        dataset.test_edges = split_edge["test"]
        dataset.x = torch.eye(dataset.num_nodes)
    else:
        dataset = dataset[0]

    # Convert to 1D if multi-dimensional
    if dataset.y is not None:
        if dataset.y.dim() > 1:
            dataset.y = dataset.y.argmax(dim=-1)  # Convert to 1D tensor with primary label

    edgelist_file_path = os.path.join(dataset_dir, DATA_EDGE_LIST_DEFAULT_FILE_NAME)

    # Apply sampling if specified
    if dataset_params[CONFIG_DATA_SUBSAMPLING_INDICATOR_KEY]:
        sampling_ratio = dataset_params[CONFIG_DATA_SUBSAMPLING_RATIO_KEY]
        seed = dataset_params[CONFIG_DATA_SAMPLING_SEED_KEY]

        dataset_dir = osp.join(dataset_dir, SUBSAMPLED_DATA_DIR_NAME)
        os.makedirs(dataset_dir, exist_ok=True)
        dataset_file_path = osp.join(dataset_dir, SAMPLED_DATA_FILE_NAME(sampling_ratio, seed))

        # Check if the sampled dataset already exists
        sampled_edgelist_path = osp.join(dataset_dir, SAMPLED_DATA_FILE_NAME(sampling_ratio, seed, is_edge_list=True))

        if osp.exists(dataset_file_path):
            print(f"Loading sampled dataset from {dataset_file_path}")
            with open(dataset_file_path, "rb") as f:
                sampled_dataset = pickle.load(f)

            # Backward compatibility: old cache may store only Data, future cache can also store tuples.
            if isinstance(sampled_dataset, tuple):
                sampled_dataset = sampled_dataset[0]

            if not osp.isfile(sampled_edgelist_path):
                _write_edge_list_file(sampled_dataset, sampled_edgelist_path)

            return sampled_dataset, sampled_edgelist_path

        # Apply snowball sampling
        sampled_dataset = snowball_sampling(dataset, sampling_ratio, seed)
        sampled_dataset = create_nc_training_masks(sampled_dataset)

        # Save the sampled dataset for future use
        with open(dataset_file_path, "wb") as f:
            pickle.dump(sampled_dataset, f)
        _write_edge_list_file(sampled_dataset, sampled_edgelist_path)
        print(f"Sampled dataset saved to {dataset_file_path}")

        return sampled_dataset, sampled_edgelist_path

    elif not hasattr(dataset, "train_mask"):
        dataset = create_nc_training_masks(dataset)

    downstream_df_path = os.path.join(dataset_dir, DOWNSTREAM_TASK_DATA_FILE_NAME)
    if not osp.isfile(downstream_df_path) or reload:
        create_downstream_df(dataset, dataset_params)

    # Save edge list to a text file for the existing pipeline.
    if not osp.isfile(edgelist_file_path) or reload:
        _write_edge_list_file(dataset, edgelist_file_path)

    return dataset, edgelist_file_path


def load_synthetic_dataset(dataset_name: DATASET, dataset_params: Dict[str, Any], reload: bool = False) -> Tuple[Data, str]:
    dataset_dir = BUILD_DATASET_SRC_DIR(dataset_params)
    num_nodes = dataset_params[CONFIG_SYNTH_DATA_NUM_NODES_KEY]
    density = dataset_params[CONFIG_SYNTH_DATA_DENSITY_KEY]
    seed = dataset_params[CONFIG_DATA_SAMPLING_SEED_KEY]

    edgelist_file_path = osp.join(dataset_dir, DATA_EDGE_LIST_DEFAULT_FILE_NAME)

    # Check if dataset already exists
    if osp.exists(edgelist_file_path):
        # Load the dataset from file
        with open(edgelist_file_path, "rb") as f:
            G = nx.read_edgelist(edgelist_file_path)
    else:

        # Generate synthetic graph
        if dataset_name == BARABASI_ALBERT:
            m = round(density * num_nodes / 2)
            G = nx.barabasi_albert_graph(num_nodes, m, seed=seed)
        elif dataset_name == WATTS_STROGATZ:
            # In NetworkX WS graphs, odd k effectively behaves like k-1 for edge count.
            # Choose the closest valid even k to best match the requested density.
            raw_k = density * (num_nodes - 1)
            max_even_k = (num_nodes - 1) if (num_nodes - 1) % 2 == 0 else (num_nodes - 2)
            lower_even_k = int(np.floor(raw_k / 2.0) * 2)
            upper_even_k = lower_even_k + 2
            candidates = []
            for candidate_k in [lower_even_k, upper_even_k]:
                if 0 <= candidate_k <= max_even_k:
                    candidates.append(candidate_k)
            if len(candidates) == 0:
                k = 0
            else:
                # Tie-break toward smaller k to minimize deviations from legacy behavior.
                k = min(candidates, key=lambda candidate_k: (abs(raw_k - candidate_k), candidate_k))
            G = nx.watts_strogatz_graph(num_nodes, k, p=WATTS_STROGATZ_DEFAULT_REWIRING_PROBABILITY, seed=seed)
        else:
            raise ValueError("Unknown dataset_name for synthetic graphs.")

        # Save edge list to a text file for your existing pipeline
    with open(edgelist_file_path, "w") as f:
        for edge in G.edges():
            f.write(f"{edge[0]} {edge[1]}\n")

    synthetic_dataset = SyntheticDataset(root=dataset_dir)
    dataset: Data = synthetic_dataset[0]

    downstream_df_path = os.path.join(dataset_dir, DOWNSTREAM_TASK_DATA_FILE_NAME)
    if not osp.isfile(downstream_df_path) or reload:
        create_downstream_df(dataset, dataset_params)

    return dataset, edgelist_file_path


def create_nc_training_masks(dataset):
    num_nodes = dataset.num_nodes
    num_train = int(num_nodes * NODE_CLASSIFICATION_DEFAULT_TRAINING_RATIO)
    num_val = int(num_nodes * NODE_CLASSIFICATION_DEFAULT_VALIDATION_RATIO)

    torch.manual_seed(EXPERIMENTS_DEFAULT_SEED)
    indices = torch.randperm(num_nodes)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[indices[:num_train]] = True
    val_mask[indices[num_train : num_train + num_val]] = True
    test_mask[indices[num_train + num_val :]] = True

    dataset.train_mask = train_mask
    dataset.val_mask = val_mask
    dataset.test_mask = test_mask

    return dataset
