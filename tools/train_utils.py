"""MODULE WITH TRAINING UTILITY SCRIPTS WHICH ARE USED WHEN TRAINING EMBEDDINGS.
SHOULD ONLY INCLUDE FUNCTIONS WHICH ARE USABLE ACROSS ALL EMBEDDING METHODS, I.E.,
LIBRARIES SUCH AS PYTORCH (GEOMETRIC), TENSORFLOW, ETC, SHOULD BE AVOIDED"""

import json

import pandas as pd

from paths_globals import *
from typing import Union


def _edge_set_undirected(edges: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(sorted((int(u), int(v)))) for u, v in edges}


def _sample_negative_edges(
    all_edges: np.ndarray,
    num_samples: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Sample node pairs that are not present in ``all_edges`` (treating graph as undirected)."""
    all_nodes = np.unique(all_edges)
    edge_set = {tuple(sorted((int(u), int(v)))) for u, v in all_edges}

    negatives: list[tuple[int, int]] = []
    while len(negatives) < num_samples:
        src = int(all_nodes[rng.randint(0, len(all_nodes))])
        dst = int(all_nodes[rng.randint(0, len(all_nodes))])
        if src == dst:
            continue

        candidate = tuple(sorted((src, dst)))
        if candidate in edge_set:
            continue

        negatives.append((src, dst))

    return np.array(negatives, dtype=np.int64)


def prepare_link_prediction_data(
    downstream_df: pd.DataFrame,
    edge_list: str | List[Tuple[int, int]] | np.ndarray,
    embedding: np.ndarray,
    seed: int = EXPERIMENTS_DEFAULT_SEED,
    return_val_data: bool = False,
    return_test_data: bool = False,
) -> Union[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
]:

    if type(edge_list) == str:
        all_edges = pd.read_csv(edge_list, sep=" ", header=None).to_numpy()
    else:
        all_edges = np.array(edge_list)

    if DOWNSTREAM_TASK_DATA_TRAIN_CATEGORY in downstream_df.loc[:, DOWNSTREAM_TASK_DATA_SPLIT_COL_KEY]:
        pos_train_edges = downstream_df.loc[
            downstream_df.loc[:, DOWNSTREAM_TASK_DATA_SPLIT_COL_KEY] == DOWNSTREAM_TASK_DATA_TRAIN_CATEGORY
        ][[DOWNSTREAM_TASK_DATA_SRC_EDGE_COL_KEY, DOWNSTREAM_TASK_DATA_DEST_EDGE_COL_KEY]].to_numpy()

    else:
        pos_train_edges = all_edges

    rng = np.random.RandomState(seed)
    neg_train_edges = _sample_negative_edges(
        all_edges=all_edges,
        num_samples=len(pos_train_edges),
        rng=rng,
    )
    # sanity check: sampled negatives must be true non-edges
    all_edge_set = _edge_set_undirected(all_edges)
    neg_edge_set = _edge_set_undirected(neg_train_edges)
    if len(all_edge_set.intersection(neg_edge_set)) > 0:
        raise RuntimeError("Negative LP training samples contain true graph edges.")

    train_edges = np.vstack((pos_train_edges, neg_train_edges))
    shuffle_idx = np.arange(len(train_edges), dtype=int)
    np.random.RandomState(seed).shuffle(shuffle_idx)

    X_train = np.hstack((embedding[train_edges[:, 0]], embedding[train_edges[:, 1]]))[shuffle_idx]
    X_train = X_train.astype(np.float32, copy=False)
    y_train = np.hstack((np.ones(len(pos_train_edges), dtype=np.int8), np.zeros(len(neg_train_edges), dtype=np.int8)))[
        shuffle_idx
    ]

    return_tuple = (X_train, y_train)

    if return_val_data:
        val_data = downstream_df.loc[
            downstream_df.loc[:, DOWNSTREAM_TASK_DATA_SPLIT_COL_KEY] == DOWNSTREAM_TASK_DATA_VAL_CATEGORY
        ]
        X_val = np.hstack(
            (
                embedding[val_data[DOWNSTREAM_TASK_DATA_SRC_EDGE_COL_KEY].to_numpy()],
                embedding[val_data[DOWNSTREAM_TASK_DATA_DEST_EDGE_COL_KEY].to_numpy()],
            )
        )
        X_val = X_val.astype(np.float32, copy=False)

        y_val = val_data[DOWNSTREAM_TASK_DATA_LABEL_COL_KEY].to_numpy(dtype=np.int8, copy=False)

        return_tuple += (X_val, y_val)

    if return_test_data:
        test_data = downstream_df.loc[
            downstream_df.loc[:, DOWNSTREAM_TASK_DATA_SPLIT_COL_KEY] == DOWNSTREAM_TASK_DATA_TEST_CATEGORY
        ]
        X_test = np.hstack(
            (
                embedding[test_data[DOWNSTREAM_TASK_DATA_SRC_EDGE_COL_KEY].to_numpy()],
                embedding[test_data[DOWNSTREAM_TASK_DATA_DEST_EDGE_COL_KEY].to_numpy()],
            )
        )
        X_test = X_test.astype(np.float32, copy=False)

        y_test = test_data[DOWNSTREAM_TASK_DATA_LABEL_COL_KEY].to_numpy(dtype=np.int8, copy=False)

        return_tuple += (X_test, y_test)

    return return_tuple


def iter_link_prediction_train_batches(
    downstream_df: pd.DataFrame,
    edge_list: str | List[Tuple[int, int]] | np.ndarray,
    embedding: np.ndarray,
    batch_size: int,
    seed: int = EXPERIMENTS_DEFAULT_SEED,
):
    """Yield mini-batches (X_batch, y_batch) for LP training without materializing full X_train."""

    if type(edge_list) == str:
        all_edges = pd.read_csv(edge_list, sep=" ", header=None).to_numpy()
    else:
        all_edges = np.array(edge_list)

    if DOWNSTREAM_TASK_DATA_TRAIN_CATEGORY in downstream_df.loc[:, DOWNSTREAM_TASK_DATA_SPLIT_COL_KEY]:
        pos_train_edges = downstream_df.loc[
            downstream_df.loc[:, DOWNSTREAM_TASK_DATA_SPLIT_COL_KEY] == DOWNSTREAM_TASK_DATA_TRAIN_CATEGORY
        ][[DOWNSTREAM_TASK_DATA_SRC_EDGE_COL_KEY, DOWNSTREAM_TASK_DATA_DEST_EDGE_COL_KEY]].to_numpy()

        # sanity check for potential data leakage through train-vs-eval positive overlap
        eval_pos_edges = downstream_df.loc[
            (
                downstream_df.loc[:, DOWNSTREAM_TASK_DATA_SPLIT_COL_KEY].isin(
                    [DOWNSTREAM_TASK_DATA_VAL_CATEGORY, DOWNSTREAM_TASK_DATA_TEST_CATEGORY]
                )
            )
            & (downstream_df.loc[:, DOWNSTREAM_TASK_DATA_LABEL_COL_KEY] == 1)
        ][[DOWNSTREAM_TASK_DATA_SRC_EDGE_COL_KEY, DOWNSTREAM_TASK_DATA_DEST_EDGE_COL_KEY]].to_numpy()
        if len(eval_pos_edges) > 0:
            train_pos_set = _edge_set_undirected(pos_train_edges)
            eval_pos_set = _edge_set_undirected(eval_pos_edges)
            if len(train_pos_set.intersection(eval_pos_set)) > 0:
                raise RuntimeError("Detected overlap between LP train positive edges and eval positive edges.")
    else:
        pos_train_edges = all_edges

    rng = np.random.RandomState(seed)

    pos_idx_perm = rng.permutation(len(pos_train_edges))
    for start_idx in range(0, len(pos_idx_perm), batch_size):
        curr_pos_idx = pos_idx_perm[start_idx : start_idx + batch_size]
        pos_batch = pos_train_edges[curr_pos_idx]
        n_pos = len(pos_batch)

        neg_batch = _sample_negative_edges(
            all_edges=all_edges,
            num_samples=n_pos,
            rng=rng,
        )

        batch_edges = np.vstack((pos_batch, neg_batch))
        y_batch = np.hstack((np.ones(n_pos, dtype=np.int8), np.zeros(n_pos, dtype=np.int8)))

        shuffle_idx = rng.permutation(len(batch_edges))
        batch_edges = batch_edges[shuffle_idx]
        y_batch = y_batch[shuffle_idx]

        X_batch = np.hstack((embedding[batch_edges[:, 0]], embedding[batch_edges[:, 1]])).astype(np.float32, copy=False)
        yield X_batch, y_batch


def prepare_link_prediction_eval_data(
    downstream_df: pd.DataFrame,
    embedding: np.ndarray,
    split_category: str,
) -> Tuple[np.ndarray, np.ndarray]:
    curr_data = downstream_df.loc[downstream_df.loc[:, DOWNSTREAM_TASK_DATA_SPLIT_COL_KEY] == split_category]
    X = np.hstack(
        (
            embedding[curr_data[DOWNSTREAM_TASK_DATA_SRC_EDGE_COL_KEY].to_numpy()],
            embedding[curr_data[DOWNSTREAM_TASK_DATA_DEST_EDGE_COL_KEY].to_numpy()],
        )
    ).astype(np.float32, copy=False)
    y = curr_data[DOWNSTREAM_TASK_DATA_LABEL_COL_KEY].to_numpy(dtype=np.int8, copy=False)
    return X, y


def prepare_node_classification_data(
    downstream_df: pd.DataFrame,
    embedding: List[np.ndarray],
    return_val_data: bool = False,
    return_test_data: bool = False,
) -> Union[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
]:

    y = downstream_df[DOWNSTREAM_TASK_DATA_LABEL_COL_KEY].to_numpy()
    train_mask = (
        downstream_df.loc[:, DOWNSTREAM_TASK_DATA_SPLIT_COL_KEY] == DOWNSTREAM_TASK_DATA_TRAIN_CATEGORY
    ).to_numpy()
    return_tuple = (embedding[train_mask], y[train_mask])

    if return_val_data:
        val_mask = (
            downstream_df.loc[:, DOWNSTREAM_TASK_DATA_SPLIT_COL_KEY] == DOWNSTREAM_TASK_DATA_VAL_CATEGORY
        ).to_numpy()
        return_tuple += (embedding[val_mask], y[val_mask])

    if return_test_data:
        test_mask = (
            downstream_df.loc[:, DOWNSTREAM_TASK_DATA_SPLIT_COL_KEY] == DOWNSTREAM_TASK_DATA_TEST_CATEGORY
        ).to_numpy()
        return_tuple += (embedding[test_mask], y[test_mask])

    return return_tuple


def load_default_config(emb_method, n_iterations=EXPERIMENTS_NUM_ITERATIONS) -> Dict[str, Any]:
    # Load the configuration from the JSON file
    with open(CONFIG_DEFAULTS_FILE_PATH, "r") as f:
        config_full = json.load(f)
    config = config_full[emb_method]
    config[CONFIG_ITERATIONS_KEY] = n_iterations
    return config


def get_embedding_params(config) -> Dict[str, Any]:
    params = dict()
    embedding_name = config.get(CONFIG_EMBEDDING_NAME_KEY)
    params[CONFIG_EMBEDDING_NAME_KEY] = embedding_name
    if embedding_name == GRAPHSAGE:
        params["number_of_layers"] = config.get("num_layers")
        params["neighbor_sampling_size"] = config.get("num_neighbors")
    elif embedding_name == NODE2VEC:
        params["p"] = config.get("p")
        params["q"] = config.get("q")
        params["walk_length"] = config.get("walk_length")
        params["number_of_walks"] = config.get("number_of_walks")
    elif embedding_name == DGI:
        params["learning_rate"] = config.get("learning_rate")
    elif embedding_name == HOPE:
        params["beta"] = config.get("beta")
    elif embedding_name == SDNE:
        params["beta"] = config.get("beta")
        params["rho"] = config.get("rho")
        params["nu"] = config.get("nu")
        params["xeta"] = config.get("xeta")

    return params


def get_dataset_params(config: Dict[str, Any]) -> Dict[str, Any]:
    params = dict()
    dataset_name = config[CONFIG_DATASET_NAME_KEY]
    params[CONFIG_DATASET_NAME_KEY] = dataset_name
    if dataset_name in EMPIRICAL_DATASET_LIST:
        params[CONFIG_DATA_SUBSAMPLING_INDICATOR_KEY] = config.get("snowball_sample")
        params[CONFIG_DATA_SUBSAMPLING_RATIO_KEY] = config.get(CONFIG_DATA_SUBSAMPLING_RATIO_KEY)
        params[CONFIG_DATA_SAMPLING_SEED_KEY] = config.get(CONFIG_DATA_SAMPLING_SEED_KEY)
    else:
        params["num_nodes"] = (config.get("size"),)
        params["density"] = (config.get("density"),)

    return params


def get_best_parameter_dict(
    embedding_method: EMBEDDING_ALGORITHM, dataset_params: Dict[str, Any], dimensions: List[int]
) -> Dict[int, Dict[str, Any]]:
    dataset_name = dataset_params[CONFIG_DATASET_NAME_KEY]
    parameter_dict = dict()
    if embedding_method != SDNE:
        if dataset_name in SYNTHETIC_DATASET_LIST:
            tune_dir = CREATE_SYNTH_TUNING_RESULTS_PATH(
                dataset_params=dataset_params, embedding_name=embedding_method
            )
        else:
            tune_dir = CREATE_MODELS_PATH(
                dataset_params=dataset_params,
                embedding_name=embedding_method,
                embedding_dim=TUNING_DEFAULT_DIMENSION,
                b_tune=True,
            )
        tuning_results_file_path = osp.join(tune_dir, TUNING_SUMMARY_FILE_NAME)

        if osp.isfile(tuning_results_file_path):
            with open(tuning_results_file_path) as f:
                tuning_summary = json.load(f)
            tune_id_list = list(tuning_summary.keys())
            tune_scores = list(tuning_summary[tid][TUNING_SUMMARY_SCORE_KEY] for tid in tune_id_list)
            best_id = tune_id_list[tune_scores.index(max(tune_scores))]
            for embedding_dim in dimensions:
                parameter_dict[embedding_dim] = tuning_summary[best_id][TUNING_SUMMARY_PARAMS_KEY]
        else:
            raise FileNotFoundError(
                f"Tuning results file does not exist - {embedding_method} still needs to be tuned"
                f" on {dataset_name} dataset!"
            )
    else:
        for embedding_dim in dimensions:
            tune_dir = CREATE_MODELS_PATH(
                dataset_params=dataset_params,
                embedding_name=embedding_method,
                embedding_dim=embedding_dim,
                b_tune=True,
            )
            tuning_results_file_path = osp.join(tune_dir, TUNING_SUMMARY_FILE_NAME)

            with open(tuning_results_file_path) as f:
                tuning_summary = json.load(f)
            tune_id_list = list(tuning_summary.keys())
            tune_scores = list(tuning_summary[tid][TUNING_SUMMARY_SCORE_KEY] for tid in tune_id_list)
            best_id = tune_id_list[tune_scores.index(max(tune_scores))]
            parameter_dict[embedding_dim] = tuning_summary[best_id][TUNING_SUMMARY_PARAMS_KEY]
    return parameter_dict


# Function to update the config dictionary with command-line arguments
def update_config(config, args):
    for key, value in vars(args).items():
        if value is not None and key != "config" and key != "model":
            config[key] = value
    return config


def save_results_to_json(
    results: Dict,
    dataset_params: Dict[str, Any],
    embedding_name: EMBEDDING_ALGORITHM,
    embedding_dim: int,
    b_tune: bool,
):
    """Saves results dict to a JSON file."""
    file_path = CREATE_MODELS_PATH(
        dataset_params=dataset_params, embedding_name=embedding_name, embedding_dim=embedding_dim, b_tune=b_tune
    )
    with open(osp.join(file_path, "summary.json"), "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {file_path}")


# def save_model(model, embedding_name, embedding_dim, dataset_params, iteration, tune_id=None):
#     b_tune = False if tune_id is None else True
#     save_dir = CREATE_MODELS_PATH(
#         dataset_params=dataset_params, embedding_name=embedding_name, embedding_dim=embedding_dim, b_tune=b_tune
#     )
#
#     save_path = osp.join(save_dir, MODEL_FILE_NAME(iteration, tune_id))
#     if not osp.exists(save_dir):
#         print(f"Directory {save_dir} does not exist.")
#     elif not os.access(save_dir, os.W_OK):
#         print(f"Directory {save_dir} is not writable.")
#     else:
#         print(f"Saving model to {save_path}")
#         torch.save(model.state_dict(), save_path)


def get_environment(alg_env_name: str):

    env = os.environ.copy()
    default_env_name = env["CONDA_DEFAULT_ENV"]
    env["PATH"] = env["PATH"].replace(default_env_name, alg_env_name)
    env["CONDA_DEFAULT_ENV"] = alg_env_name
    env["CONDA_PREFIX"] = env["CONDA_PREFIX"].replace(default_env_name, alg_env_name)

    return env
