import argparse
import json
import os.path as osp

import numpy as np
import networkx as nx
import pandas as pd
from scipy.sparse import coo_matrix

from karateclub import ASNE

from tools.train_utils import prepare_node_classification_data, prepare_link_prediction_data
from sklearn.linear_model import LogisticRegression
from paths_globals import (
    CONFIG_ITERATIONS_KEY,
    CONFIG_DIMENSION_KEY,
    EMBEDDING_FILE_NAME,
    DOWNSTREAM_TASK_DATA_SRC_EDGE_COL_KEY,
    TMP_TUNING_RESULTS_FILE_NAME,
)

from typing import Dict


def parse_args():
    """Parses arguments given to script
    Returns:
        dict-like object -- Dict-like object containing all given arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_path",
        type=str,
        help="Path to graphml file",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        help="Path to embedding config file",
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        help="Directory to save embeddings in.",
    )
    parser.add_argument(
        "--downstream_data_path",
        type=str,
        help="Path to downstream data file.",
    )
    parser.add_argument(
        "--feature_matrix_path",
        type=str,
        help="Path to downstream data file.",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="Number of parallel jobs to run.",
    )
    parser.add_argument(
        "--tune_id", type=int, help="ID of tuning parameters - only used for file naming purposes", default=None
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Should existing embeddings be overwritten?",
    )

    return parser.parse_args()


def train_model(
    edge_list_path: str,
    feature_matrix_path: str,
    embedding_config: Dict,
    save_dir: str,
    downstream_path: str,
    tune_id: int,
    seed: int,
    n_jobs: int,
    b_overwrite: bool,
) -> float:

    # # Load graph
    graph = nx.read_edgelist(edge_list_path)

    node_list = sorted(list(graph.nodes()))
    graph = nx.relabel_nodes(graph, mapping={v: i for i, v in enumerate(node_list)})

    with open(feature_matrix_path, "rb") as f:
        X = np.load(f)

    embedding_file_path = osp.join(save_dir, EMBEDDING_FILE_NAME(tune_id=tune_id, model_seed=seed))
    if osp.isfile(embedding_file_path) and not b_overwrite:
        embedding = np.load(embedding_file_path)
    else:
        model = ASNE(
            dimensions=embedding_config[CONFIG_DIMENSION_KEY],
            epochs=embedding_config["epochs"],
            learning_rate=embedding_config["learning_rate"],
            down_sampling=embedding_config["down_sampling"],
            seed=seed,
            workers=n_jobs,
        )

        model.fit(graph, coo_matrix(X))
        embedding = model.get_embedding()

        np.save(embedding_file_path, embedding)

    downstream_df = pd.read_csv(downstream_path, index_col=0)
    if DOWNSTREAM_TASK_DATA_SRC_EDGE_COL_KEY in list(downstream_df):
        X_train, y_train, X_val, y_val = prepare_link_prediction_data(
            downstream_df=downstream_df,
            edge_list=list(graph.edges),
            embedding=embedding,
            return_val_data=True,
        )
    else:
        X_train, y_train, X_val, y_val = prepare_node_classification_data(
            downstream_df=downstream_df,
            embedding=embedding,
            return_val_data=True,
        )

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    return clf.score(X_val, y_val)


if __name__ == "__main__":

    args = parse_args()

    with open(args.config_path, "r") as f:
        config = json.load(f)

    iterations = config[CONFIG_ITERATIONS_KEY]
    overwrite = args.overwrite

    scores = [
        train_model(
            edge_list_path=args.data_path,
            feature_matrix_path=args.feature_matrix_path,
            embedding_config=config,
            save_dir=args.models_dir,
            downstream_path=args.downstream_data_path,
            tune_id=args.tune_id,
            seed=iteration,
            n_jobs=args.n_jobs,
            b_overwrite=overwrite,
        )
        for iteration in range(iterations)
    ]

    results_dict = {i: score for i, score in enumerate(scores)}

    with open(osp.join(args.models_dir, TMP_TUNING_RESULTS_FILE_NAME(args.tune_id)), "w") as f:
        json.dump(results_dict, f)
