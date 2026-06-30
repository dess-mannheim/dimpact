import argparse
import json

import pandas as pd
from joblib import Parallel, delayed

from grape.embedders import Node2VecSkipGramEnsmallen
from grape import Graph

from tools.train_utils import prepare_node_classification_data, prepare_link_prediction_data
from paths_globals import *


def parse_args():
    """Parses arguments given to script
    Returns:
        dict-like object -- Dict-like object containing all given arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_path",
        type=str,
        help="Path to edgelist file",
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
    embedding_config: Dict,
    save_dir: str,
    downstream_path: str,
    tune_id: int,
    seed: int,
    b_overwrite: bool,
) -> float:

    # # Load graph
    graph = Graph.from_csv(
        # Edges related parameters
        # The path to the edges list tsv
        edge_path=edge_list_path,
        # Set the tab as the separator between values
        edge_list_separator=" ",
        # The first rows should NOT be used as the columns names
        edge_list_header=False,
        # The source nodes are in the first nodes
        sources_column_number=0,
        # The destination nodes are in the second column
        destinations_column_number=1,
        # Both source and destinations columns use numeric node_ids instead of node names
        edge_list_numeric_node_ids=True,
        # Graph related parameters
        # The graph is undirected
        directed=False,
        # The name of the graph is HomoSapiens
        name="CoAuthor",
        # Display a progress bar, (this might be in the terminal and not in the notebook)
        verbose=True,
    )

    embedding_file_path = os.path.join(save_dir, EMBEDDING_FILE_NAME(tune_id=tune_id, model_seed=seed))
    if osp.isfile(embedding_file_path) and not b_overwrite:
        if tune_id is None:
            print("Embeddings has already been computed and overwrite is False, return")
            return 0.5
        embedding = np.load(embedding_file_path)
    else:
        model = Node2VecSkipGramEnsmallen(
            embedding_size=embedding_config[CONFIG_DIMENSION_KEY],
            walk_length=embedding_config["walk_length"],
            window_size=embedding_config["context_size"],
            iterations=embedding_config["walks_per_node"],
            number_of_negative_samples=embedding_config["num_negative_samples"],
            return_weight=1.0 / embedding_config["p"],
            explore_weight=1.0 / embedding_config["q"],
            epochs=embedding_config["epochs"],
            learning_rate=embedding_config["learning_rate"],
            random_state=seed,
            verbose=True,
        )

        # retrieve embedding and sort index by node ID
        embedding = model.fit_transform(graph).get_all_node_embedding()[0]
        embedding.index = embedding.index.astype(int)
        embedding = embedding.sort_index().to_numpy()

        np.save(embedding_file_path, embedding)

    downstream_df = pd.read_csv(downstream_path, index_col=0)
    if DOWNSTREAM_TASK_DATA_SRC_EDGE_COL_KEY in list(downstream_df):
        X_train, y_train, X_val, y_val = prepare_link_prediction_data(
            downstream_df=downstream_df,
            edge_list=edge_list_path,
            embedding=embedding,
            seed=seed,
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

    seeds = config.get(CONFIG_TRAINING_SEEDS_KEY, list(range(config[CONFIG_ITERATIONS_KEY])))
    overwrite = args.overwrite

    print(f"Train {len(seeds)} embeddings sequentially")
    scores = [
        train_model(
            args.data_path, config, args.models_dir, args.downstream_data_path, args.tune_id, seed, overwrite
        )
        for seed in seeds
    ]

    results_dict = {seed: score for seed, score in zip(seeds, scores)}

    with open(os.path.join(args.models_dir, TMP_TUNING_RESULTS_FILE_NAME(args.tune_id)), "w") as f:
        json.dump(results_dict, f)
