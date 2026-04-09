import os.path

from models.pyg import (
    dgi_inductive as deep_graph_infomax,
    graphsage as graphsage,
)

from models.verse import verse
import argparse
from paths_globals import *
from tools import train_utils, data_utils
import json
import subprocess
import uuid
from argparse import Namespace


def parse_args() -> Namespace:
    """Parses arguments given to script
    Returns:
        dict-like object -- Dict-like object containing all given arguments
    """
    parser = argparse.ArgumentParser()

    # Test parameters
    parser.add_argument(
        "-a",
        "--algorithms",
        nargs="+",
        type=str,
        choices=EMBEDDING_ALGORITHM_LIST,
        help="Algorithms used in evaluation.",
    )
    parser.add_argument(
        "-d",
        "--datasets",
        nargs="*",
        type=str,
        choices=EMPIRICAL_DATASET_LIST,
        default=DEFAULT_DATASET_LIST,
        help="Datasets used in evaluation.",
    )
    parser.add_argument("-gpu", "--gpu_id", default=0, type=int, help="ID of GPU to use")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Should existing embeddings be overwritten?",
    )
    parser.add_argument(
        "-dim", "--dimensions", nargs="+", type=int, default=EXPERIMENTS_DIMENSIONS_LIST, help="List of dimensions."
    )
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of parallel jobs to run (if possible)")

    return parser.parse_args()


def train_embeddings(
    embedding_name: EMBEDDING_ALGORITHM,
    dataset_params: Dict[str, Any],
    embedding_config: Dict[str, Any],
    tune_id: int | None = None,
    overwrite: bool = False,
    n_jobs: int = 1,
) -> Dict[int, float]:

    data, edge_list_path = data_utils.load_dataset(dataset_params.copy())

    dataset_name = dataset_params[CONFIG_DATASET_NAME_KEY]
    embedding_name = embedding_name
    embedding_dim = embedding_config["dimension"]
    results_dict = dict()

    save_dir = CREATE_MODELS_PATH(
        dataset_params=dataset_params,
        embedding_name=embedding_name,
        embedding_dim=embedding_dim,
        b_tune=True if tune_id is not None else False,
    )

    if tune_id is None:
        print(f"Training {embedding_name} model on {dataset_name} with embedding dimension {embedding_dim}")
    else:
        print(f"Tune {embedding_name} model on {dataset_name} dataset at dimension {embedding_dim}")

    if embedding_name in [DGI, GRAPHSAGE]:

        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        os.environ["TORCH_USE_CUDA_DSA"] = "1"
        for iteration in range(embedding_config[CONFIG_ITERATIONS_KEY]):

            save_path = osp.join(save_dir, MODEL_FILE_NAME(model_seed=iteration, tune_id=tune_id))
            emb_path = osp.join(save_dir, EMBEDDING_FILE_NAME(model_seed=iteration, tune_id=tune_id))

            if not osp.isfile(emb_path) or overwrite:
                if embedding_name == DGI:
                    best_acc = deep_graph_infomax.train_model(
                        dataset=data,
                        embedding_dim=embedding_dim,
                        config=embedding_config,
                        save_path=save_path,
                        seed=iteration,
                        embedding_path=emb_path,
                    )
                elif embedding_name == GRAPHSAGE:
                    best_acc = graphsage.train_model(
                        dataset=data,
                        embedding_dim=embedding_dim,
                        config=embedding_config,
                        save_path=save_path,
                        seed=iteration,
                        embedding_path=emb_path,
                    )
                else:
                    raise ValueError("Invalid embedding type!")

                results_dict[iteration] = best_acc
            else:
                print(f"Embedding with seed {iteration} has already been trained and overwrite is False.")

    elif embedding_name == VERSE:
        data_dir = BUILD_DATASET_SRC_DIR(dataset_params)
        downstream_df_path = os.path.join(data_dir, DOWNSTREAM_TASK_DATA_FILE_NAME)
        for iteration in range(embedding_config[CONFIG_ITERATIONS_KEY]):
            save_path = os.path.join(save_dir, EMBEDDING_FILE_NAME(model_seed=iteration, tune_id=tune_id))

            if not osp.isfile(save_path) or overwrite:
                best_acc = verse.train_model(
                    edge_list_path=edge_list_path,
                    embedding_config=embedding_config,
                    save_path=save_path,
                    seed=iteration,
                    downstream_path=downstream_df_path,
                    n_jobs=n_jobs,
                )
                results_dict[iteration] = best_acc

            else:
                print(f"Embedding with seed {iteration} has already been trained and overwrite is False.")
    else:

        if (
            all(
                [
                    osp.isfile(os.path.join(save_dir, EMBEDDING_FILE_NAME(tune_id=tune_id, model_seed=seed)))
                    for seed in range(embedding_config[CONFIG_ITERATIONS_KEY])
                ]
            )
            and not overwrite
        ):
            print(f"Embeddings have already been trained and overwrite is False.")
            return results_dict
        # Extract edge list from the loaded or newly created dataset
        # edge_list = data.edge_index.t().tolist()
        target_env = train_utils.get_environment(ENVIRONMENTS_DICT[embedding_name])
        config_file_name = f"{embedding_name}_config_{uuid.uuid4().hex}.json"
        config_path = osp.join(CONFIGS_DIR, config_file_name)

        with open(config_path, "w") as f:
            json.dump(embedding_config, f, indent=4)

        module_name = MODULE_NAME_DICT[embedding_name]
        command = (
            f'python -m {module_name} --data_path "{edge_list_path}" --config_path "{config_path}" '
            f'--models_dir "{save_dir}"'
        )
        if embedding_name in [NODE2VEC, ASNE]:
            data_dir = BUILD_DATASET_SRC_DIR(dataset_params)
            downstream_df_path = os.path.join(data_dir, DOWNSTREAM_TASK_DATA_FILE_NAME)
            command += f' --downstream_data_path "{downstream_df_path}"'
            if embedding_name == ASNE:
                feature_matrix_file_path = osp.join(
                    osp.dirname(edge_list_path), DATA_FEATURE_MATRIX_DEFAULT_FILE_NAME
                )
                if not osp.isfile(feature_matrix_file_path):
                    np.save(feature_matrix_file_path, data.x.numpy())

                command += f' --n_jobs {n_jobs} --feature_matrix_path "{feature_matrix_file_path}"'

        if tune_id is not None:
            command += f" --tune_id {tune_id} "

        if overwrite:
            command += " --overwrite"

        print("Run embedding subprocess")
        subprocess.run(command, shell=True, env=target_env)

        results_dict_fpath = os.path.join(save_dir, TMP_TUNING_RESULTS_FILE_NAME(tune_id))
        with open(results_dict_fpath, "r") as f:
            results_dict = json.load(f)

        os.remove(results_dict_fpath)
        os.remove(config_path)

    return {int(k): v for k, v in results_dict.items()}


def train_embeddings_over_dimensions(
    embedding_method: EMBEDDING_ALGORITHM,
    dataset_name: DATASET,
    dimensions: List[int],
    overwrite: bool,
    n_jobs: int,
) -> None:

    config = train_utils.load_default_config(embedding_method)

    dataset_params = {CONFIG_DATASET_NAME_KEY: dataset_name, CONFIG_DATA_SUBSAMPLING_INDICATOR_KEY: False}

    config[CONFIG_ITERATIONS_KEY] = EXPERIMENTS_NUM_ITERATIONS

    parameter_dict = train_utils.get_best_parameter_dict(
        embedding_method=embedding_method,
        dataset_params=dataset_params,
        dimensions=dimensions,
    )

    for embedding_dim in dimensions:
        config[CONFIG_DIMENSION_KEY] = embedding_dim

        params = parameter_dict[embedding_dim]
        for k, v in params.items():
            config[k] = v

        train_embeddings(
            embedding_name=embedding_method,
            dataset_params=dataset_params,
            embedding_config=config,
            overwrite=overwrite,
            n_jobs=n_jobs,
        )


if __name__ == "__main__":

    args = parse_args()
    datasets = args.datasets
    algorithms = args.algorithms

    for dataset in datasets:

        for algorithm in algorithms:

            train_embeddings_over_dimensions(
                embedding_method=algorithm,
                dataset_name=dataset,
                dimensions=args.dimensions,
                overwrite=args.overwrite,
                n_jobs=args.n_jobs,
            )
