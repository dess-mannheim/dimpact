import argparse
import json
from argparse import Namespace

from sklearn.model_selection import ParameterGrid

from paths_globals import *
from train import train_embeddings
from tools import train_utils


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
        choices=DATASET_LIST,
        default=DEFAULT_DATASET_LIST,
        help="Datasets used in evaluation.",
    )
    parser.add_argument("-gpu", "--gpu_id", default=0, type=int, help="ID of GPU to use")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Should the existing bestparam file be overwritten?",
    )
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of parallel jobs to run (if possible)")
    parser.add_argument("-dim", "--dimensions", nargs="+", type=int, default=None, help="List of dimensions.")

    return parser.parse_args()


def tune_grid_search(
    embedding_method: EMBEDDING_ALGORITHM,
    dataset_name: DATASET,
    param_grid: List[Dict[str, Any]],
    embedding_dim: int,
    overwrite: bool = False,
    n_jobs: int = 1,
) -> None:

    config = train_utils.load_default_config(embedding_method)

    dataset_params = {CONFIG_DATASET_NAME_KEY: dataset_name, CONFIG_DATA_SUBSAMPLING_INDICATOR_KEY: False}

    save_dir = CREATE_MODELS_PATH(
        dataset_params=dataset_params, embedding_name=embedding_method, embedding_dim=embedding_dim, b_tune=True
    )

    config[CONFIG_DIMENSION_KEY] = embedding_dim
    config[CONFIG_ITERATIONS_KEY] = TUNING_DEFAULT_ITERATIONS
    tuning_results_file_path = os.path.join(save_dir, TUNING_SUMMARY_FILE_NAME)

    if os.path.isfile(tuning_results_file_path) and not overwrite:
        with open(tuning_results_file_path) as f:
            tuning_summary = json.load(f)
        start_index = max([int(k) for k in tuning_summary.keys()]) + 1
    else:
        tuning_summary = dict()
        start_index = 0

    for i in range(start_index, len(param_grid)):

        params = param_grid[i]

        for k, v in params.items():
            config[k] = v

        results_dict = train_embeddings(
            embedding_name=embedding_method,
            dataset_params=dataset_params,
            embedding_config=config,
            tune_id=i,
            overwrite=overwrite,
            n_jobs=n_jobs,
        )
        tuning_summary[str(i)] = {
            TUNING_SUMMARY_PARAMS_KEY: params,
            TUNING_SUMMARY_RESULTS_KEY: results_dict,
            TUNING_SUMMARY_SCORE_KEY: np.mean(list(results_dict.values())),
        }
        with open(osp.join(save_dir, TUNING_SUMMARY_FILE_NAME), "w") as f:
            json.dump(tuning_summary, f, indent=4)
        print(f"Tuning results updated.")


if __name__ == "__main__":

    args = parse_args()
    datasets = args.datasets
    algorithms = args.algorithms

    for dataset in datasets:

        for algorithm in algorithms:

            print(algorithm)
            pspace = list(ParameterGrid(TUNING_PARAM_GRID_DICT[algorithm]))
            dimensions_list = [TUNING_DEFAULT_DIMENSION] if args.dimensions is None else args.dimensions

            for dim in dimensions_list:

                tune_grid_search(
                    embedding_method=algorithm,
                    dataset_name=dataset,
                    embedding_dim=dim,
                    param_grid=pspace,
                    overwrite=args.overwrite,
                    n_jobs=args.n_jobs,
                )
