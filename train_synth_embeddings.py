import argparse
import json
from argparse import Namespace

from sklearn.model_selection import ParameterGrid
from paths_globals import *
from tools import train_utils

from train import train_embeddings


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
        choices=SYNTHETIC_DATASET_LIST,
        help="Datasets used in evaluation.",
    )
    parser.add_argument(
        "--num_nodes",
        nargs="+",
        type=int,
        default=[SYNTH_DATA_EXPERIMENTS_DEFAULT_NUM_NODES],
        help="Datasets used in evaluation.",
    )
    parser.add_argument(
        "--density",
        nargs="+",
        type=float,
        default=[SYNTH_DATA_EXPERIMENTS_DEFAULT_DENSITY],
        help="Densities to consider",
    )
    # parser.add_argument("-gpu", "--gpu_id", default=0, type=int, help="ID of GPU to use")
    parser.add_argument(
        "--vary_size",
        action="store_true",
        help="Whether to vary network size at fixed density",
    )
    parser.add_argument(
        "--vary_density",
        action="store_true",
        help="Whether to vary network density at fixed size",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Should existing embeddings be overwritten?",
    )
    parser.add_argument(
        "-dim", "--dimensions", nargs="+", type=int, default=EXPERIMENTS_DIMENSIONS_LIST, help="List of dimensions."
    )
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of parallel jobs to run (if possible)")
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=SYNTH_DATA_EXPERIMENTS_NUM_SEEDS,
        help="Number of random graph generation seeds.",
    )

    return parser.parse_args()


def tune_synthetic_embeddings(
    embedding_method: EMBEDDING_ALGORITHM,
    dataset_params: Dict[str, Any],
    overwrite: bool = False,
    n_jobs: int = 1,
    num_gen_seeds: int = SYNTH_DATA_EXPERIMENTS_NUM_SEEDS,
) -> None:
    param_grid = list(ParameterGrid(TUNING_PARAM_GRID_DICT[embedding_method]))

    config = train_utils.load_default_config(embedding_method)

    save_dir = CREATE_SYNTH_TUNING_RESULTS_PATH(dataset_params=dataset_params, embedding_name=embedding_method)

    config[CONFIG_DIMENSION_KEY] = TUNING_DEFAULT_DIMENSION
    config[CONFIG_ITERATIONS_KEY] = 1
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

        param_results = dict()
        for gen_seed in range(num_gen_seeds):
            dataset_params[CONFIG_DATA_SAMPLING_SEED_KEY] = gen_seed

            results_dict = train_embeddings(
                embedding_name=embedding_method,
                dataset_params=dataset_params,
                embedding_config=config,
                tune_id=i,
                overwrite=overwrite,
                n_jobs=n_jobs,
            )
            param_results = {**param_results, **{gen_seed: results_dict[0]}}

        tuning_summary[str(i)] = {
            TUNING_SUMMARY_PARAMS_KEY: params,
            TUNING_SUMMARY_RESULTS_KEY: param_results,
            TUNING_SUMMARY_SCORE_KEY: np.mean(list(param_results.values())),
        }
        with open(osp.join(save_dir, TUNING_SUMMARY_FILE_NAME), "w") as f:
            json.dump(tuning_summary, f, indent=4)
        print(f"Tuning results updated.")


def train_synthetic_embeddings(
    embedding_method: EMBEDDING_ALGORITHM,
    dataset_params: Dict[str, Any],
    dimensions: List[int],
    num_gen_seeds: int,
    overwrite: bool,
    n_jobs: int,
) -> None:

    config = train_utils.load_default_config(embedding_method)

    config[CONFIG_ITERATIONS_KEY] = EXPERIMENTS_NUM_SYNTH_ITERATIONS

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

        for gen_seed in range(num_gen_seeds):
            dataset_params[CONFIG_DATA_SAMPLING_SEED_KEY] = gen_seed
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
    densities = args.density
    network_sizes = args.num_nodes

    if args.vary_density:
        densities = SYNTH_DATA_EXPERIMENTS_DENSITIES_LIST
    if args.vary_size:
        network_sizes = SYNTH_DATA_EXPERIMENTS_NUM_NODES_LIST

    for dataset in datasets:

        for n in network_sizes:
            for d in densities:
                data_params = {
                    CONFIG_DATASET_NAME_KEY: dataset,
                    CONFIG_SYNTH_DATA_DENSITY_KEY: d,
                    CONFIG_SYNTH_DATA_NUM_NODES_KEY: n,
                    CONFIG_DATA_SUBSAMPLING_INDICATOR_KEY: False,
                }
                for algorithm in algorithms:

                    tune_synthetic_embeddings(
                        embedding_method=algorithm,
                        dataset_params=data_params,
                        overwrite=args.overwrite,
                        n_jobs=args.n_jobs,
                        num_gen_seeds=args.num_iterations,
                    )

                    train_synthetic_embeddings(
                        embedding_method=algorithm,
                        dataset_params=data_params,
                        dimensions=args.dimensions,
                        overwrite=args.overwrite,
                        n_jobs=args.n_jobs,
                        num_gen_seeds=args.num_iterations,
                    )
