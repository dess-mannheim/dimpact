import json
from itertools import combinations
import argparse
from joblib import Parallel, delayed

from tools import train_utils, data_utils
from stability.measures import *
from stability.measures.repsim import ND_SHAPE

from paths_globals import *

# from tools.load_embedding import load_embedding


def _compute_embedding_pair_similarity(
    measure_name: str,
    load_path: str,
    pair: tuple[int, int],
) -> float:
    left = np.load(osp.join(load_path, EMBEDDING_FILE_NAME(model_seed=pair[0])), mmap_mode="r")
    right = np.load(osp.join(load_path, EMBEDDING_FILE_NAME(model_seed=pair[1])), mmap_mode="r")
    return ALL_MEASURES[measure_name](left, right, shape=ND_SHAPE)


def parse_args():
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
    parser.add_argument(
        "--num_nodes",
        nargs="+",
        type=int,
        default=[SYNTH_DATA_EXPERIMENTS_DEFAULT_NUM_NODES],
        help="Number of nodes for synthetic datasets.",
    )
    parser.add_argument(
        "--density",
        nargs="+",
        type=float,
        default=[SYNTH_DATA_EXPERIMENTS_DEFAULT_DENSITY],
        help="Density values for synthetic datasets.",
    )
    parser.add_argument(
        "--vary_size",
        action="store_true",
        help="Whether to vary network size for synthetic datasets at fixed density.",
    )
    parser.add_argument(
        "--vary_density",
        action="store_true",
        help="Whether to vary density for synthetic datasets at fixed size.",
    )
    parser.add_argument(
        "--num_gen_seeds",
        type=int,
        default=SYNTH_DATA_EXPERIMENTS_NUM_SEEDS,
        help="Number of synthetic graph generation seeds to evaluate.",
    )
    parser.add_argument(
        "-m",
        "--measures",
        nargs="*",
        type=str,
        choices=list(ALL_MEASURES.keys()),
        default=[
            "AlignedCosineSimilarity",
            "JaccardSimilarity",
            "SecondOrderCosineSimilarity",
            "DistanceCorrelation",
        ],
        help="Datasets used in evaluation.",
    )
    parser.add_argument("--n_jobs", type=int, default=1, help="number of parallel jobs to run")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Should existing results be overwritten?",
    )
    parser.add_argument(
        "-dim", "--dimensions", nargs="+", type=int, default=EXPERIMENTS_DIMENSIONS_LIST, help="List of dimensions."
    )

    return parser.parse_args()


def analyze_representational_stability(
    dataset_params: Dict,
    embedding_name: EMBEDDING_ALGORITHM,
    embedding_config: Dict,
    measures: List,
    dimensions: List[int],
    n_jobs: int,
    overwrite: bool = False,
):
    data, _ = data_utils.load_dataset(dataset_params.copy())
    parameter_dict = train_utils.get_best_parameter_dict(
        embedding_method=embedding_name,
        dataset_params=dataset_params,
        dimensions=dimensions,
    )

    results_file_path = osp.join(
        CREATE_STABILITY_RESULTS_PATH(dataset_params, embedding_name),
        STABILITY_RESULTS_JSON_FILE_NAME(REPRESENTATIONAL),
    )

    if osp.exists(results_file_path):
        with open(results_file_path, "r") as f:
            results = json.load(f)
        for measure in measures:
            if measure not in results.keys():
                results[measure] = dict()
    else:
        results = {measure: dict() for measure in measures}

    embedding_pairs = list(combinations(range(0, embedding_config[CONFIG_ITERATIONS_KEY]), 2))
    for measure in measures:
        print(f"Compute stability with respect to {measure} measure.")
        for dimension in dimensions:
            dim_key = str(dimension)
            if dim_key in results[measure].keys() and not overwrite:
                print(f"Stability results for dimension {dimension} already exist and overwrite is False, skipping.")
                continue
            print(f"Embedding dimension is {dimension}")
            embedding_config[CONFIG_DIMENSION_KEY] = dimension

            params = parameter_dict[dimension]
            for k, v in params.items():
                embedding_config[k] = v

            if dim_key not in results[measure].keys() or overwrite:
                load_path = CREATE_MODELS_PATH(
                    dataset_params=dataset_params, embedding_name=embedding_name, embedding_dim=dimension
                )
                results[measure][dim_key] = Parallel(n_jobs=n_jobs)(
                    delayed(_compute_embedding_pair_similarity)(measure, load_path, pair) for pair in embedding_pairs
                )
            results[measure][dim_key] = [float(np_val) for np_val in results[measure][dim_key]]

            print("Similarities were calculated, saving resulting similarity scores...")
            with open(results_file_path, "w") as f:
                json.dump(results, f)


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

    for dataset_name in datasets:

        for algorithm in algorithms:
            config = train_utils.load_default_config(algorithm)
            config[CONFIG_ITERATIONS_KEY] = EXPERIMENTS_NUM_ITERATIONS

            dimensions_list = [d for d in args.dimensions if d <= MAX_DIMENSION_DICT[algorithm][dataset_name]]

            if dataset_name in SYNTHETIC_DATASET_LIST:
                config[CONFIG_ITERATIONS_KEY] = EXPERIMENTS_NUM_SYNTH_ITERATIONS
                for n in network_sizes:
                    for density in densities:
                        for gen_seed in range(args.num_gen_seeds):
                            data_params = {
                                CONFIG_DATASET_NAME_KEY: dataset_name,
                                CONFIG_SYNTH_DATA_DENSITY_KEY: density,
                                CONFIG_SYNTH_DATA_NUM_NODES_KEY: n,
                                CONFIG_DATA_SAMPLING_SEED_KEY: gen_seed,
                                CONFIG_DATA_SUBSAMPLING_INDICATOR_KEY: False,
                            }

                            print(
                                f"Analyze representational stability of {algorithm} on {dataset_name} dataset "
                                f"(n={n}, density={density}, graph_seed={gen_seed})."
                            )
                            analyze_representational_stability(
                                dataset_params=data_params,
                                embedding_name=algorithm,
                                embedding_config=config,
                                measures=args.measures,
                                dimensions=dimensions_list,
                                n_jobs=args.n_jobs,
                                overwrite=args.overwrite,
                            )
            else:
                data_params = {CONFIG_DATASET_NAME_KEY: dataset_name, CONFIG_DATA_SUBSAMPLING_INDICATOR_KEY: False}

                print(f"Analyze representational stability of {algorithm} on {dataset_name} dataset.")
                analyze_representational_stability(
                    dataset_params=data_params,
                    embedding_name=algorithm,
                    embedding_config=config,
                    measures=args.measures,
                    dimensions=dimensions_list,
                    n_jobs=args.n_jobs,
                    overwrite=args.overwrite,
                )
