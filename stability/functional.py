import json
from itertools import combinations
import argparse

from tools import train_utils
from joblib import Parallel, delayed
from stability.measures import *

from paths_globals import *


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
        "-c",
        "--classifiers",
        nargs="+",
        type=str,
        choices=DOWNSTREAM_CLASSIFIERS,
        default=[LOGISTIC_REGRESSION, MULTILAYER_PERCEPTRON],
        help="Downstream Classifier to use to run",
    )
    parser.add_argument(
        "-m",
        "--measures",
        nargs="*",
        type=str,
        choices=list(ALL_FUNCSIM_MEASURES.keys()),
        default=list(ALL_FUNCSIM_MEASURES.keys()),
        help="Datasets used in evaluation.",
    )
    parser.add_argument("-gpu", "--gpu_id", default=0, type=int, help="ID of GPU to use")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Should existing results be overwritten?",
    )
    parser.add_argument(
        "-dim", "--dimensions", nargs="+", type=int, default=EXPERIMENTS_DIMENSIONS_LIST, help="List of dimensions."
    )
    parser.add_argument(
        "--compute_control_values",
        action="store_true",
        help="Also compute stability values controlling for classifier variability",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="Number of parallel processes to run",
    )
    return parser.parse_args()


def analyze_functional_stability(
    embedding_method: EMBEDDING_ALGORITHM,
    dataset_params: Dict,
    classifiers: List[DOWNSTREAM_CLASSIFIER],
    measures: List,
    dimensions: List[int],
    n_jobs: int,
    overwrite=False,
):

    results_file_path = osp.join(
        CREATE_STABILITY_RESULTS_PATH(dataset_params, embedding_method),
        STABILITY_RESULTS_JSON_FILE_NAME(FUNCTIONAL),
    )
    if osp.exists(results_file_path):
        with open(results_file_path, "r") as f:
            results = json.load(f)
    else:
        results = {clf_name: {measure: dict() for measure in measures} for clf_name in classifiers}

    prediction_pairs = list(combinations(range(EXPERIMENTS_NUM_ITERATIONS), 2))

    for clf_name in classifiers:
        results[clf_name] = dict()
        print(f"Consider predictions from {clf_name}.")

        for measure in measures:
            results[clf_name][measure] = dict()
            print(f"Compute downstream stability with respect to {measure} measure.")

            for dimension in dimensions:
                if str(dimension) in results[clf_name][measure].keys() and not overwrite:
                    continue
                print(f"Embedding dimension is {dimension}")

                results_dir = CREATE_DOWNSTREAM_RESULTS_PATH(
                    dataset_params,
                    embedding_method,
                    embedding_dim=dimension,
                    clf_name=clf_name,
                )
                predictions = [
                    np.load(
                        osp.join(
                            results_dir,
                            DOWNSTREAM_PREDICTIONS_FILENAME(emb_id=i, model_seed=EXPERIMENTS_DEFAULT_SEED),
                        )
                    )
                    for i in range(EXPERIMENTS_NUM_ITERATIONS)
                ]

                with open(
                    osp.join(
                        results_dir,
                        DOWNSTREAM_PERFORMANCE_JSON_FILE_NAME,
                    ),
                    "r",
                ) as f:
                    performance_dict = json.load(fp=f)

                accuracies = [
                    performance_dict[ACCURACY_SCORE][str(i)][str(EXPERIMENTS_DEFAULT_SEED)]
                    for i in range(EXPERIMENTS_NUM_ITERATIONS)
                ]

                if measure in PAIRWISE_FUNCTIONAL_SIMILARITY_MEASURES:
                    results[clf_name][measure][dimension] = Parallel(n_jobs=n_jobs)(
                        delayed(ALL_FUNCSIM_MEASURES[measure])(predictions[pair[0]], predictions[pair[1]])
                        for pair in prediction_pairs
                    )
                elif measure in PERFORMANCE_BASED_FUNCTIONAL_SIMILARITY_MEASURES:
                    results[clf_name][measure][dimension] = Parallel(n_jobs=n_jobs)(
                        delayed(ALL_FUNCSIM_MEASURES[measure])(
                            predictions[pair[0]], predictions[pair[1]], accuracies[pair[0]], accuracies[pair[1]]
                        )
                        for pair in prediction_pairs
                    )
                elif measure in GROUPWISE_FUNCTIONAL_SIMILARITY_MEASURES:
                    results[clf_name][measure][dimension] = ALL_FUNCSIM_MEASURES[measure](predictions)

    print("Saving resulting similarity scores...")
    with open(results_file_path, "w") as f:
        json.dump(results, f)


def analyze_control_stability(
    embedding_method: EMBEDDING_ALGORITHM,
    dataset_params: Dict,
    classifiers: List[DOWNSTREAM_CLASSIFIER],
    measures: List,
    dimensions: List[int],
    n_jobs: int,
    overwrite: bool = False,
    negative_sampling=False,
):

    fname = (
        FUNCSIM_NEGATIVE_SAMPLING_CONTROL_RESULTS_JSON_FILE_NAME
        if negative_sampling
        else FUNCSIM_CLF_CONTROL_RESULTS_JSON_FILE_NAME
    )
    results_file_path = osp.join(
        CREATE_STABILITY_RESULTS_PATH(dataset_params, embedding_method),
        fname,
    )
    if osp.exists(results_file_path):
        with open(results_file_path, "r") as f:
            results = json.load(f)
    else:
        results = dict()

    prediction_pairs = list(combinations(range(EXPERIMENTS_NUM_DOWNSTREAM_CLF_RUNS), 2))

    for clf_name in classifiers:
        if clf_name not in results.keys():
            results[clf_name] = dict()
        print(f"Consider predictions from {clf_name}.")

        for measure in measures:
            if measure not in results[clf_name].keys():
                results[clf_name][measure] = dict()
            print(f"Compute downstream stability with respect to {measure} measure.")

            for dimension in dimensions:
                dim_key = str(dimension)
                if dim_key in results[clf_name][measure].keys() and not overwrite:
                    continue
                print(f"Embedding dimension is {dimension}")

                results_dir = CREATE_DOWNSTREAM_RESULTS_PATH(
                    dataset_params,
                    embedding_method,
                    embedding_dim=dimension,
                    clf_name=clf_name,
                )

                results[clf_name][measure][dim_key] = dict()

                performance_fname = (
                    DOWNSTREAM_PERFORMANCE_NS_JSON_FILE_NAME
                    if negative_sampling
                    else DOWNSTREAM_PERFORMANCE_JSON_FILE_NAME
                )
                with open(osp.join(results_dir, performance_fname), "r") as f:
                    performance_dict = json.load(fp=f)

                for emb_id in range(EXPERIMENTS_NUM_CLF_CONTROL_EMBEDDINGS):
                    emb_key = str(emb_id)
                    if negative_sampling:
                        predictions = [
                            np.load(
                                osp.join(
                                    results_dir,
                                    DOWNSTREAM_PREDICTIONS_FILENAME(
                                        emb_id=emb_id,
                                        model_seed=EXPERIMENTS_DEFAULT_SEED,
                                        negative_sampling_seed=ns_seed,
                                    ),
                                )
                            )
                            for ns_seed in range(EXPERIMENTS_NUM_DOWNSTREAM_CLF_RUNS)
                        ]
                        accuracies = [
                            performance_dict[ACCURACY_SCORE][emb_key][str(ns_seed)]
                            for ns_seed in range(EXPERIMENTS_NUM_DOWNSTREAM_CLF_RUNS)
                        ]
                    else:
                        predictions = [
                            np.load(
                                osp.join(
                                    results_dir,
                                    DOWNSTREAM_PREDICTIONS_FILENAME(emb_id=emb_id, model_seed=model_seed),
                                )
                            )
                            for model_seed in range(EXPERIMENTS_NUM_DOWNSTREAM_CLF_RUNS)
                        ]
                        accuracies = [
                            performance_dict[ACCURACY_SCORE][emb_key][str(model_seed)]
                            for model_seed in range(EXPERIMENTS_NUM_DOWNSTREAM_CLF_RUNS)
                        ]

                    if measure in PAIRWISE_FUNCTIONAL_SIMILARITY_MEASURES:
                        similarities = Parallel(n_jobs=n_jobs)(
                            delayed(ALL_FUNCSIM_MEASURES[measure])(predictions[pair[0]], predictions[pair[1]])
                            for pair in prediction_pairs
                        )
                    elif measure in PERFORMANCE_BASED_FUNCTIONAL_SIMILARITY_MEASURES:
                        similarities = Parallel(n_jobs=n_jobs)(
                            delayed(ALL_FUNCSIM_MEASURES[measure])(
                                predictions[pair[0]], predictions[pair[1]], accuracies[pair[0]], accuracies[pair[1]]
                            )
                            for pair in prediction_pairs
                        )
                    elif measure in GROUPWISE_FUNCTIONAL_SIMILARITY_MEASURES:
                        similarities = ALL_FUNCSIM_MEASURES[measure](predictions)
                    else:
                        continue

                    # Keep similarities for each control embedding separately.
                    results[clf_name][measure][dim_key][emb_key] = similarities

    print("Saving resulting similarity scores...")
    with open(results_file_path, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    args = parse_args()
    datasets = args.datasets
    algorithms = args.algorithms

    for dataset_name in datasets:

        for algorithm in algorithms:
            config = train_utils.load_default_config(algorithm)

            data_params = {CONFIG_DATASET_NAME_KEY: dataset_name, CONFIG_DATA_SUBSAMPLING_INDICATOR_KEY: False}

            dimensions_list = [d for d in args.dimensions if d <= MAX_DIMENSION_DICT[algorithm][dataset_name]]

            downstream_task = DATASET_TASK_DICT[dataset_name]
            print(f"Analyze functional stability of {algorithm} on {dataset_name} dataset.")
            analyze_functional_stability(
                embedding_method=algorithm,
                dataset_params=data_params,
                classifiers=args.classifiers,
                measures=args.measures,
                dimensions=dimensions_list,
                n_jobs=args.n_jobs,
                overwrite=args.overwrite,
            )

            if args.compute_control_values:
                analyze_control_stability(
                    embedding_method=algorithm,
                    dataset_params=data_params,
                    classifiers=args.classifiers,
                    measures=args.measures,
                    dimensions=dimensions_list,
                    n_jobs=args.n_jobs,
                    overwrite=args.overwrite,
                )

                if downstream_task == LINK_PREDICTION:
                    analyze_control_stability(
                        embedding_method=algorithm,
                        dataset_params=data_params,
                        classifiers=args.classifiers,
                        measures=args.measures,
                        dimensions=dimensions_list,
                        n_jobs=args.n_jobs,
                        overwrite=args.overwrite,
                        negative_sampling=True,
                    )
