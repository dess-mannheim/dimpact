from joblib import Parallel, delayed, parallel_backend
import argparse
from argparse import Namespace
import pandas as pd
from sklearn.model_selection import ParameterGrid
from sklearn.linear_model import SGDClassifier
import json
import gc

from paths_globals import *
from models.classifiers.mlp import TorchMLPClassifier
from tools import train_utils, data_utils
from tools.train_utils import (
    prepare_link_prediction_data,
    iter_link_prediction_train_batches,
    prepare_link_prediction_eval_data,
)


# # Limit BLAS/OMP Threads to avoid oversubscription
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


_WORKER_CACHE = {}
_LP_FEATURE_CACHE = {}
_LP_CACHE_DISABLED_NOTICE_EMITTED = set()


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
        "-dim", "--dimensions", nargs="+", type=int, default=EXPERIMENTS_DIMENSIONS_LIST, help="List of dimensions."
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        type=str,
        choices=DOWNSTREAM_TASKS,
        default=DOWNSTREAM_TASKS,
        help="Downstream Tasks to run",
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
        "--n_runs",
        type=int,
        default=EXPERIMENTS_NUM_DOWNSTREAM_CLF_RUNS,
        help="Dimensions to analyze.",
    )
    parser.add_argument(
        "--n_clf_control_runs",
        type=int,
        default=EXPERIMENTS_NUM_DOWNSTREAM_CLF_RUNS,
        help="Dimensions to analyze.",
    )
    parser.add_argument(
        "--n_clf_control_embeddings",
        type=int,
        default=EXPERIMENTS_NUM_CLF_CONTROL_EMBEDDINGS,
        help="Dimensions to analyze.",
    )
    parser.add_argument(
        "--downstream_performance_measures",
        nargs="+",
        type=str,
        choices=DOWNSTREAM_PERFORMANCE_MEASURES,
        default=DOWNSTREAM_PERFORMANCE_MEASURES,
        help="Dimensions to analyze.",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="Number of parallel processes to run",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Should existing results be overwritten?",
    )
    parser.add_argument(
        "--enable_lp_feature_cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable in-memory link-prediction feature cache.",
    )
    parser.add_argument(
        "--lp_cache_disable_dimension_threshold",
        type=int,
        default=DOWNSTREAM_LP_CACHE_DISABLE_DIMENSION_THRESHOLD_DEFAULT,
        help="Disable LP feature caching above this embedding dimension threshold.",
    )
    parser.add_argument(
        "--prediction_batch_size",
        type=int,
        default=DOWNSTREAM_PREDICTION_BATCH_SIZE_DEFAULT,
        help="Batch size used for batched predict/predict_proba during downstream evaluation.",
    )
    parser.add_argument(
        "--lp_train_batch_size",
        type=int,
        default=DOWNSTREAM_LP_TRAIN_BATCH_SIZE_DEFAULT,
        help=(
            "Batch size for streamed LP training for any classifier that supports streaming. "
            "Set to 0 to disable streamed LP training and use full in-memory X_train."
        ),
    )
    return parser.parse_args()


def validate_args(args):
    if args.n_jobs == 0:
        raise ValueError("--n_jobs must be non-zero.")
    if args.prediction_batch_size <= 0:
        raise ValueError("--prediction_batch_size must be positive.")
    if args.lp_train_batch_size < 0:
        raise ValueError("--lp_train_batch_size must be non-negative.")
    if args.lp_cache_disable_dimension_threshold < 0:
        raise ValueError("--lp_cache_disable_dimension_threshold must be non-negative.")


def _predict_in_batches(clf: Any, X: np.ndarray, batch_size: int = DOWNSTREAM_PREDICTION_BATCH_SIZE_DEFAULT) -> np.ndarray:
    if len(X) <= batch_size:
        return clf.predict(X)

    preds = [clf.predict(X[i : i + batch_size]) for i in range(0, len(X), batch_size)]
    return np.concatenate(preds, axis=0)


def _predict_proba_in_batches(clf: Any, X: np.ndarray, batch_size: int = DOWNSTREAM_PREDICTION_BATCH_SIZE_DEFAULT) -> np.ndarray:
    if len(X) <= batch_size:
        return clf.predict_proba(X)

    probs = [clf.predict_proba(X[i : i + batch_size]) for i in range(0, len(X), batch_size)]
    return np.concatenate(probs, axis=0)


def _fit_lp_logreg_in_batches(
    clf_params: Dict[str, Any],
    downstream_df: pd.DataFrame,
    edge_list: np.ndarray,
    emb: np.ndarray,
    seed: int,
    lp_train_batch_size: int,
) -> SGDClassifier:
    """Train LP logistic classifier incrementally with streamed feature batches."""
    max_iter = int(clf_params.get("max_iter", 1000))
    c_val = float(clf_params.get("C", 1.0))
    alpha = 1.0 / max(c_val, 1e-12)

    clf = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=alpha,
        max_iter=1,
        warm_start=True,
        random_state=seed,
        learning_rate="optimal",
        average=True,
    )

    classes = np.array([0, 1], dtype=np.int8)
    first_batch = True
    for _ in range(max_iter):
        for X_batch, y_batch in iter_link_prediction_train_batches(
            downstream_df=downstream_df,
            edge_list=edge_list,
            embedding=emb,
            batch_size=lp_train_batch_size,
            seed=seed,
        ):
            if first_batch:
                clf.partial_fit(X_batch, y_batch, classes=classes)
                first_batch = False
            else:
                clf.partial_fit(X_batch, y_batch)

    return clf


def tune_classifier_parallel(
    embedding_method: EMBEDDING_ALGORITHM,
    embedding_dim: int,
    dataset_params: Dict[str, Any],
    clf_name: DOWNSTREAM_CLASSIFIER,
    n_iterations: int,
    n_jobs: int,
    prediction_batch_size: int,
    lp_train_batch_size: int,
    enable_lp_feature_cache: bool,
    lp_cache_disable_dimension_threshold: int,
    result_dict: Dict[int, Dict[str, Any]] | None = None,
    overwrite: bool = False,
    checkpoint_file_path: str | None = None,
) -> Dict[int, Dict[str, Any]]:

    if result_dict is None:
        result_dict = {}

    # JSON deserialization turns dict keys into strings; normalize for robust resume behavior.
    result_dict = {int(k): v for k, v in result_dict.items()}

    param_grid = list(
        ParameterGrid(DOWNSTREAM_CLASSIFIER_DICT[clf_name][DOWNSTREAM_CLASSIFIER_DICT_TUNING_PARAMS_KEY])
    )
    pending_tune_setups = [(tune_id, params) for tune_id, params in enumerate(param_grid) if overwrite or tune_id not in result_dict]

    if len(pending_tune_setups) == 0:
        return result_dict

    with parallel_backend("loky", inner_max_num_threads=1):
        # Reuse one process pool across all tuning setups to reduce spawn/teardown overhead.
        parallel = Parallel(n_jobs=n_jobs, backend="loky", prefer="processes")

        tuning_setups_per_batch = max(1, (n_jobs + max(1, n_iterations) - 1) // max(1, n_iterations))

        for batch_start in range(0, len(pending_tune_setups), tuning_setups_per_batch):
            curr_tune_batch = pending_tune_setups[batch_start : batch_start + tuning_setups_per_batch]
            curr_tasks = []
            curr_params_by_tune_id = {}

            for tune_id, params in curr_tune_batch:
                print(f"Tuning Setup {tune_id + 1}/{len(param_grid)}")
                print("Current hyperparameter configuration:", ", ".join([f"{k}={v}" for k, v in params.items()]))

                curr_params = {
                    **DOWNSTREAM_CLASSIFIER_DICT[clf_name][DOWNSTREAM_CLASSIFIER_DICT_BASE_PARAMS_KEY],
                    **params,
                }
                curr_params_by_tune_id[tune_id] = params

                curr_tasks.extend(
                    [
                        delayed(_evaluate_single_embedding)(
                            i,
                            embedding_dim,
                            embedding_method,
                            dataset_params,
                            clf_name,
                            curr_params,
                            EXPERIMENTS_DEFAULT_SEED,
                            [ACCURACY_SCORE, MACRO_F1_SCORE, MICRO_F1_SCORE],
                            True,
                            prediction_batch_size=prediction_batch_size,
                            lp_train_batch_size=lp_train_batch_size,
                            enable_lp_feature_cache=enable_lp_feature_cache,
                            lp_cache_disable_dimension_threshold=lp_cache_disable_dimension_threshold,
                        )
                        for i in range(n_iterations)
                    ]
                )

            metrics = parallel(curr_tasks)

            for batch_idx, (tune_id, params) in enumerate(curr_tune_batch):
                start_idx = batch_idx * n_iterations
                end_idx = start_idx + n_iterations
                curr_metrics = metrics[start_idx:end_idx]

                accuracies = [m[0] for m in curr_metrics]
                micro_f1_scores = [m[1] for m in curr_metrics]
                macro_f1_scores = [m[2] for m in curr_metrics]

                results = {
                    ACCURACY_SCORE: accuracies,
                    MACRO_F1_SCORE: macro_f1_scores,
                    MICRO_F1_SCORE: micro_f1_scores,
                }
                tune_score = float(np.mean(accuracies))
                print("Average Accuracy for given hyperparameters:", tune_score)

                result_dict[tune_id] = {
                    TUNING_SUMMARY_PARAMS_KEY: curr_params_by_tune_id[tune_id],
                    TUNING_SUMMARY_RESULTS_KEY: results,
                    TUNING_SUMMARY_SCORE_KEY: tune_score,
                }

            if checkpoint_file_path is not None:
                os.makedirs(osp.dirname(checkpoint_file_path), exist_ok=True)
                with open(checkpoint_file_path, "w") as f:
                    json.dump(result_dict, f)

            del metrics, curr_tasks, curr_params_by_tune_id
            gc.collect()

    return result_dict


def tune_downstream_classifiers(
    embedding_method: EMBEDDING_ALGORITHM,
    dataset_params: Dict[str, Any],
    classifiers: List[DOWNSTREAM_CLASSIFIER],
    dimensions: List[int],
    n_jobs: int,
    prediction_batch_size: int,
    lp_train_batch_size: int,
    enable_lp_feature_cache: bool,
    lp_cache_disable_dimension_threshold: int,
    overwrite: bool = False,
) -> None:
    data_utils.load_dataset(dataset_params.copy())

    # Tuning should use the dedicated tuning iteration budget, not full experiment runs.
    n_iterations = TUNING_DEFAULT_ITERATIONS

    for embedding_dim in dimensions:
        print("Tune on embeddings of dimension", embedding_dim)

        print("Classifiers under consideration are", "'".join(classifiers))
        for clf_name in classifiers:
            print(f"Run {clf_name} classifier")
            tuning_results_file_path = osp.join(
                CREATE_DOWNSTREAM_RESULTS_PATH(
                    dataset_params,
                    embedding_method,
                    embedding_dim=embedding_dim,
                    clf_name=clf_name,
                ),
                TUNING_SUMMARY_FILE_NAME,
            )
            param_grid_size = len(
                list(ParameterGrid(DOWNSTREAM_CLASSIFIER_DICT[clf_name][DOWNSTREAM_CLASSIFIER_DICT_TUNING_PARAMS_KEY]))
            )

            existing_results = None
            if os.path.isfile(tuning_results_file_path) and not overwrite:
                with open(tuning_results_file_path, "r") as f:
                    existing_results = json.load(f)

                if len(existing_results) >= param_grid_size:
                    print("Classifier has already been tuned completely and overwrite is False, skipping.")
                    continue

                print(
                    f"Found partial tuning summary ({len(existing_results)}/{param_grid_size}) - resume tuning."
                )

            print("Start parameter tuning")
            tuning_summary = tune_classifier_parallel(
                embedding_method=embedding_method,
                embedding_dim=embedding_dim,
                dataset_params=dataset_params,
                clf_name=clf_name,
                n_iterations=n_iterations,
                n_jobs=n_jobs,
                prediction_batch_size=prediction_batch_size,
                lp_train_batch_size=lp_train_batch_size,
                enable_lp_feature_cache=enable_lp_feature_cache,
                lp_cache_disable_dimension_threshold=lp_cache_disable_dimension_threshold,
                result_dict=existing_results,
                overwrite=overwrite,
                checkpoint_file_path=tuning_results_file_path,
            )
            os.makedirs(os.path.dirname(tuning_results_file_path), exist_ok=True)
            with open(tuning_results_file_path, "w") as f:
                json.dump(tuning_summary, f)
            print("Tuning finished")


def _evaluate_single_embedding(
    iteration_id: int,
    embedding_dim: int,
    embedding_method: EMBEDDING_ALGORITHM,
    dataset_params: Dict[str, Any],
    clf_name: DOWNSTREAM_CLASSIFIER,
    clf_params: Dict[str, Any],
    seed: int,
    measures: List[DOWNSTREAM_PERFORMANCE_MEASURE],
    b_tune: bool = False,
    pred_save_path: str = None,
    ns_seed: int = EXPERIMENTS_DEFAULT_SEED,
    prediction_batch_size: int = DOWNSTREAM_PREDICTION_BATCH_SIZE_DEFAULT,
    lp_train_batch_size: int = DOWNSTREAM_LP_TRAIN_BATCH_SIZE_DEFAULT,
    enable_lp_feature_cache: bool = True,
    lp_cache_disable_dimension_threshold: int = DOWNSTREAM_LP_CACHE_DISABLE_DIMENSION_THRESHOLD_DEFAULT,
) -> List[float]:
    cache_key = dataset_params[CONFIG_DATASET_NAME_KEY]

    if cache_key not in _WORKER_CACHE:
        dataset_dir = BUILD_DATASET_SRC_DIR(dataset_params)
        edge_list = pd.read_csv(
            osp.join(dataset_dir, DATA_EDGE_LIST_DEFAULT_FILE_NAME),
            sep=" ",
            header=None,
        ).to_numpy()
        _WORKER_CACHE[cache_key] = {
            "downstream_df": pd.read_csv(
                osp.join(dataset_dir, DOWNSTREAM_TASK_DATA_FILE_NAME),
                index_col=0,
            ),
            "dataset_dir": dataset_dir,
            "edge_list": edge_list,
        }

    downstream_df = _WORKER_CACHE[cache_key]["downstream_df"]

    emb_dir = CREATE_MODELS_PATH(
        dataset_params=dataset_params, embedding_name=embedding_method, embedding_dim=embedding_dim
    )
    emb = np.load(osp.join(emb_dir, EMBEDDING_FILE_NAME(model_seed=iteration_id)), mmap_mode="r")

    task = DATASET_TASK_DICT[dataset_params[CONFIG_DATASET_NAME_KEY]]
    if task == LINK_PREDICTION:

        cache_key = (
            embedding_method,
            embedding_dim,
            iteration_id,
            ns_seed,
            b_tune,
        )

        edge_list = _WORKER_CACHE[dataset_params[CONFIG_DATASET_NAME_KEY]]["edge_list"]

        use_cache = enable_lp_feature_cache and embedding_dim <= lp_cache_disable_dimension_threshold
        if enable_lp_feature_cache and not use_cache and embedding_dim not in _LP_CACHE_DISABLED_NOTICE_EMITTED:
            print(
                f"LP feature cache auto-disabled for dim={embedding_dim} > "
                f"{lp_cache_disable_dimension_threshold} to reduce RAM."
            )
            _LP_CACHE_DISABLED_NOTICE_EMITTED.add(embedding_dim)

        use_streamed_lp_training = (
            task == LINK_PREDICTION and clf_name in [LOGISTIC_REGRESSION, MULTILAYER_PERCEPTRON] and lp_train_batch_size > 0
        )

        if use_streamed_lp_training:
            if b_tune:
                X_test, y_test = prepare_link_prediction_eval_data(
                    downstream_df=downstream_df,
                    embedding=emb,
                    split_category=DOWNSTREAM_TASK_DATA_VAL_CATEGORY,
                )
            else:
                X_test, y_test = prepare_link_prediction_eval_data(
                    downstream_df=downstream_df,
                    embedding=emb,
                    split_category=DOWNSTREAM_TASK_DATA_TEST_CATEGORY,
                )
            X_train = y_train = None
        elif (not use_cache) or cache_key not in _LP_FEATURE_CACHE:

            # if embedding is not cached (or caching is disabled), load it freshly
            X_train, y_train, X_test, y_test = prepare_link_prediction_data(
                downstream_df=downstream_df,
                edge_list=edge_list,
                embedding=emb,
                return_val_data=b_tune,
                return_test_data=not b_tune,
                seed=ns_seed,
            )

            if use_cache and ns_seed == EXPERIMENTS_DEFAULT_SEED:
                _LP_FEATURE_CACHE[cache_key] = (X_train, y_train, X_test, y_test)

        else:
            X_train, y_train, X_test, y_test = _LP_FEATURE_CACHE[cache_key]
    else:
        X_train, y_train, X_test, y_test = train_utils.prepare_node_classification_data(
            downstream_df=downstream_df,
            embedding=emb,
            return_val_data=b_tune,
            return_test_data=not b_tune,
        )

    clf_params_local = dict(clf_params)

    if clf_name == MULTILAYER_PERCEPTRON:
        input_dim = emb.shape[1] * 2 if (X_train is None) else int(X_train.shape[1])
        input_dim = int(input_dim)
        clf_params_local["hidden_layer_sizes"] = MLP_LAYER_DICT[input_dim]

    if task == LINK_PREDICTION and clf_name == LOGISTIC_REGRESSION and lp_train_batch_size > 0:
        clf = _fit_lp_logreg_in_batches(
            clf_params=clf_params_local,
            downstream_df=downstream_df,
            edge_list=edge_list,
            emb=emb,
            seed=seed,
            lp_train_batch_size=lp_train_batch_size,
        )
    elif clf_name == MULTILAYER_PERCEPTRON and task == LINK_PREDICTION:
        clf_params_local["random_state"] = seed
        clf = TorchMLPClassifier(**clf_params_local)
        if lp_train_batch_size > 0:
            batch_iterator_fn = lambda: iter_link_prediction_train_batches(
                downstream_df=downstream_df,
                edge_list=edge_list,
                embedding=emb,
                batch_size=lp_train_batch_size,
                seed=ns_seed,
            )
            clf.fit_streaming(batch_iterator_fn=batch_iterator_fn, input_dim=input_dim)
    else:
        np.random.seed(seed)
        clf = DOWNSTREAM_CLASSIFIER_DICT[clf_name][DOWNSTREAM_CLASSIFIER_DICT_CLF_KEY](**clf_params_local)

    if not (
        task == LINK_PREDICTION and clf_name in [LOGISTIC_REGRESSION, MULTILAYER_PERCEPTRON] and lp_train_batch_size > 0
    ):
        clf.fit(X_train, y_train)
    y_pred = _predict_in_batches(clf, X_test, batch_size=prediction_batch_size)

    if pred_save_path is not None:
        y_prob = _predict_proba_in_batches(clf, X_test, batch_size=prediction_batch_size)
        pred_matrix = np.concatenate((np.expand_dims(y_pred, axis=1), y_prob), axis=1)
        save_matrix = np.concatenate((np.expand_dims(y_test, axis=1), pred_matrix), axis=1)
        np.save(pred_save_path, arr=save_matrix)

    # garbage collection
    del emb, X_test
    if X_train is not None:
        del X_train
    if y_train is not None:
        del y_train
    gc.collect()

    return [DOWNSTREAM_MEASURE_DICT[measure](y_test, y_pred) for measure in measures]


def run_downstream_tasks(
    embedding_method: EMBEDDING_ALGORITHM,
    dataset_params: Dict[str, Any],
    classifiers: List[DOWNSTREAM_CLASSIFIER],
    dimensions: List[int],
    downstream_performance_measures: List[DOWNSTREAM_PERFORMANCE_MEASURE],
    prediction_batch_size: int,
    lp_train_batch_size: int,
    enable_lp_feature_cache: bool,
    lp_cache_disable_dimension_threshold: int,
    n_jobs: int,
    n_iterations: int,
    n_clf_control_runs: int,
    n_control_embeddings: int,
    overwrite: bool = False,
) -> None:
    dataset_name = dataset_params[CONFIG_DATASET_NAME_KEY]

    for embedding_dim in dimensions:
        print("Run on embeddings of dimension", embedding_dim)

        for clf_name in classifiers:

            results_dir = CREATE_DOWNSTREAM_RESULTS_PATH(
                dataset_params,
                embedding_method,
                embedding_dim=embedding_dim,
                clf_name=clf_name,
            )
            if (
                len(os.listdir(results_dir)) >= 2 * n_control_embeddings * n_clf_control_runs + n_iterations
                and not overwrite
            ):
                print(f"Predictions for {clf_name} classifier have already been computed, skip to next classifier.")
                continue

            print(f"Run {clf_name} classifier")
            tuning_results_file_path = osp.join(
                results_dir,
                TUNING_SUMMARY_FILE_NAME,
            )
            if not os.path.isfile(tuning_results_file_path):
                raise ValueError(f"{clf_name} classifier has not yet been tuned on {dataset_name} data.")

            with open(tuning_results_file_path, "r") as f:
                tuning_summary = json.load(f)

            print("Start prediction using optimal parameters.")
            tune_id_list = list(tuning_summary.keys())
            tune_scores = list(tuning_summary[tid][TUNING_SUMMARY_SCORE_KEY] for tid in tune_id_list)
            best_id = tune_id_list[tune_scores.index(max(tune_scores))]
            clf_params = {
                **DOWNSTREAM_CLASSIFIER_DICT[clf_name][DOWNSTREAM_CLASSIFIER_DICT_BASE_PARAMS_KEY],
                **tuning_summary[best_id][TUNING_SUMMARY_PARAMS_KEY],
            }

            performance_dict = dict({pm: dict() for pm in downstream_performance_measures})

            with parallel_backend("loky", inner_max_num_threads=1):
                parallel = Parallel(n_jobs=n_jobs, backend="loky", prefer="processes")

                # run models on fixed training date -> pure clf variability
                print(f"Predict on {n_iterations} different embeddings")
                performance_scores = parallel(
                    delayed(_evaluate_single_embedding)(
                        i,
                        embedding_dim,
                        embedding_method,
                        dataset_params,
                        clf_name,
                        clf_params,
                        EXPERIMENTS_DEFAULT_SEED,
                        downstream_performance_measures,
                        False,
                        osp.join(
                            results_dir,
                            DOWNSTREAM_PREDICTIONS_FILENAME(emb_id=i, model_seed=EXPERIMENTS_DEFAULT_SEED),
                        ),
                        prediction_batch_size=prediction_batch_size,
                        lp_train_batch_size=lp_train_batch_size,
                        enable_lp_feature_cache=enable_lp_feature_cache,
                        lp_cache_disable_dimension_threshold=lp_cache_disable_dimension_threshold,
                    )
                    for i in range(n_iterations)
                )
                print("Done")

            for i in range(n_iterations):
                for m_i, measure in enumerate(downstream_performance_measures):
                    performance_dict[measure][i] = dict({EXPERIMENTS_DEFAULT_SEED: performance_scores[i][m_i]})

            print("Control for classifier-based instability")
            parallel_tasks = [
                (i, model_seed) for model_seed in range(n_clf_control_runs) for i in range(n_control_embeddings)
            ]

            with parallel_backend("loky", inner_max_num_threads=1):
                parallel = Parallel(n_jobs=n_jobs, prefer="processes")
                performance_scores = parallel(
                    delayed(_evaluate_single_embedding)(
                        i,
                        embedding_dim,
                        embedding_method,
                        dataset_params,
                        clf_name,
                        clf_params,
                        model_seed,
                        downstream_performance_measures,
                        False,
                        osp.join(
                            results_dir,
                            DOWNSTREAM_PREDICTIONS_FILENAME(emb_id=i, model_seed=model_seed),
                        ),
                        prediction_batch_size=prediction_batch_size,
                        lp_train_batch_size=lp_train_batch_size,
                        enable_lp_feature_cache=enable_lp_feature_cache,
                        lp_cache_disable_dimension_threshold=lp_cache_disable_dimension_threshold,
                    )
                    for i, model_seed in parallel_tasks
                )

            for idx, (i, model_seed) in enumerate(parallel_tasks):
                for m_i, measure in enumerate(downstream_performance_measures):
                    performance_dict[measure][i][model_seed] = performance_scores[idx][m_i]

            with open(osp.join(results_dir, DOWNSTREAM_PERFORMANCE_JSON_FILE_NAME), "w") as f:
                json.dump(performance_dict, fp=f)

            if DATASET_TASK_DICT[dataset_name] == LINK_PREDICTION:

                # clear link prediction feature cache, from now, only new embeddings will be created
                _LP_FEATURE_CACHE.clear()

                print("Control for instability due to varying negative training samples")
                neg_sampling_performance_dict = dict(
                    {
                        pm: dict({ns: dict() for ns in range(n_control_embeddings)})
                        for pm in downstream_performance_measures
                    }
                )

                parallel_tasks = [
                    (i, negative_sampling_seed)
                    for negative_sampling_seed in range(n_control_embeddings)
                    for i in range(n_control_embeddings)
                ]

                with parallel_backend("loky", inner_max_num_threads=1):
                    parallel = Parallel(n_jobs=n_jobs, prefer="processes")
                    performance_scores = parallel(
                        delayed(_evaluate_single_embedding)(
                            i,
                            embedding_dim,
                            embedding_method,
                            dataset_params,
                            clf_name,
                            clf_params,
                            EXPERIMENTS_DEFAULT_SEED,
                            downstream_performance_measures,
                            False,
                            osp.join(
                                results_dir,
                                DOWNSTREAM_PREDICTIONS_FILENAME(
                                    emb_id=i,
                                    model_seed=EXPERIMENTS_DEFAULT_SEED,
                                    negative_sampling_seed=ns_seed,
                                ),
                            ),
                            ns_seed,
                            prediction_batch_size=prediction_batch_size,
                            lp_train_batch_size=lp_train_batch_size,
                            enable_lp_feature_cache=enable_lp_feature_cache,
                            lp_cache_disable_dimension_threshold=lp_cache_disable_dimension_threshold,
                        )
                        for i, ns_seed in parallel_tasks
                    )

                for idx, (i, ns_seed) in enumerate(parallel_tasks):
                    for m_i, measure in enumerate(downstream_performance_measures):
                        neg_sampling_performance_dict[measure][i][ns_seed] = performance_scores[idx][m_i]
                with open(osp.join(results_dir, DOWNSTREAM_PERFORMANCE_NS_JSON_FILE_NAME), "w") as f:
                    json.dump(neg_sampling_performance_dict, fp=f)

            print(f"Predictions for {clf_name} classifier are done")


if __name__ == "__main__":
    args = parse_args()
    validate_args(args)

    if not args.enable_lp_feature_cache:
        _LP_FEATURE_CACHE.clear()
        print("LP feature cache disabled (--no-enable-lp-feature-cache).")
    datasets = args.datasets
    algorithms = args.algorithms
    tasks = args.tasks

    for data_name in datasets:

        if data_name in EMPIRICAL_DATASET_LIST:
            num_iterations = EXPERIMENTS_NUM_ITERATIONS
        else:
            num_iterations = EXPERIMENTS_NUM_SYNTH_ITERATIONS

        for algorithm in algorithms:
            data_params = {CONFIG_DATASET_NAME_KEY: data_name, CONFIG_DATA_SUBSAMPLING_INDICATOR_KEY: False}

            dimensions_list = [d for d in args.dimensions if d <= MAX_DIMENSION_DICT[algorithm][data_name]]

            for embedding_dim in dimensions_list:
                print(
                    f"Tune downstream classifiers on {algorithm} embeddings of {data_name} "
                    f"dataset at dim={embedding_dim}."
                )
                tune_downstream_classifiers(
                    embedding_method=algorithm,
                    dataset_params=data_params,
                    classifiers=args.classifiers,
                    dimensions=[embedding_dim],
                    prediction_batch_size=args.prediction_batch_size,
                    lp_train_batch_size=args.lp_train_batch_size,
                    enable_lp_feature_cache=args.enable_lp_feature_cache,
                    lp_cache_disable_dimension_threshold=args.lp_cache_disable_dimension_threshold,
                    overwrite=args.overwrite,
                    n_jobs=args.n_jobs,
                )

                print(
                    f"Run downstream tasks on {algorithm} embeddings of {data_name} "
                    f"dataset at dim={embedding_dim}."
                )
                run_downstream_tasks(
                    embedding_method=algorithm,
                    dataset_params=data_params,
                    classifiers=args.classifiers,
                    dimensions=[embedding_dim],
                    downstream_performance_measures=args.downstream_performance_measures,
                    prediction_batch_size=args.prediction_batch_size,
                    lp_train_batch_size=args.lp_train_batch_size,
                    enable_lp_feature_cache=args.enable_lp_feature_cache,
                    lp_cache_disable_dimension_threshold=args.lp_cache_disable_dimension_threshold,
                    overwrite=args.overwrite,
                    n_jobs=args.n_jobs,
                    n_iterations=num_iterations,
                    n_clf_control_runs=args.n_clf_control_runs,
                    n_control_embeddings=args.n_clf_control_embeddings,
                )
