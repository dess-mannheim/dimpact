from joblib import Parallel, delayed, parallel_backend
import argparse
from argparse import Namespace
import pandas as pd
from sklearn.model_selection import ParameterGrid
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
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

LP_EDGE_FEATURE_OPS = ["hadamard", "concat"]


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
        default=0,
        help=(
            "Batch size for streamed LP training for any classifier that supports streaming. "
            "Default 0 disables streamed training; values >0 force streamed LP training."
        ),
    )
    parser.add_argument(
        "--lp_edge_feature_op",
        type=str,
        choices=LP_EDGE_FEATURE_OPS,
        default="hadamard",
        help="Feature operation for LP edge embeddings.",
    )
    parser.add_argument(
        "--lp_logreg_train_pos_sample_ratio",
        type=float,
        default=1.0,
        help="Fraction of LP positive training edges to use for logistic regression (0,1].",
    )
    parser.add_argument(
        "--lp_resample_negative_per_epoch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="For streamed LP training, resample negative edges every epoch/batch instead of using a fixed sampled set.",
    )
    return parser.parse_args()


def validate_args(args):
    if args.n_jobs == 0:
        raise ValueError("--n_jobs must be non-zero.")
    if args.prediction_batch_size <= 0:
        raise ValueError("--prediction_batch_size must be positive.")
    if args.lp_train_batch_size is not None and args.lp_train_batch_size < 0:
        raise ValueError("--lp_train_batch_size must be non-negative when specified.")
    if args.lp_cache_disable_dimension_threshold < 0:
        raise ValueError("--lp_cache_disable_dimension_threshold must be non-negative.")
    if not (0 < args.lp_logreg_train_pos_sample_ratio <= 1.0):
        raise ValueError("--lp_logreg_train_pos_sample_ratio must be in (0, 1].")


def _resolve_lp_streaming_mode(
    task: DOWNSTREAM_TASK,
    clf_name: DOWNSTREAM_CLASSIFIER,
    embedding_dim: int,
    lp_train_batch_size: int | None,
) -> tuple[bool, int]:
    lp_streaming_supported = task == LINK_PREDICTION and clf_name in [LOGISTIC_REGRESSION, MULTILAYER_PERCEPTRON]
    if not lp_streaming_supported:
        return False, 0

    if lp_train_batch_size is None:
        use_streamed = embedding_dim > DOWNSTREAM_LP_CACHE_DISABLE_DIMENSION_THRESHOLD_DEFAULT
        batch_size = DOWNSTREAM_LP_TRAIN_BATCH_SIZE_DEFAULT if use_streamed else 0
        return use_streamed, batch_size

    # explicit override: 0 forces unbatched, >0 forces streamed
    if lp_train_batch_size == 0:
        return False, 0
    return True, lp_train_batch_size


def _predict_in_batches(
    clf: Any, X: np.ndarray, batch_size: int = DOWNSTREAM_PREDICTION_BATCH_SIZE_DEFAULT
) -> np.ndarray:
    if len(X) <= batch_size:
        return clf.predict(X)

    preds = [clf.predict(X[i : i + batch_size]) for i in range(0, len(X), batch_size)]
    return np.concatenate(preds, axis=0)


def _predict_proba_in_batches(
    clf: Any, X: np.ndarray, batch_size: int = DOWNSTREAM_PREDICTION_BATCH_SIZE_DEFAULT
) -> np.ndarray:
    if len(X) <= batch_size:
        return clf.predict_proba(X)

    probs = [clf.predict_proba(X[i : i + batch_size]) for i in range(0, len(X), batch_size)]
    return np.concatenate(probs, axis=0)


def _should_standard_l2_scale_dgi_mlp_node_classification(
    embedding_method: EMBEDDING_ALGORITHM,
    clf_name: DOWNSTREAM_CLASSIFIER,
    task: DOWNSTREAM_TASK,
) -> bool:
    return embedding_method == DGI and clf_name == MULTILAYER_PERCEPTRON and task == NODE_CLASSIFICATION


def _standard_l2_scale_train_test_data(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    scaler = Pipeline([("standard", StandardScaler()), ("l2", Normalizer(norm="l2"))])
    return scaler.fit_transform(X_train), scaler.transform(X_test)


def _fit_lp_logreg_in_batches(
    clf_params: Dict[str, Any],
    downstream_df: pd.DataFrame,
    edge_list: np.ndarray,
    emb: np.ndarray,
    seed: int,
    lp_train_batch_size: int | None,
    lp_edge_feature_op: str,
    lp_logreg_train_pos_sample_ratio: float,
    lp_resample_negative_per_epoch: bool,
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
            feature_op=lp_edge_feature_op,
            train_pos_sample_ratio=lp_logreg_train_pos_sample_ratio,
            resample_negative_per_epoch=lp_resample_negative_per_epoch,
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
    lp_train_batch_size: int | None,
    enable_lp_feature_cache: bool,
    lp_cache_disable_dimension_threshold: int,
    lp_edge_feature_op: str,
    lp_logreg_train_pos_sample_ratio: float,
    lp_resample_negative_per_epoch: bool,
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
    pending_tune_setups = [
        (tune_id, params) for tune_id, params in enumerate(param_grid) if overwrite or tune_id not in result_dict
    ]

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
                            lp_edge_feature_op=lp_edge_feature_op,
                            lp_logreg_train_pos_sample_ratio=lp_logreg_train_pos_sample_ratio,
                            lp_resample_negative_per_epoch=lp_resample_negative_per_epoch,
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
    lp_train_batch_size: int | None,
    enable_lp_feature_cache: bool,
    lp_cache_disable_dimension_threshold: int,
    lp_edge_feature_op: str,
    lp_logreg_train_pos_sample_ratio: float,
    lp_resample_negative_per_epoch: bool,
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

                print(f"Found partial tuning summary ({len(existing_results)}/{param_grid_size}) - resume tuning.")

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
                lp_edge_feature_op=lp_edge_feature_op,
                lp_logreg_train_pos_sample_ratio=lp_logreg_train_pos_sample_ratio,
                lp_resample_negative_per_epoch=lp_resample_negative_per_epoch,
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
    lp_train_batch_size: int | None = None,
    enable_lp_feature_cache: bool = True,
    lp_cache_disable_dimension_threshold: int = DOWNSTREAM_LP_CACHE_DISABLE_DIMENSION_THRESHOLD_DEFAULT,
    lp_edge_feature_op: str = "hadamard",
    lp_logreg_train_pos_sample_ratio: float = 1.0,
    lp_resample_negative_per_epoch: bool = False,
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
            lp_edge_feature_op,
        )

        edge_list = _WORKER_CACHE[dataset_params[CONFIG_DATASET_NAME_KEY]]["edge_list"]

        use_cache = enable_lp_feature_cache and embedding_dim <= lp_cache_disable_dimension_threshold
        if enable_lp_feature_cache and not use_cache and embedding_dim not in _LP_CACHE_DISABLED_NOTICE_EMITTED:
            print(
                f"LP feature cache auto-disabled for dim={embedding_dim} > "
                f"{lp_cache_disable_dimension_threshold} to reduce RAM."
            )
            _LP_CACHE_DISABLED_NOTICE_EMITTED.add(embedding_dim)

        use_streamed_lp_training, effective_lp_train_batch_size = _resolve_lp_streaming_mode(
            task=task,
            clf_name=clf_name,
            embedding_dim=embedding_dim,
            lp_train_batch_size=lp_train_batch_size,
        )

        if use_streamed_lp_training:
            if b_tune:
                X_test, y_test = prepare_link_prediction_eval_data(
                    downstream_df=downstream_df,
                    embedding=emb,
                    split_category=DOWNSTREAM_TASK_DATA_VAL_CATEGORY,
                    feature_op=lp_edge_feature_op,
                )
            else:
                X_test, y_test = prepare_link_prediction_eval_data(
                    downstream_df=downstream_df,
                    embedding=emb,
                    split_category=DOWNSTREAM_TASK_DATA_TEST_CATEGORY,
                    feature_op=lp_edge_feature_op,
                )
            X_train = y_train = None
        elif (not use_cache) or cache_key not in _LP_FEATURE_CACHE:

            # if embedding is not cached (or caching is disabled), load it freshly
            curr_train_sample_ratio = lp_logreg_train_pos_sample_ratio if clf_name == LOGISTIC_REGRESSION else 1.0
            X_train, y_train, X_test, y_test = prepare_link_prediction_data(
                downstream_df=downstream_df,
                edge_list=edge_list,
                embedding=emb,
                return_val_data=b_tune,
                return_test_data=not b_tune,
                seed=ns_seed,
                feature_op=lp_edge_feature_op,
                train_pos_sample_ratio=curr_train_sample_ratio,
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

        if _should_standard_l2_scale_dgi_mlp_node_classification(embedding_method, clf_name, task):
            X_train, X_test = _standard_l2_scale_train_test_data(X_train, X_test)

    clf_params_local = dict(clf_params)

    if clf_name == MULTILAYER_PERCEPTRON:
        input_dim = emb.shape[1] if lp_edge_feature_op == "hadamard" else emb.shape[1] * 2
        if X_train is not None:
            input_dim = int(X_train.shape[1])
        input_dim = int(input_dim)
        clf_params_local["hidden_layer_sizes"] = MLP_LAYER_DICT[input_dim]

    clf_is_fitted = False

    if task == LINK_PREDICTION and clf_name == LOGISTIC_REGRESSION and use_streamed_lp_training:
        clf = _fit_lp_logreg_in_batches(
            clf_params=clf_params_local,
            downstream_df=downstream_df,
            edge_list=edge_list,
            emb=emb,
            seed=seed,
            lp_train_batch_size=effective_lp_train_batch_size,
            lp_edge_feature_op=lp_edge_feature_op,
            lp_logreg_train_pos_sample_ratio=lp_logreg_train_pos_sample_ratio,
            lp_resample_negative_per_epoch=lp_resample_negative_per_epoch,
        )
        clf_is_fitted = True
    elif clf_name == MULTILAYER_PERCEPTRON and task == LINK_PREDICTION:
        clf_params_local["random_state"] = seed
        clf = TorchMLPClassifier(**clf_params_local)
        if use_streamed_lp_training:
            batch_iterator_fn = lambda: iter_link_prediction_train_batches(
                downstream_df=downstream_df,
                edge_list=edge_list,
                embedding=emb,
                batch_size=effective_lp_train_batch_size,
                seed=ns_seed,
                feature_op=lp_edge_feature_op,
                resample_negative_per_epoch=lp_resample_negative_per_epoch,
            )
            clf.fit_streaming(batch_iterator_fn=batch_iterator_fn, input_dim=input_dim)
            clf_is_fitted = True
    else:
        np.random.seed(seed)
        clf = DOWNSTREAM_CLASSIFIER_DICT[clf_name][DOWNSTREAM_CLASSIFIER_DICT_CLF_KEY](**clf_params_local)

    if not clf_is_fitted:
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


def _load_scores_from_prediction_file(pred_path: str, measures: List[DOWNSTREAM_PERFORMANCE_MEASURE]) -> List[float]:
    prediction_matrix = np.load(pred_path, mmap_mode="r")
    y_test = prediction_matrix[:, 0]
    y_pred = prediction_matrix[:, 1]
    return [DOWNSTREAM_MEASURE_DICT[measure](y_test, y_pred) for measure in measures]


def run_downstream_tasks(
    embedding_method: EMBEDDING_ALGORITHM,
    dataset_params: Dict[str, Any],
    classifiers: List[DOWNSTREAM_CLASSIFIER],
    dimensions: List[int],
    downstream_performance_measures: List[DOWNSTREAM_PERFORMANCE_MEASURE],
    prediction_batch_size: int,
    lp_train_batch_size: int | None,
    enable_lp_feature_cache: bool,
    lp_cache_disable_dimension_threshold: int,
    lp_edge_feature_op: str,
    lp_logreg_train_pos_sample_ratio: float,
    lp_resample_negative_per_epoch: bool,
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
            is_link_prediction = DATASET_TASK_DICT[dataset_name] == LINK_PREDICTION
            performance_dict_path = osp.join(results_dir, DOWNSTREAM_PERFORMANCE_JSON_FILE_NAME)
            neg_sampling_performance_dict_path = osp.join(results_dir, DOWNSTREAM_PERFORMANCE_NS_JSON_FILE_NAME)

            expected_prediction_paths = [
                osp.join(
                    results_dir,
                    DOWNSTREAM_PREDICTIONS_FILENAME(emb_id=i, model_seed=EXPERIMENTS_DEFAULT_SEED),
                )
                for i in range(n_iterations)
            ]
            expected_prediction_paths.extend(
                [
                    osp.join(
                        results_dir,
                        DOWNSTREAM_PREDICTIONS_FILENAME(emb_id=i, model_seed=model_seed),
                    )
                    for model_seed in range(n_clf_control_runs)
                    for i in range(n_control_embeddings)
                ]
            )
            if is_link_prediction:
                expected_prediction_paths.extend(
                    [
                        osp.join(
                            results_dir,
                            DOWNSTREAM_PREDICTIONS_FILENAME(
                                emb_id=i,
                                model_seed=EXPERIMENTS_DEFAULT_SEED,
                                negative_sampling_seed=ns_seed,
                            ),
                        )
                        for ns_seed in range(n_control_embeddings)
                        for i in range(n_control_embeddings)
                    ]
                )

            all_predictions_exist = all(osp.isfile(path) for path in expected_prediction_paths)
            all_summary_files_exist = osp.isfile(performance_dict_path) and (
                (not is_link_prediction) or osp.isfile(neg_sampling_performance_dict_path)
            )
            if not overwrite and all_predictions_exist and all_summary_files_exist:
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
            tune_id_list = list(tuning_summary.keys())
            tune_scores = list(tuning_summary[tid][TUNING_SUMMARY_SCORE_KEY] for tid in tune_id_list)
            best_id = tune_id_list[tune_scores.index(max(tune_scores))]
            clf_params = {
                **DOWNSTREAM_CLASSIFIER_DICT[clf_name][DOWNSTREAM_CLASSIFIER_DICT_BASE_PARAMS_KEY],
                **tuning_summary[best_id][TUNING_SUMMARY_PARAMS_KEY],
            }
            print("Start prediction using optimal parameters.")

            performance_dict = dict({pm: dict() for pm in downstream_performance_measures})

            # run models on fixed training date -> pure clf variability
            print(f"Predict on {n_iterations} different embeddings")
            fixed_embedding_tasks = [
                (
                    i,
                    osp.join(
                        results_dir,
                        DOWNSTREAM_PREDICTIONS_FILENAME(emb_id=i, model_seed=EXPERIMENTS_DEFAULT_SEED),
                    ),
                )
                for i in range(n_iterations)
            ]
            fixed_embedding_scores: Dict[int, List[float]] = {}
            missing_fixed_embedding_tasks = []
            for i, pred_path in fixed_embedding_tasks:
                if not overwrite and osp.isfile(pred_path):
                    fixed_embedding_scores[i] = _load_scores_from_prediction_file(
                        pred_path, downstream_performance_measures
                    )
                else:
                    missing_fixed_embedding_tasks.append((i, pred_path))

            if len(missing_fixed_embedding_tasks) > 0:
                with parallel_backend("loky", inner_max_num_threads=1):
                    parallel = Parallel(n_jobs=n_jobs, backend="loky", prefer="processes")
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
                            pred_path,
                            prediction_batch_size=prediction_batch_size,
                            lp_train_batch_size=lp_train_batch_size,
                            enable_lp_feature_cache=enable_lp_feature_cache,
                            lp_cache_disable_dimension_threshold=lp_cache_disable_dimension_threshold,
                            lp_edge_feature_op=lp_edge_feature_op,
                            lp_logreg_train_pos_sample_ratio=lp_logreg_train_pos_sample_ratio,
                            lp_resample_negative_per_epoch=lp_resample_negative_per_epoch,
                        )
                        for i, pred_path in missing_fixed_embedding_tasks
                    )
                for score_idx, (i, _) in enumerate(missing_fixed_embedding_tasks):
                    fixed_embedding_scores[i] = performance_scores[score_idx]
                print("Done")
            else:
                print("Done (reused existing prediction files)")

            for i, _ in fixed_embedding_tasks:
                for m_i, measure in enumerate(downstream_performance_measures):
                    performance_dict[measure][i] = dict({EXPERIMENTS_DEFAULT_SEED: fixed_embedding_scores[i][m_i]})

            print("Control for classifier-based instability")
            control_tasks = [
                (
                    i,
                    model_seed,
                    osp.join(
                        results_dir,
                        DOWNSTREAM_PREDICTIONS_FILENAME(emb_id=i, model_seed=model_seed),
                    ),
                )
                for model_seed in range(n_clf_control_runs)
                for i in range(n_control_embeddings)
            ]
            control_scores: Dict[tuple[int, int], List[float]] = {}
            missing_control_tasks = []
            for i, model_seed, pred_path in control_tasks:
                if not overwrite and osp.isfile(pred_path):
                    control_scores[(i, model_seed)] = _load_scores_from_prediction_file(
                        pred_path, downstream_performance_measures
                    )
                else:
                    missing_control_tasks.append((i, model_seed, pred_path))

            if len(missing_control_tasks) > 0:
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
                            pred_path,
                            prediction_batch_size=prediction_batch_size,
                            lp_train_batch_size=lp_train_batch_size,
                            enable_lp_feature_cache=enable_lp_feature_cache,
                            lp_cache_disable_dimension_threshold=lp_cache_disable_dimension_threshold,
                            lp_edge_feature_op=lp_edge_feature_op,
                            lp_logreg_train_pos_sample_ratio=lp_logreg_train_pos_sample_ratio,
                            lp_resample_negative_per_epoch=lp_resample_negative_per_epoch,
                        )
                        for i, model_seed, pred_path in missing_control_tasks
                    )
                for score_idx, (i, model_seed, _) in enumerate(missing_control_tasks):
                    control_scores[(i, model_seed)] = performance_scores[score_idx]

            for i, model_seed, _ in control_tasks:
                for m_i, measure in enumerate(downstream_performance_measures):
                    performance_dict[measure].setdefault(i, {})
                    performance_dict[measure][i][model_seed] = control_scores[(i, model_seed)][m_i]

            with open(performance_dict_path, "w") as f:
                json.dump(performance_dict, fp=f)

            if is_link_prediction:

                # clear link prediction feature cache, from now, only new embeddings will be created
                _LP_FEATURE_CACHE.clear()

                print("Control for instability due to varying negative training samples")
                neg_sampling_performance_dict = dict(
                    {
                        pm: dict({ns: dict() for ns in range(n_control_embeddings)})
                        for pm in downstream_performance_measures
                    }
                )

                neg_sampling_tasks = [
                    (
                        i,
                        ns_seed,
                        osp.join(
                            results_dir,
                            DOWNSTREAM_PREDICTIONS_FILENAME(
                                emb_id=i,
                                model_seed=EXPERIMENTS_DEFAULT_SEED,
                                negative_sampling_seed=ns_seed,
                            ),
                        ),
                    )
                    for ns_seed in range(n_control_embeddings)
                    for i in range(n_control_embeddings)
                ]
                neg_sampling_scores: Dict[tuple[int, int], List[float]] = {}
                missing_neg_sampling_tasks = []
                for i, ns_seed, pred_path in neg_sampling_tasks:
                    if not overwrite and osp.isfile(pred_path):
                        neg_sampling_scores[(i, ns_seed)] = _load_scores_from_prediction_file(
                            pred_path, downstream_performance_measures
                        )
                    else:
                        missing_neg_sampling_tasks.append((i, ns_seed, pred_path))

                if len(missing_neg_sampling_tasks) > 0:
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
                                pred_path,
                                ns_seed,
                                prediction_batch_size=prediction_batch_size,
                                lp_train_batch_size=lp_train_batch_size,
                                enable_lp_feature_cache=enable_lp_feature_cache,
                                lp_cache_disable_dimension_threshold=lp_cache_disable_dimension_threshold,
                                lp_edge_feature_op=lp_edge_feature_op,
                                lp_logreg_train_pos_sample_ratio=lp_logreg_train_pos_sample_ratio,
                                lp_resample_negative_per_epoch=lp_resample_negative_per_epoch,
                            )
                            for i, ns_seed, pred_path in missing_neg_sampling_tasks
                        )
                    for score_idx, (i, ns_seed, _) in enumerate(missing_neg_sampling_tasks):
                        neg_sampling_scores[(i, ns_seed)] = performance_scores[score_idx]

                for i, ns_seed, _ in neg_sampling_tasks:
                    for m_i, measure in enumerate(downstream_performance_measures):
                        neg_sampling_performance_dict[measure][i][ns_seed] = neg_sampling_scores[(i, ns_seed)][m_i]
                with open(neg_sampling_performance_dict_path, "w") as f:
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
                    lp_edge_feature_op=args.lp_edge_feature_op,
                    lp_logreg_train_pos_sample_ratio=args.lp_logreg_train_pos_sample_ratio,
                    lp_resample_negative_per_epoch=args.lp_resample_negative_per_epoch,
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
                    lp_edge_feature_op=args.lp_edge_feature_op,
                    lp_logreg_train_pos_sample_ratio=args.lp_logreg_train_pos_sample_ratio,
                    lp_resample_negative_per_epoch=args.lp_resample_negative_per_epoch,
                    overwrite=args.overwrite,
                    n_jobs=args.n_jobs,
                    n_iterations=num_iterations,
                    n_clf_control_runs=args.n_clf_control_runs,
                    n_control_embeddings=args.n_clf_control_embeddings,
                )
