from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import subprocess
import threading
import time
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
from types import TracebackType

import psutil

from paths_globals import *


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure wall-clock runtime and process RSS memory while computing individual embeddings."
    )
    parser.add_argument(
        "-a",
        "--algorithms",
        nargs="+",
        type=str,
        choices=EMBEDDING_COSTS_SUPPORTED_ALGORITHMS,
        required=True,
        help="Embedding algorithms to run.",
    )
    parser.add_argument(
        "-d",
        "--datasets",
        nargs="+",
        type=str,
        choices=EMPIRICAL_DATASET_LIST,
        default=EMPIRICAL_DATASET_LIST,
        help="Empirical datasets used in the case study.",
    )
    parser.add_argument(
        "-dim",
        "--dimensions",
        nargs="+",
        type=int,
        default=EXPERIMENTS_DIMENSIONS_LIST,
        help="Embedding dimensions to measure.",
    )
    parser.add_argument(
        "-n",
        "--num_embeddings",
        type=int,
        default=1,
        help="Number of independently seeded embeddings to compute per dataset/algorithm/dimension.",
    )
    parser.add_argument("--seed_start", type=int, default=0, help="First training seed used for measured embeddings.")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of worker threads used by methods that support it.")
    parser.add_argument(
        "-c",
        "--classifiers",
        nargs="+",
        type=str,
        choices=DOWNSTREAM_CLASSIFIERS,
        default=[LOGISTIC_REGRESSION],
        help="Downstream classifiers to time after each embedding is produced.",
    )
    parser.add_argument(
        "--downstream_performance_measures",
        nargs="+",
        type=str,
        choices=DOWNSTREAM_PERFORMANCE_MEASURES,
        default=DOWNSTREAM_PERFORMANCE_MEASURES,
        help="Downstream performance measures to compute.",
    )
    parser.add_argument(
        "--skip_downstream",
        action="store_true",
        help="Only measure embedding computation and skip the downstream evaluation pass.",
    )
    parser.add_argument(
        "--prediction_batch_size",
        type=int,
        default=DOWNSTREAM_PREDICTION_BATCH_SIZE_DEFAULT,
        help="Batch size used for batched downstream prediction.",
    )
    parser.add_argument(
        "--lp_train_batch_size",
        type=int,
        default=None,
        help=(
            "Batch size for streamed link-prediction training. If omitted, streamed mode follows "
            "run_downstream_tasks.py defaults."
        ),
    )
    parser.add_argument(
        "--lp_cache_disable_dimension_threshold",
        type=int,
        default=DOWNSTREAM_LP_CACHE_DISABLE_DIMENSION_THRESHOLD_DEFAULT,
        help="Dimension threshold used when deciding streamed link-prediction evaluation.",
    )
    parser.add_argument(
        "--lp_edge_feature_op",
        type=str,
        choices=["hadamard", "concat"],
        default="hadamard",
        help="Feature operation for link-prediction edge embeddings.",
    )
    parser.add_argument(
        "--lp_logreg_train_pos_sample_ratio",
        type=float,
        default=1.0,
        help="Fraction of positive LP training edges used for logistic regression.",
    )
    parser.add_argument(
        "--lp_resample_negative_per_epoch",
        action="store_true",
        help="For streamed LP training, resample negative edges every epoch/batch instead of using a fixed sample.",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Optional output run identifier. Defaults to a timestamped directory.",
    )
    parser.add_argument(
        "--sample_interval",
        type=float,
        default=0.2,
        help="Seconds between process-memory samples.",
    )
    return parser.parse_args()


def _rss_for_process_tree(pid: int) -> int:
    try:
        root = psutil.Process(pid)
    except psutil.Error:
        return 0

    total = 0
    processes = [root]
    try:
        processes.extend(root.children(recursive=True))
    except psutil.Error:
        pass

    for proc in processes:
        try:
            total += proc.memory_info().rss
        except psutil.Error:
            continue
    return total


class MemorySampler:
    def __init__(self, pid: int, interval: float) -> None:
        self.pid = pid
        self.interval = interval
        self.start_rss_bytes = 0
        self.peak_rss_bytes = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)

    def __enter__(self) -> "MemorySampler":
        first_sample = _rss_for_process_tree(self.pid)
        self.start_rss_bytes = first_sample
        self.peak_rss_bytes = first_sample
        self._thread.start()
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self._stop.set()
        self._thread.join()
        self.peak_rss_bytes = max(self.peak_rss_bytes, _rss_for_process_tree(self.pid))

    def _sample_loop(self) -> None:
        while not self._stop.is_set():
            self.peak_rss_bytes = max(self.peak_rss_bytes, _rss_for_process_tree(self.pid))
            self._stop.wait(self.interval)

    @property
    def peak_rss_delta_bytes(self) -> int:
        return max(0, self.peak_rss_bytes - self.start_rss_bytes)


def _measure_callable(
    fn: Callable[[], float | None],
    log_path: Path,
    sample_interval: float,
    measure_cuda: bool,
) -> dict[str, Any]:
    cuda_enabled = False
    if measure_cuda:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                cuda_enabled = True
        except Exception:
            cuda_enabled = False

    start = time.perf_counter()
    with MemorySampler(os.getpid(), sample_interval) as sampler:
        with log_path.open("w", encoding="utf-8") as log_file:
            with redirect_stdout(log_file), redirect_stderr(log_file):
                score = fn()
    elapsed = time.perf_counter() - start
    peak_cuda_allocated_bytes = None
    if cuda_enabled:
        try:
            import torch

            if torch.cuda.is_available():
                peak_cuda_allocated_bytes = int(torch.cuda.max_memory_allocated())
        except Exception:
            peak_cuda_allocated_bytes = None

    return {
        "elapsed_seconds": elapsed,
        "start_rss_bytes": sampler.start_rss_bytes,
        "peak_rss_bytes": sampler.peak_rss_bytes,
        "peak_rss_delta_bytes": sampler.peak_rss_delta_bytes,
        "peak_cuda_allocated_bytes": peak_cuda_allocated_bytes,
        "validation_score": score,
    }


def _write_report_set(
    rows: list[dict[str, Any]],
    csv_path: Path,
    summary_csv_path: Path,
    summary_builder: Callable[[list[dict[str, Any]]], list[dict[str, Any]]],
) -> None:
    if rows:
        fieldnames = list(rows[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    summary_rows = summary_builder(rows)
    if summary_rows:
        with summary_csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _std(values: list[float]) -> float | None:
    return float(np.std(values, ddof=1)) if len(values) > 1 else None


def _build_embedding_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["dataset"], row["algorithm"], row["dimension"])
        grouped.setdefault(key, []).append(row)

    summary_rows = []
    for (dataset, algorithm, dimension), group_rows in sorted(grouped.items()):
        successful = [row for row in group_rows if row["status"] == "success"]
        elapsed = [float(row["elapsed_seconds"]) for row in successful]
        peak_rss = [float(row["peak_rss_bytes"]) for row in successful]
        peak_delta = [float(row["peak_rss_delta_bytes"]) for row in successful]
        summary_rows.append(
            {
                "dataset": dataset,
                "algorithm": algorithm,
                "dimension": dimension,
                "num_requested": len(group_rows),
                "num_successful": len(successful),
                "elapsed_seconds_mean": _mean(elapsed),
                "elapsed_seconds_std": _std(elapsed),
                "peak_rss_bytes_mean": _mean(peak_rss),
                "peak_rss_bytes_std": _std(peak_rss),
                "peak_rss_delta_bytes_mean": _mean(peak_delta),
                "peak_rss_delta_bytes_std": _std(peak_delta),
            }
        )
    return summary_rows


def _build_downstream_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["dataset"], row["algorithm"], row["dimension"], row["classifier"])
        grouped.setdefault(key, []).append(row)

    summary_rows = []
    for (dataset, algorithm, dimension, classifier), group_rows in sorted(grouped.items()):
        successful = [row for row in group_rows if row["status"] == "success"]
        elapsed = [float(row["elapsed_seconds"]) for row in successful]
        peak_rss = [float(row["peak_rss_bytes"]) for row in successful]
        peak_delta = [float(row["peak_rss_delta_bytes"]) for row in successful]
        summary_row = {
            "dataset": dataset,
            "algorithm": algorithm,
            "dimension": dimension,
            "classifier": classifier,
            "num_requested": len(group_rows),
            "num_successful": len(successful),
            "elapsed_seconds_mean": _mean(elapsed),
            "elapsed_seconds_std": _std(elapsed),
            "peak_rss_bytes_mean": _mean(peak_rss),
            "peak_rss_bytes_std": _std(peak_rss),
            "peak_rss_delta_bytes_mean": _mean(peak_delta),
            "peak_rss_delta_bytes_std": _std(peak_delta),
        }
        for measure in DOWNSTREAM_PERFORMANCE_MEASURES:
            values = [float(row[measure]) for row in successful if row.get(measure) is not None]
            if values:
                summary_row[f"{measure}_mean"] = _mean(values)
                summary_row[f"{measure}_std"] = _std(values)
        summary_rows.append(summary_row)
    return summary_rows


def _run_external_embedding(
    algorithm: EMBEDDING_ALGORITHM,
    edge_list_path: str,
    data: Any,
    dataset_params: dict[str, Any],
    config: dict[str, Any],
    seed: int,
    save_dir: Path,
    run_dir: Path,
    log_path: Path,
    n_jobs: int,
    sample_interval: float,
) -> float | None:
    config = copy.deepcopy(config)
    config[CONFIG_TRAINING_SEEDS_KEY] = [seed]
    config[CONFIG_ITERATIONS_KEY] = 1

    config_dir = run_dir / CONFIGS_DIR_NAME
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / EMBEDDING_COSTS_CONFIG_FILE_TEMPLATE.format(
        algorithm=algorithm,
        dataset=dataset_params[CONFIG_DATASET_NAME_KEY],
        dimension=config[CONFIG_DIMENSION_KEY],
        seed=seed,
    )
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

    data_dir = BUILD_DATASET_SRC_DIR(dataset_params)
    downstream_path = osp.join(data_dir, DOWNSTREAM_TASK_DATA_FILE_NAME)
    command = [
        "python",
        "-m",
        MODULE_NAME_DICT[algorithm],
        "--data_path",
        edge_list_path,
        "--config_path",
        str(config_path),
        "--models_dir",
        str(save_dir),
        "--downstream_data_path",
        downstream_path,
        "--overwrite",
    ]

    if algorithm == ASNE:
        feature_matrix_path = osp.join(osp.dirname(edge_list_path), DATA_FEATURE_MATRIX_DEFAULT_FILE_NAME)
        if not osp.isfile(feature_matrix_path):
            np.save(feature_matrix_path, data.x.numpy())
        command.extend(["--n_jobs", str(n_jobs), "--feature_matrix_path", feature_matrix_path])

    from tools import train_utils

    env = train_utils.get_environment(ENVIRONMENTS_DICT[algorithm])
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(command, env=env, stdout=log_file, stderr=subprocess.STDOUT)
        with MemorySampler(process.pid, sample_interval) as sampler:
            return_code = process.wait()
    elapsed = time.perf_counter() - start
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)
    _measurements = {
        "elapsed_seconds": elapsed,
        "start_rss_bytes": sampler.start_rss_bytes,
        "peak_rss_bytes": sampler.peak_rss_bytes,
        "peak_rss_delta_bytes": sampler.peak_rss_delta_bytes,
        "peak_cuda_allocated_bytes": None,
    }

    results_path = save_dir / TMP_TUNING_RESULTS_FILE_NAME(None)
    score = None
    if results_path.is_file():
        with results_path.open("r", encoding="utf-8") as f:
            results = json.load(f)
        score = results.get(str(seed), results.get(seed))
        results_path.unlink()

    _run_external_embedding.last_measurements = _measurements
    return score


_run_external_embedding.last_measurements = {}


def _run_single_embedding(
    algorithm: EMBEDDING_ALGORITHM,
    dataset_params: dict[str, Any],
    data: Any,
    edge_list_path: str,
    config: dict[str, Any],
    seed: int,
    embedding_index: int,
    save_dir: Path,
    run_dir: Path,
    logs_dir: Path,
    n_jobs: int,
    sample_interval: float,
) -> dict[str, Any]:
    dimension = config[CONFIG_DIMENSION_KEY]
    log_path = logs_dir / EMBEDDING_RUN_LOG_FILE_TEMPLATE.format(
        algorithm=algorithm,
        dataset=dataset_params[CONFIG_DATASET_NAME_KEY],
        dimension=dimension,
        seed=seed,
    )
    embedding_path = save_dir / EMBEDDING_FILE_NAME(model_seed=seed)
    model_path = save_dir / MODEL_FILE_NAME(model_seed=seed)

    row = {
        "run_id": run_dir.name,
        "measurement_scope": EMBEDDING_COSTS_MEASUREMENT_SCOPE,
        "dataset": dataset_params[CONFIG_DATASET_NAME_KEY],
        "algorithm": algorithm,
        "dimension": dimension,
        "embedding_index": embedding_index,
        "seed": seed,
        "status": "success",
        "elapsed_seconds": None,
        "start_rss_bytes": None,
        "peak_rss_bytes": None,
        "peak_rss_delta_bytes": None,
        "peak_cuda_allocated_bytes": None,
        "validation_score": None,
        "embedding_path": str(embedding_path),
        "log_path": str(log_path),
        "error": None,
    }

    try:
        if algorithm == GRAPHSAGE:
            from models.pyg import graphsage

            os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
            os.environ["TORCH_USE_CUDA_DSA"] = "1"
            measurements = _measure_callable(
                lambda: graphsage.train_model(
                    dataset=data,
                    embedding_dim=dimension,
                    config=copy.deepcopy(config),
                    save_path=str(model_path),
                    seed=seed,
                    embedding_path=str(embedding_path),
                ),
                log_path=log_path,
                sample_interval=sample_interval,
                measure_cuda=True,
            )
        elif algorithm == DGI:
            from models.pyg import dgi_inductive as deep_graph_infomax

            os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
            os.environ["TORCH_USE_CUDA_DSA"] = "1"
            measurements = _measure_callable(
                lambda: deep_graph_infomax.train_model(
                    dataset=data,
                    embedding_dim=dimension,
                    config=copy.deepcopy(config),
                    save_path=str(model_path),
                    seed=seed,
                    embedding_path=str(embedding_path),
                ),
                log_path=log_path,
                sample_interval=sample_interval,
                measure_cuda=True,
            )
        elif algorithm == VERSE:
            from models.verse import verse

            data_dir = BUILD_DATASET_SRC_DIR(dataset_params)
            downstream_path = osp.join(data_dir, DOWNSTREAM_TASK_DATA_FILE_NAME)
            measurements = _measure_callable(
                lambda: verse.train_model(
                    edge_list_path=edge_list_path,
                    embedding_config=copy.deepcopy(config),
                    save_path=str(embedding_path),
                    downstream_path=downstream_path,
                    seed=seed,
                    n_jobs=n_jobs,
                ),
                log_path=log_path,
                sample_interval=sample_interval,
                measure_cuda=False,
            )
        elif algorithm in [NODE2VEC, ASNE]:
            score = _run_external_embedding(
                algorithm=algorithm,
                edge_list_path=edge_list_path,
                data=data,
                dataset_params=dataset_params,
                config=config,
                seed=seed,
                save_dir=save_dir,
                run_dir=run_dir,
                log_path=log_path,
                n_jobs=n_jobs,
                sample_interval=sample_interval,
            )
            measurements = dict(_run_external_embedding.last_measurements)
            measurements["validation_score"] = score
        else:
            raise ValueError(f"Unsupported algorithm for this case study: {algorithm}")

        row.update(measurements)
    except Exception as exc:
        row["status"] = "failed"
        row["error"] = repr(exc)

    return row


def _load_best_downstream_classifier_params(
    dataset_params: dict[str, Any],
    algorithm: EMBEDDING_ALGORITHM,
    dimension: int,
    classifier: DOWNSTREAM_CLASSIFIER,
) -> dict[str, Any]:
    results_dir = CREATE_DOWNSTREAM_RESULTS_PATH(
        dataset_params=dataset_params,
        embedding_name=algorithm,
        clf_name=classifier,
        embedding_dim=dimension,
    )
    tuning_results_file_path = osp.join(results_dir, TUNING_SUMMARY_FILE_NAME)
    if not osp.isfile(tuning_results_file_path):
        raise FileNotFoundError(
            f"Downstream tuning results are missing for {classifier} on {algorithm}/{dataset_params[CONFIG_DATASET_NAME_KEY]} "
            f"at dim={dimension}: {tuning_results_file_path}"
        )

    with open(tuning_results_file_path, "r", encoding="utf-8") as f:
        tuning_summary = json.load(f)
    tune_id_list = list(tuning_summary.keys())
    tune_scores = [tuning_summary[tid][TUNING_SUMMARY_SCORE_KEY] for tid in tune_id_list]
    best_id = tune_id_list[tune_scores.index(max(tune_scores))]
    return {
        **DOWNSTREAM_CLASSIFIER_DICT[classifier][DOWNSTREAM_CLASSIFIER_DICT_BASE_PARAMS_KEY],
        **tuning_summary[best_id][TUNING_SUMMARY_PARAMS_KEY],
    }


def _run_downstream_evaluation(
    dataset_params: dict[str, Any],
    algorithm: EMBEDDING_ALGORITHM,
    dimension: int,
    embedding_path: Path,
    classifier: DOWNSTREAM_CLASSIFIER,
    classifier_params: dict[str, Any],
    measures: list[DOWNSTREAM_PERFORMANCE_MEASURE],
    seed: int,
    ns_seed: int,
    prediction_batch_size: int,
    lp_train_batch_size: int | None,
    lp_cache_disable_dimension_threshold: int,
    lp_edge_feature_op: LP_EDGE_FEATURE_OP,
    lp_logreg_train_pos_sample_ratio: float,
    lp_resample_negative_per_epoch: bool,
) -> dict[str, float]:
    import pandas as pd
    from run_downstream_tasks import (
        _fit_lp_logreg_in_batches,
        _predict_in_batches,
        _should_standard_l2_scale_dgi_mlp_node_classification,
        _standard_l2_scale_train_test_data,
    )
    from tools import train_utils
    from tools.train_utils import iter_link_prediction_train_batches, prepare_link_prediction_eval_data

    dataset_dir = BUILD_DATASET_SRC_DIR(dataset_params)
    downstream_df = pd.read_csv(osp.join(dataset_dir, DOWNSTREAM_TASK_DATA_FILE_NAME), index_col=0)
    edge_list = pd.read_csv(osp.join(dataset_dir, DATA_EDGE_LIST_DEFAULT_FILE_NAME), sep=" ", header=None).to_numpy()
    emb = np.load(embedding_path, mmap_mode="r")
    task = DATASET_TASK_DICT[dataset_params[CONFIG_DATASET_NAME_KEY]]

    clf_params_local = dict(classifier_params)
    if task == LINK_PREDICTION:
        lp_streaming_supported = classifier in [LOGISTIC_REGRESSION, MULTILAYER_PERCEPTRON]
        if not lp_streaming_supported:
            use_streamed_lp_training = False
            effective_lp_train_batch_size = 0
        elif lp_train_batch_size is None:
            use_streamed_lp_training = dimension > lp_cache_disable_dimension_threshold
            effective_lp_train_batch_size = DOWNSTREAM_LP_TRAIN_BATCH_SIZE_DEFAULT if use_streamed_lp_training else 0
        elif lp_train_batch_size == 0:
            use_streamed_lp_training = False
            effective_lp_train_batch_size = 0
        else:
            use_streamed_lp_training = True
            effective_lp_train_batch_size = lp_train_batch_size

        if use_streamed_lp_training:
            X_test, y_test = prepare_link_prediction_eval_data(
                downstream_df=downstream_df,
                embedding=emb,
                split_category="test",
                feature_op=lp_edge_feature_op,
            )
            X_train = y_train = None
        else:
            curr_train_sample_ratio = lp_logreg_train_pos_sample_ratio if classifier == LOGISTIC_REGRESSION else 1.0
            X_train, y_train, X_test, y_test = train_utils.prepare_link_prediction_data(
                downstream_df=downstream_df,
                edge_list=edge_list,
                embedding=emb,
                return_test_data=True,
                seed=ns_seed,
                feature_op=lp_edge_feature_op,
                train_pos_sample_ratio=curr_train_sample_ratio,
            )
    else:
        X_train, y_train, X_test, y_test = train_utils.prepare_node_classification_data(
            downstream_df=downstream_df,
            embedding=emb,
            return_test_data=True,
        )
        if _should_standard_l2_scale_dgi_mlp_node_classification(algorithm, classifier, task):
            X_train, X_test = _standard_l2_scale_train_test_data(X_train, X_test)
        use_streamed_lp_training = False
        effective_lp_train_batch_size = 0

    if classifier == MULTILAYER_PERCEPTRON:
        input_dim = emb.shape[1] if lp_edge_feature_op == "hadamard" else emb.shape[1] * 2
        if X_train is not None:
            input_dim = int(X_train.shape[1])
        clf_params_local["hidden_layer_sizes"] = MLP_LAYER_DICT[int(input_dim)]

    clf_is_fitted = False
    if task == LINK_PREDICTION and classifier == LOGISTIC_REGRESSION and use_streamed_lp_training:
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
    elif classifier == MULTILAYER_PERCEPTRON and task == LINK_PREDICTION:
        from models.classifiers.mlp import TorchMLPClassifier

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
            input_dim = emb.shape[1] if lp_edge_feature_op == "hadamard" else emb.shape[1] * 2
            clf.fit_streaming(batch_iterator_fn=batch_iterator_fn, input_dim=int(input_dim))
            clf_is_fitted = True
    else:
        np.random.seed(seed)
        clf = DOWNSTREAM_CLASSIFIER_DICT[classifier][DOWNSTREAM_CLASSIFIER_DICT_CLF_KEY](**clf_params_local)

    if not clf_is_fitted:
        clf.fit(X_train, y_train)

    y_pred = _predict_in_batches(clf, X_test, batch_size=prediction_batch_size)
    scores = {measure: float(DOWNSTREAM_MEASURE_DICT[measure](y_test, y_pred)) for measure in measures}

    del emb, X_test
    if X_train is not None:
        del X_train
    if y_train is not None:
        del y_train
    gc.collect()

    return scores


def _run_single_downstream(
    algorithm: EMBEDDING_ALGORITHM,
    dataset_params: dict[str, Any],
    dimension: int,
    embedding_path: Path,
    seed: int,
    embedding_index: int,
    classifier: DOWNSTREAM_CLASSIFIER,
    measures: list[DOWNSTREAM_PERFORMANCE_MEASURE],
    run_dir: Path,
    logs_dir: Path,
    sample_interval: float,
    prediction_batch_size: int,
    lp_train_batch_size: int | None,
    lp_cache_disable_dimension_threshold: int,
    lp_edge_feature_op: LP_EDGE_FEATURE_OP,
    lp_logreg_train_pos_sample_ratio: float,
    lp_resample_negative_per_epoch: bool,
) -> dict[str, Any]:
    log_path = logs_dir / EMBEDDING_COSTS_DOWNSTREAM_LOG_FILE_TEMPLATE.format(
        algorithm=algorithm,
        dataset=dataset_params[CONFIG_DATASET_NAME_KEY],
        dimension=dimension,
        seed=seed,
        classifier=classifier,
    )
    row = {
        "run_id": run_dir.name,
        "measurement_scope": EMBEDDING_COSTS_DOWNSTREAM_MEASUREMENT_SCOPE,
        "dataset": dataset_params[CONFIG_DATASET_NAME_KEY],
        "algorithm": algorithm,
        "dimension": dimension,
        "embedding_index": embedding_index,
        "seed": seed,
        "classifier": classifier,
        "status": "success",
        "elapsed_seconds": None,
        "start_rss_bytes": None,
        "peak_rss_bytes": None,
        "peak_rss_delta_bytes": None,
        "peak_cuda_allocated_bytes": None,
        "embedding_path": str(embedding_path),
        "log_path": str(log_path),
        "error": None,
    }
    for measure in measures:
        row[measure] = None

    try:
        classifier_params = _load_best_downstream_classifier_params(
            dataset_params=dataset_params,
            algorithm=algorithm,
            dimension=dimension,
            classifier=classifier,
        )
        measurements = _measure_callable(
            lambda: _run_downstream_evaluation(
                dataset_params=dataset_params,
                algorithm=algorithm,
                dimension=dimension,
                embedding_path=embedding_path,
                classifier=classifier,
                classifier_params=classifier_params,
                measures=measures,
                seed=EXPERIMENTS_DEFAULT_SEED,
                ns_seed=EXPERIMENTS_DEFAULT_SEED,
                prediction_batch_size=prediction_batch_size,
                lp_train_batch_size=lp_train_batch_size,
                lp_cache_disable_dimension_threshold=lp_cache_disable_dimension_threshold,
                lp_edge_feature_op=lp_edge_feature_op,
                lp_logreg_train_pos_sample_ratio=lp_logreg_train_pos_sample_ratio,
                lp_resample_negative_per_epoch=lp_resample_negative_per_epoch,
            ),
            log_path=log_path,
            sample_interval=sample_interval,
            measure_cuda=classifier == MULTILAYER_PERCEPTRON,
        )
        scores = measurements.pop("validation_score")
        row.update(measurements)
        row.update(scores)
    except Exception as exc:
        row["status"] = "failed"
        row["error"] = repr(exc)

    return row


def main() -> None:
    args = parse_args()
    if args.num_embeddings < 1:
        raise ValueError("--num_embeddings must be at least 1.")
    if args.sample_interval <= 0:
        raise ValueError("--sample_interval must be greater than 0.")
    if args.prediction_batch_size <= 0:
        raise ValueError("--prediction_batch_size must be greater than 0.")
    if args.lp_train_batch_size is not None and args.lp_train_batch_size < 0:
        raise ValueError("--lp_train_batch_size must be non-negative when specified.")
    if args.lp_cache_disable_dimension_threshold < 0:
        raise ValueError("--lp_cache_disable_dimension_threshold must be non-negative.")
    if not (0 < args.lp_logreg_train_pos_sample_ratio <= 1.0):
        raise ValueError("--lp_logreg_train_pos_sample_ratio must be in (0, 1].")

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(EMBEDDING_COSTS_CASE_STUDY_OUTPUT_DIR) / run_id
    reports_dir = run_dir / REPORTS_DIR_NAME
    logs_dir = run_dir / LOGS_DIR_NAME
    run_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    embedding_rows: list[dict[str, Any]] = []
    downstream_rows: list[dict[str, Any]] = []
    run_metadata = {
        "run_id": run_id,
        "measurement_scope": EMBEDDING_COSTS_MEASUREMENT_SCOPE,
        "downstream_measurement_scope": EMBEDDING_COSTS_DOWNSTREAM_MEASUREMENT_SCOPE,
        "algorithms": args.algorithms,
        "datasets": args.datasets,
        "dimensions": args.dimensions,
        "num_embeddings": args.num_embeddings,
        "seed_start": args.seed_start,
        "n_jobs": args.n_jobs,
        "classifiers": args.classifiers,
        "downstream_performance_measures": args.downstream_performance_measures,
        "skip_downstream": args.skip_downstream,
        "prediction_batch_size": args.prediction_batch_size,
        "lp_train_batch_size": args.lp_train_batch_size,
        "lp_cache_disable_dimension_threshold": args.lp_cache_disable_dimension_threshold,
        "lp_edge_feature_op": args.lp_edge_feature_op,
        "lp_logreg_train_pos_sample_ratio": args.lp_logreg_train_pos_sample_ratio,
        "lp_resample_negative_per_epoch": args.lp_resample_negative_per_epoch,
        "sample_interval": args.sample_interval,
        "output_dir": str(run_dir),
    }
    with (reports_dir / RUN_METADATA_FILE_NAME).open("w", encoding="utf-8") as f:
        json.dump(run_metadata, f, indent=4)

    for dataset in args.datasets:
        from tools import data_utils

        dataset_params = {
            CONFIG_DATASET_NAME_KEY: dataset,
            CONFIG_DATA_SUBSAMPLING_INDICATOR_KEY: False,
        }
        data, edge_list_path = data_utils.load_dataset(dataset_params.copy())

        for algorithm in args.algorithms:
            from tools import train_utils

            base_config = train_utils.load_default_config(algorithm)
            dataset_name = dataset_params[CONFIG_DATASET_NAME_KEY]
            for key, value in DATASET_SPECIFIC_PARAM_DICT[algorithm][dataset_name].items():
                base_config[key] = value

            best_params_by_dim = train_utils.get_best_parameter_dict(
                embedding_method=algorithm,
                dataset_params=dataset_params,
                dimensions=args.dimensions,
            )
            configs_by_dim = {}
            for config_dimension in args.dimensions:
                config = copy.deepcopy(base_config)
                config[CONFIG_DIMENSION_KEY] = config_dimension
                config[CONFIG_ITERATIONS_KEY] = 1
                for key, value in best_params_by_dim[config_dimension].items():
                    config[key] = value
                configs_by_dim[config_dimension] = config

            for dimension in args.dimensions:
                save_dir = (
                    run_dir
                    / EMBEDDINGS_DIR_NAME
                    / algorithm
                    / dataset_params[CONFIG_DATASET_NAME_KEY]
                    / BUILD_DATASET_DIR_NAME(dataset_params)
                    / DIMENSION_SUBDIR_NAME(dimension)
                )
                save_dir.mkdir(parents=True, exist_ok=True)
                for embedding_index in range(args.num_embeddings):
                    seed = args.seed_start + embedding_index
                    print(
                        f"Measure {algorithm} on {dataset}, dim={dimension}, "
                        f"embedding {embedding_index + 1}/{args.num_embeddings}, seed={seed}"
                    )
                    row = _run_single_embedding(
                        algorithm=algorithm,
                        dataset_params=dataset_params,
                        data=data,
                        edge_list_path=edge_list_path,
                        config=configs_by_dim[dimension],
                        seed=seed,
                        embedding_index=embedding_index,
                        save_dir=save_dir,
                        run_dir=run_dir,
                        logs_dir=logs_dir,
                        n_jobs=args.n_jobs,
                        sample_interval=args.sample_interval,
                    )
                    embedding_rows.append(row)

                    if not args.skip_downstream and row["status"] == "success" and Path(row["embedding_path"]).is_file():
                        for classifier in args.classifiers:
                            print(
                                f"Measure downstream {classifier} on {algorithm}/{dataset}, "
                                f"dim={dimension}, seed={seed}"
                            )
                            downstream_row = _run_single_downstream(
                                algorithm=algorithm,
                                dataset_params=dataset_params,
                                dimension=dimension,
                                embedding_path=Path(row["embedding_path"]),
                                seed=seed,
                                embedding_index=embedding_index,
                                classifier=classifier,
                                measures=args.downstream_performance_measures,
                                run_dir=run_dir,
                                logs_dir=logs_dir,
                                sample_interval=args.sample_interval,
                                prediction_batch_size=args.prediction_batch_size,
                                lp_train_batch_size=args.lp_train_batch_size,
                                lp_cache_disable_dimension_threshold=args.lp_cache_disable_dimension_threshold,
                                lp_edge_feature_op=args.lp_edge_feature_op,
                                lp_logreg_train_pos_sample_ratio=args.lp_logreg_train_pos_sample_ratio,
                                lp_resample_negative_per_epoch=args.lp_resample_negative_per_epoch,
                            )
                            downstream_rows.append(downstream_row)

                    reports_dir.mkdir(parents=True, exist_ok=True)
                    _write_report_set(
                        rows=embedding_rows,
                        csv_path=reports_dir / EMBEDDING_COSTS_FILE_NAME,
                        summary_csv_path=reports_dir / EMBEDDING_COSTS_SUMMARY_FILE_NAME,
                        summary_builder=_build_embedding_summary_rows,
                    )
                    _write_report_set(
                        rows=downstream_rows,
                        csv_path=reports_dir / DOWNSTREAM_COSTS_FILE_NAME,
                        summary_csv_path=reports_dir / DOWNSTREAM_COSTS_SUMMARY_FILE_NAME,
                        summary_builder=_build_downstream_summary_rows,
                    )
                    gc.collect()

    print(f"Reports written to {reports_dir}")


if __name__ == "__main__":
    main()
