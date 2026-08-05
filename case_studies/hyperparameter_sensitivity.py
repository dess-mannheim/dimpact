from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

from sklearn.model_selection import ParameterGrid

from paths_globals import *


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Case-study workflow for sensitivity to tuning embeddings at a fixed dimension."
    )
    parser.add_argument("-a", "--algorithm", choices=EMBEDDING_ALGORITHM_LIST, default=GRAPHSAGE)
    parser.add_argument("-d", "--dataset", choices=EMPIRICAL_DATASET_LIST, default="Cora")
    parser.add_argument("-dim", "--dimensions", nargs="+", type=int, default=EXPERIMENTS_DIMENSIONS_LIST)
    parser.add_argument(
        "--anchor_dimension",
        type=int,
        default=TUNING_DEFAULT_DIMENSION,
        help="Dimension whose existing tuning result defines the original fixed configuration.",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=["retune", "stability"],
        default=["retune", "stability"],
        help="Workflow stages to run.",
    )
    parser.add_argument(
        "--num_tuning_seeds",
        type=int,
        default=5,
        help="Number of seeds per hyperparameter setup in Stage 1.",
    )
    parser.add_argument(
        "--num_stage2_embeddings",
        type=int,
        default=10,
        help="Number of embeddings per selected dimension/configuration in Stage 2.",
    )
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument(
        "-c",
        "--classifiers",
        nargs="+",
        choices=DOWNSTREAM_CLASSIFIERS,
        default=[LOGISTIC_REGRESSION],
        help="Downstream classifiers used in Stage 2.",
    )
    parser.add_argument(
        "--stage2_tune_logreg",
        action="store_true",
        help=(
            "Tune LogisticRegression hyperparameters within Stage 2 for each selected "
            "dimension/configuration before final downstream evaluation."
        ),
    )
    parser.add_argument(
        "--num_stage2_tuning_embeddings",
        type=int,
        default=5,
        help="Number of successful Stage 2 embeddings used to tune LogisticRegression.",
    )
    parser.add_argument(
        "--downstream_performance_measures",
        nargs="+",
        choices=DOWNSTREAM_PERFORMANCE_MEASURES,
        default=[ACCURACY_SCORE],
    )
    parser.add_argument(
        "--representational_measures",
        nargs="+",
        default=["JaccardSimilarity", "AlignedCosineSimilarity"],
        help="Representational stability measures for Stage 2.",
    )
    parser.add_argument(
        "--functional_measures",
        nargs="+",
        default=["Disagreement", "JSD"],
        help="Functional stability measures for Stage 2.",
    )
    parser.add_argument(
        "--material_improvement",
        type=float,
        default=0.005,
        help="Minimum validation-score improvement required to trigger Stage 2 if parameters did not change.",
    )
    parser.add_argument(
        "--force_stage2_dimensions",
        nargs="*",
        type=int,
        default=[],
        help="Dimensions to include in Stage 2 regardless of the automatic selection rule.",
    )
    parser.add_argument(
        "--skip_stage2_auto_selection",
        action="store_true",
        help="Run Stage 2 only for --force_stage2_dimensions.",
    )
    parser.add_argument(
        "--prediction_batch_size",
        type=int,
        default=DOWNSTREAM_PREDICTION_BATCH_SIZE_DEFAULT,
    )
    parser.add_argument("--lp_train_batch_size", type=int, default=None)
    parser.add_argument(
        "--lp_cache_disable_dimension_threshold",
        type=int,
        default=DOWNSTREAM_LP_CACHE_DISABLE_DIMENSION_THRESHOLD_DEFAULT,
    )
    parser.add_argument("--lp_edge_feature_op", choices=["hadamard", "concat"], default="hadamard")
    parser.add_argument("--lp_logreg_train_pos_sample_ratio", type=float, default=1.0)
    parser.add_argument("--lp_resample_negative_per_epoch", action="store_true")
    parser.add_argument(
        "--collect_existing_stage2_results",
        action="store_true",
        help=(
            "Rebuild/merge Stage 2 reports from existing embeddings, predictions, "
            "scores, and downstream-tuning artifacts without training or evaluating."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--sample_interval", type=float, default=0.2)
    return parser.parse_args()


def _dataset_params(dataset: DATASET) -> dict[str, Any]:
    return {
        CONFIG_DATASET_NAME_KEY: dataset,
        CONFIG_DATA_SUBSAMPLING_INDICATOR_KEY: False,
    }


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_json_if_exists(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    return _read_json(path)


def _read_csv_if_exists(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8") as f:
            f.write("")
        return
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _merge_rows_by_key(
    existing_rows: list[dict[str, Any]],
    new_rows: list[dict[str, Any]],
    key_columns: list[str],
) -> list[dict[str, Any]]:
    merged: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in existing_rows:
        merged[tuple(str(row.get(column, "")) for column in key_columns)] = row
    for row in new_rows:
        merged[tuple(str(row.get(column, "")) for column in key_columns)] = row
    return sorted(merged.values(), key=lambda row: tuple(str(row.get(column, "")) for column in key_columns))


def _merge_functional_rows(
    existing_rows: list[dict[str, Any]],
    new_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    def key(row: dict[str, Any]) -> tuple[Any, ...]:
        return (
            row.get("dataset"),
            row.get("algorithm"),
            row.get("dimension"),
            row.get("config_label"),
            row.get("classifier"),
            row.get("measure"),
            row.get("seed_left"),
            row.get("seed_right"),
            row.get("seeds"),
        )

    merged = {tuple(str(value) for value in key(row)): row for row in existing_rows}
    for row in new_rows:
        merged[tuple(str(value) for value in key(row))] = row
    return sorted(merged.values(), key=lambda row: tuple(str(value) for value in key(row)))


def _existing_row_keys(rows: list[dict[str, Any]], key_columns: list[str]) -> set[tuple[Any, ...]]:
    return {tuple(str(row.get(column, "")) for column in key_columns) for row in rows}


def _functional_row_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("dataset"),
        row.get("algorithm"),
        row.get("dimension"),
        row.get("config_label"),
        row.get("classifier"),
        row.get("measure"),
        row.get("seed_left"),
        row.get("seed_right"),
        row.get("seeds"),
    )


def _canonical_params(params: dict[str, Any]) -> str:
    return json.dumps(params, sort_keys=True)


def _params_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return _canonical_params(left) == _canonical_params(right)


def _load_base_config(algorithm: EMBEDDING_ALGORITHM, dataset: DATASET) -> dict[str, Any]:
    from tools import train_utils

    config = train_utils.load_default_config(algorithm)
    for key, value in DATASET_SPECIFIC_PARAM_DICT[algorithm][dataset].items():
        config[key] = value
    return config


def _stage2_embedding_dir(
    output_dir: Path,
    dimension: int,
    config_label: str,
) -> Path:
    path = (
        output_dir
        / EMBEDDINGS_DIR_NAME
        / HYPERPARAMETER_SENSITIVITY_STAGE2_STABILITY_DIR_NAME
        / DIMENSION_SUBDIR_NAME(dimension)
        / config_label
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def _row_mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _row_std(values: list[float]) -> float | None:
    return float(np.std(values, ddof=1)) if len(values) > 1 else None


def _run_single_embedding_regular(
    algorithm: EMBEDDING_ALGORITHM,
    dataset_params: dict[str, Any],
    config: dict[str, Any],
    seed: int,
    embedding_index: int,
    save_dir: Path,
    run_dir: Path,
    logs_dir: Path,
    n_jobs: int,
) -> dict[str, Any]:
    dimension = config[CONFIG_DIMENSION_KEY]
    log_path = logs_dir / EMBEDDING_RUN_LOG_FILE_TEMPLATE.format(
        algorithm=algorithm,
        dataset=dataset_params[CONFIG_DATASET_NAME_KEY],
        dimension=dimension,
        seed=seed,
    )
    embedding_path = save_dir / EMBEDDING_FILE_NAME(model_seed=seed)

    row = {
        "run_id": run_dir.name,
        "measurement_scope": HYPERPARAMETER_SENSITIVITY_EMBEDDING_TRAINING_SCOPE,
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
        from train import train_embeddings

        train_config = copy.deepcopy(config)
        train_config[CONFIG_TRAINING_SEEDS_KEY] = [seed]
        train_config[CONFIG_ITERATIONS_KEY] = 1
        start = time.perf_counter()
        scores = train_embeddings(
            embedding_name=algorithm,
            dataset_params=dataset_params,
            embedding_config=train_config,
            overwrite=True,
            n_jobs=n_jobs,
            seeds=[seed],
            save_dir=save_dir,
        )
        score = scores.get(seed)
        row["elapsed_seconds"] = time.perf_counter() - start
        row["validation_score"] = score
    except Exception as exc:
        row["status"] = "failed"
        row["error"] = repr(exc)

    return row


def run_stage1_retuning(
    args: argparse.Namespace,
    output_dir: Path,
    logs_dir: Path,
    reports_dir: Path,
) -> list[dict[str, Any]]:
    from tools import train_utils

    dataset_params = _dataset_params(args.dataset)
    base_config = _load_base_config(args.algorithm, args.dataset)
    anchor_params = train_utils.get_best_parameter_dict(
        embedding_method=args.algorithm,
        dataset_params=dataset_params,
        dimensions=[args.anchor_dimension],
    )[args.anchor_dimension]
    param_grid = list(ParameterGrid(TUNING_PARAM_GRID_DICT[args.algorithm]))

    all_summary: dict[str, Any] = {}
    tuning_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []

    for dimension in args.dimensions:
        dimension_summary_path = (
            output_dir
            / HYPERPARAMETER_SENSITIVITY_STAGE1_RETUNING_DIR_NAME
            / SUMMARIES_DIR_NAME
            / DIMENSION_SUBDIR_NAME(dimension)
            / TUNING_SUMMARY_FILE_NAME
        )
        if dimension == args.anchor_dimension:
            tuning_dir = Path(
                CREATE_MODELS_PATH(
                    dataset_params=dataset_params,
                    embedding_name=args.algorithm,
                    embedding_dim=dimension,
                    b_tune=True,
                )
            )
            tuning_summary_path = tuning_dir / TUNING_SUMMARY_FILE_NAME
            if not tuning_summary_path.is_file():
                raise FileNotFoundError(
                    f"Main tuning summary is missing for anchor dimension {dimension}: {tuning_summary_path}"
                )
            tuning_summary = _read_json(tuning_summary_path)
            expected_tune_keys = {str(tune_id) for tune_id in range(len(param_grid))}
            missing_tune_keys = sorted(expected_tune_keys - set(tuning_summary.keys()), key=int)
            if missing_tune_keys:
                raise FileNotFoundError(
                    f"Main tuning summary for anchor dimension {dimension} is incomplete. "
                    f"Missing tune ids: {missing_tune_keys}"
                )
            _write_json(dimension_summary_path, tuning_summary)
        elif dimension_summary_path.is_file() and not args.overwrite:
            with dimension_summary_path.open("r", encoding="utf-8") as f:
                tuning_summary = json.load(f)
        else:
            tuning_summary = {}

        for tune_id, params in enumerate(param_grid):
            tune_key = str(tune_id)
            if tune_key in tuning_summary and not args.overwrite:
                print(f"Reuse Stage 1 tuning result for dim={dimension}, tune_id={tune_id}")
            else:
                config = copy.deepcopy(base_config)
                config[CONFIG_DIMENSION_KEY] = dimension
                config[CONFIG_ITERATIONS_KEY] = 1
                config.update(params)
                seed_scores: dict[str, float] = {}
                save_dir = (
                    output_dir
                    / EMBEDDINGS_DIR_NAME
                    / HYPERPARAMETER_SENSITIVITY_STAGE1_RETUNING_DIR_NAME
                    / DIMENSION_SUBDIR_NAME(dimension)
                    / TUNE_RUN_SUBDIR_NAME(tune_id)
                )
                save_dir.mkdir(parents=True, exist_ok=True)
                candidate_logs_dir = (
                    logs_dir
                    / HYPERPARAMETER_SENSITIVITY_STAGE1_RETUNING_DIR_NAME
                    / DIMENSION_SUBDIR_NAME(dimension)
                    / TUNE_RUN_SUBDIR_NAME(tune_id)
                )
                candidate_logs_dir.mkdir(parents=True, exist_ok=True)

                for seed_offset in range(args.num_tuning_seeds):
                    seed = args.seed_start + seed_offset
                    print(
                        f"Stage 1 retuning {args.algorithm}/{args.dataset}: "
                        f"dim={dimension}, tune_id={tune_id}, seed={seed}"
                    )
                    row = _run_single_embedding_regular(
                        algorithm=args.algorithm,
                        dataset_params=dataset_params,
                        config=config,
                        seed=seed,
                        embedding_index=seed_offset,
                        save_dir=save_dir,
                        run_dir=output_dir,
                        logs_dir=candidate_logs_dir,
                        n_jobs=args.n_jobs,
                    )
                    if row["status"] != "success":
                        raise RuntimeError(
                            f"Stage 1 embedding failed for dim={dimension}, tune_id={tune_id}, seed={seed}: "
                            f"{row['error']}"
                        )
                    seed_scores[str(seed)] = float(row["validation_score"])

                tuning_summary[tune_key] = {
                    TUNING_SUMMARY_PARAMS_KEY: dict(params),
                    TUNING_SUMMARY_RESULTS_KEY: seed_scores,
                    TUNING_SUMMARY_SCORE_KEY: float(np.mean(list(seed_scores.values()))),
                }
                _write_json(dimension_summary_path, tuning_summary)

            summary = tuning_summary[tune_key]
            params = summary[TUNING_SUMMARY_PARAMS_KEY]
            results = summary[TUNING_SUMMARY_RESULTS_KEY]
            tuning_rows.append(
                {
                    "dataset": args.dataset,
                    "algorithm": args.algorithm,
                    "dimension": dimension,
                    "tune_id": tune_id,
                    "params": _canonical_params(params),
                    "mean_validation_score": summary[TUNING_SUMMARY_SCORE_KEY],
                    "std_validation_score": _row_std([float(v) for v in results.values()]),
                    "num_runs": len(results),
                    "is_anchor_configuration": _params_equal(params, anchor_params),
                }
            )

        all_summary[str(dimension)] = tuning_summary
        best_tune_id, best_summary = max(
            tuning_summary.items(),
            key=lambda item: item[1][TUNING_SUMMARY_SCORE_KEY],
        )
        anchor_entries = [
            summary
            for summary in tuning_summary.values()
            if _params_equal(summary[TUNING_SUMMARY_PARAMS_KEY], anchor_params)
        ]
        anchor_score = anchor_entries[0][TUNING_SUMMARY_SCORE_KEY] if anchor_entries else None
        best_score = best_summary[TUNING_SUMMARY_SCORE_KEY]
        comparison_rows.append(
            {
                "dataset": args.dataset,
                "algorithm": args.algorithm,
                "dimension": dimension,
                "anchor_dimension": args.anchor_dimension,
                "anchor_params": _canonical_params(anchor_params),
                "dimension_specific_tune_id": int(best_tune_id),
                "dimension_specific_params": _canonical_params(best_summary[TUNING_SUMMARY_PARAMS_KEY]),
                "anchor_validation_score_at_dimension": anchor_score,
                "dimension_specific_validation_score": best_score,
                "validation_score_improvement": None if anchor_score is None else best_score - anchor_score,
                "params_changed": not _params_equal(anchor_params, best_summary[TUNING_SUMMARY_PARAMS_KEY]),
                "material_improvement": False
                if anchor_score is None
                else (best_score - anchor_score) >= args.material_improvement,
            }
        )

    summary_path = reports_dir / HYPERPARAMETER_SENSITIVITY_STAGE1_TUNING_SUMMARY_BY_DIMENSION_FILE_NAME
    tuning_path = reports_dir / HYPERPARAMETER_SENSITIVITY_STAGE1_TUNING_RESULTS_FILE_NAME
    comparison_csv_path = reports_dir / HYPERPARAMETER_SENSITIVITY_STAGE1_COMPARISON_FILE_NAME

    merged_summary = {
        **_read_json_if_exists(summary_path, {}),
        **{str(dimension): summary for dimension, summary in all_summary.items()},
    }
    merged_tuning_rows = _merge_rows_by_key(
        _read_csv_if_exists(tuning_path),
        tuning_rows,
        ["dataset", "algorithm", "dimension", "tune_id"],
    )
    existing_comparison_rows = (
        _load_stage1_comparison(reports_dir)
        if comparison_csv_path.is_file()
        or (reports_dir / HYPERPARAMETER_SENSITIVITY_STAGE1_COMPARISON_LEGACY_JSON_FILE_NAME).is_file()
        else []
    )
    merged_comparison_rows = _merge_rows_by_key(
        existing_comparison_rows,
        comparison_rows,
        ["dataset", "algorithm", "dimension", "anchor_dimension"],
    )

    _write_json(summary_path, merged_summary)
    _write_csv(tuning_path, merged_tuning_rows)
    _write_csv(comparison_csv_path, merged_comparison_rows)
    return merged_comparison_rows


def _load_stage1_comparison(reports_dir: Path) -> list[dict[str, Any]]:
    csv_path = reports_dir / HYPERPARAMETER_SENSITIVITY_STAGE1_COMPARISON_FILE_NAME
    json_path = reports_dir / HYPERPARAMETER_SENSITIVITY_STAGE1_COMPARISON_LEGACY_JSON_FILE_NAME
    if csv_path.is_file():
        rows = _read_csv_if_exists(csv_path)
        for row in rows:
            for key in ["params_changed", "material_improvement"]:
                value = row.get(key)
                if isinstance(value, str):
                    row[key] = value.strip().lower() in {"true", "1", "yes"}
            for key in [
                "dimension",
                "anchor_dimension",
                "dimension_specific_tune_id",
            ]:
                if row.get(key) not in [None, ""]:
                    row[key] = int(row[key])
            for key in [
                "anchor_validation_score_at_dimension",
                "dimension_specific_validation_score",
                "validation_score_improvement",
            ]:
                if row.get(key) not in [None, ""]:
                    row[key] = float(row[key])
        return rows
    if json_path.is_file():
        return _read_json(json_path)
    raise FileNotFoundError(f"Stage 1 comparison report not found: {csv_path}")


def _keep_dimension_specific_stage2_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("config_label") == HYPERPARAMETER_SENSITIVITY_DIMENSION_SPECIFIC_LABEL]


def _load_best_downstream_classifier_params(
    dataset_params: dict[str, Any],
    algorithm: EMBEDDING_ALGORITHM,
    dimension: int,
    classifier: DOWNSTREAM_CLASSIFIER,
) -> dict[str, Any]:
    from paths_globals import CREATE_DOWNSTREAM_RESULTS_PATH

    results_dir = CREATE_DOWNSTREAM_RESULTS_PATH(
        dataset_params=dataset_params,
        embedding_name=algorithm,
        clf_name=classifier,
        embedding_dim=dimension,
    )
    tuning_path = osp.join(results_dir, TUNING_SUMMARY_FILE_NAME)
    if not osp.isfile(tuning_path):
        raise FileNotFoundError(f"Missing downstream classifier tuning results: {tuning_path}")
    with open(tuning_path, "r", encoding="utf-8") as f:
        tuning_summary = json.load(f)
    tune_ids = list(tuning_summary.keys())
    scores = [tuning_summary[tune_id][TUNING_SUMMARY_SCORE_KEY] for tune_id in tune_ids]
    best_id = tune_ids[scores.index(max(scores))]
    return {
        **DOWNSTREAM_CLASSIFIER_DICT[classifier][DOWNSTREAM_CLASSIFIER_DICT_BASE_PARAMS_KEY],
        **tuning_summary[best_id][TUNING_SUMMARY_PARAMS_KEY],
    }


def _stage2_downstream_tuning_path(
    output_dir: Path,
    dimension: int,
    config_label: str,
    classifier: DOWNSTREAM_CLASSIFIER,
) -> Path:
    return (
        output_dir
        / DOWNSTREAM_TUNING_DIR_NAME
        / HYPERPARAMETER_SENSITIVITY_STAGE2_STABILITY_DIR_NAME
        / DIMENSION_SUBDIR_NAME(dimension)
        / config_label
        / classifier
        / TUNING_SUMMARY_FILE_NAME
    )


def _tune_stage2_logreg_params(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    dataset_params: dict[str, Any],
    dimension: int,
    config_label: str,
    embedding_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    classifier = LOGISTIC_REGRESSION
    tuning_summary_path = _stage2_downstream_tuning_path(output_dir, dimension, config_label, classifier)
    best_params_path = _stage2_downstream_tuning_path(
        output_dir,
        dimension,
        config_label,
        classifier,
    ).with_name(HYPERPARAMETER_SENSITIVITY_BEST_PARAMS_FILE_NAME)
    param_grid = list(
        ParameterGrid(DOWNSTREAM_CLASSIFIER_DICT[classifier][DOWNSTREAM_CLASSIFIER_DICT_TUNING_PARAMS_KEY])
    )
    tuning_summary: dict[str, Any]
    if tuning_summary_path.is_file() and not args.overwrite:
        tuning_summary = _read_json(tuning_summary_path)
    else:
        tuning_summary = {}

    successful_embeddings = [row for row in embedding_rows if row.get("status") == "success"]
    tuning_embeddings = successful_embeddings[: max(1, int(args.num_stage2_tuning_embeddings))]
    if len(tuning_embeddings) < 1:
        raise ValueError(
            f"No successful Stage 2 embeddings available for downstream tuning at "
            f"dim={dimension}, config={config_label}."
        )

    tuning_rows: list[dict[str, Any]] = []
    for tune_id, params in enumerate(param_grid):
        tune_key = str(tune_id)
        if tune_key in tuning_summary and not args.overwrite:
            summary = tuning_summary[tune_key]
            print(
                f"Reuse Stage 2 LogisticRegression tuning result for "
                f"dim={dimension}, config={config_label}, tune_id={tune_id}"
            )
        else:
            curr_params = {
                **DOWNSTREAM_CLASSIFIER_DICT[classifier][DOWNSTREAM_CLASSIFIER_DICT_BASE_PARAMS_KEY],
                **params,
            }
            scores_by_embedding: dict[str, dict[str, float]] = {}
            for emb_row in tuning_embeddings:
                seed = int(emb_row["seed"])
                print(
                    f"Stage 2 LogisticRegression tuning {args.algorithm}/{args.dataset}: "
                    f"dim={dimension}, config={config_label}, tune_id={tune_id}, seed={seed}"
                )
                result = _evaluate_downstream_for_embedding(
                    dataset_params=dataset_params,
                    algorithm=args.algorithm,
                    dimension=dimension,
                    embedding_path=Path(emb_row["embedding_path"]),
                    classifier=classifier,
                    classifier_params=curr_params,
                    measures=[ACCURACY_SCORE, MICRO_F1_SCORE, MACRO_F1_SCORE],
                    seed=EXPERIMENTS_DEFAULT_SEED,
                    prediction_batch_size=args.prediction_batch_size,
                    lp_train_batch_size=args.lp_train_batch_size,
                    lp_cache_disable_dimension_threshold=args.lp_cache_disable_dimension_threshold,
                    lp_edge_feature_op=args.lp_edge_feature_op,
                    lp_logreg_train_pos_sample_ratio=args.lp_logreg_train_pos_sample_ratio,
                    lp_resample_negative_per_epoch=args.lp_resample_negative_per_epoch,
                    use_validation_split=True,
                )
                scores_by_embedding[str(seed)] = {
                    key: float(value) for key, value in result["scores"].items()
                }

            accuracy_values = [
                float(scores[ACCURACY_SCORE])
                for scores in scores_by_embedding.values()
                if ACCURACY_SCORE in scores
            ]
            tuning_summary[tune_key] = {
                TUNING_SUMMARY_PARAMS_KEY: dict(params),
                TUNING_SUMMARY_RESULTS_KEY: scores_by_embedding,
                TUNING_SUMMARY_SCORE_KEY: float(np.mean(accuracy_values)),
            }
            _write_json(tuning_summary_path, tuning_summary)
            summary = tuning_summary[tune_key]

        tune_scores = summary.get(TUNING_SUMMARY_RESULTS_KEY, {})
        accuracy_values = [
            float(scores[ACCURACY_SCORE])
            for scores in tune_scores.values()
            if isinstance(scores, dict) and ACCURACY_SCORE in scores
        ]
        tuning_rows.append(
            {
                "dataset": args.dataset,
                "algorithm": args.algorithm,
                "dimension": dimension,
                "config_label": config_label,
                "classifier": classifier,
                "tune_id": tune_id,
                "params": _canonical_params(summary[TUNING_SUMMARY_PARAMS_KEY]),
                "mean_validation_score": summary[TUNING_SUMMARY_SCORE_KEY],
                "std_validation_score": _row_std(accuracy_values),
                "num_embeddings": len(accuracy_values),
            }
        )

    if not tuning_summary:
        raise ValueError("Cannot select classifier parameters from an empty tuning summary.")
    tune_ids = list(tuning_summary.keys())
    scores = [float(tuning_summary[tune_id][TUNING_SUMMARY_SCORE_KEY]) for tune_id in tune_ids]
    best_id = tune_ids[scores.index(max(scores))]
    best_params = {
        **DOWNSTREAM_CLASSIFIER_DICT[classifier][DOWNSTREAM_CLASSIFIER_DICT_BASE_PARAMS_KEY],
        **tuning_summary[best_id][TUNING_SUMMARY_PARAMS_KEY],
    }
    _write_json(best_params_path, best_params)
    return best_params, tuning_rows


def _evaluate_downstream_for_embedding(
    dataset_params: dict[str, Any],
    algorithm: EMBEDDING_ALGORITHM,
    dimension: int,
    embedding_path: Path,
    classifier: DOWNSTREAM_CLASSIFIER,
    classifier_params: dict[str, Any],
    measures: list[DOWNSTREAM_PERFORMANCE_MEASURE],
    seed: int,
    prediction_batch_size: int,
    lp_train_batch_size: int | None,
    lp_cache_disable_dimension_threshold: int,
    lp_edge_feature_op: LP_EDGE_FEATURE_OP,
    lp_logreg_train_pos_sample_ratio: float,
    lp_resample_negative_per_epoch: bool,
    use_validation_split: bool = False,
) -> dict[str, Any]:
    import pandas as pd
    from models.classifiers.mlp import TorchMLPClassifier
    from run_downstream_tasks import (
        _fit_lp_logreg_in_batches,
        _predict_in_batches,
        _predict_proba_in_batches,
        _should_standard_l2_scale_dgi_mlp_node_classification,
        _standard_l2_scale_train_test_data,
    )
    from tools import train_utils
    from tools.train_utils import iter_link_prediction_train_batches, prepare_link_prediction_eval_data
    from paths_globals import DATA_EDGE_LIST_DEFAULT_FILE_NAME, DATASET_TASK_DICT

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
                split_category=DOWNSTREAM_TASK_DATA_VAL_CATEGORY
                if use_validation_split
                else DOWNSTREAM_TASK_DATA_TEST_CATEGORY,
                feature_op=lp_edge_feature_op,
            )
            X_train = y_train = None
        else:
            curr_train_sample_ratio = lp_logreg_train_pos_sample_ratio if classifier == LOGISTIC_REGRESSION else 1.0
            X_train, y_train, X_test, y_test = train_utils.prepare_link_prediction_data(
                downstream_df=downstream_df,
                edge_list=edge_list,
                embedding=emb,
                return_val_data=use_validation_split,
                return_test_data=not use_validation_split,
                seed=EXPERIMENTS_DEFAULT_SEED,
                feature_op=lp_edge_feature_op,
                train_pos_sample_ratio=curr_train_sample_ratio,
            )
    else:
        X_train, y_train, X_test, y_test = train_utils.prepare_node_classification_data(
            downstream_df=downstream_df,
            embedding=emb,
            return_val_data=use_validation_split,
            return_test_data=not use_validation_split,
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
        clf_params_local["random_state"] = seed
        clf = TorchMLPClassifier(**clf_params_local)
        if use_streamed_lp_training:
            batch_iterator_fn = lambda: iter_link_prediction_train_batches(
                downstream_df=downstream_df,
                edge_list=edge_list,
                embedding=emb,
                batch_size=effective_lp_train_batch_size,
                seed=EXPERIMENTS_DEFAULT_SEED,
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
    if hasattr(clf, "predict_proba"):
        outputs = _predict_proba_in_batches(clf, X_test, batch_size=prediction_batch_size)
    elif hasattr(clf, "decision_function"):
        raw = clf.decision_function(X_test)
        outputs = raw.reshape(-1, 1) if raw.ndim == 1 else raw
    else:
        classes = np.unique(y_pred)
        class_index = {label: idx for idx, label in enumerate(classes)}
        outputs = np.zeros((len(y_pred), len(classes)), dtype=np.float32)
        for idx, label in enumerate(y_pred):
            outputs[idx, class_index[label]] = 1.0
    scores = {measure: float(DOWNSTREAM_MEASURE_DICT[measure](y_test, y_pred)) for measure in measures}
    return {
        "scores": scores,
        "outputs": np.asarray(outputs),
        "y_test": np.asarray(y_test),
        "y_pred": np.asarray(y_pred),
    }


def run_stage2_stability(
    args: argparse.Namespace,
    output_dir: Path,
    logs_dir: Path,
    reports_dir: Path,
    comparison_rows: list[dict[str, Any]],
) -> None:
    from stability.measures import (
        ALL_FUNCSIM_MEASURES,
        ALL_MEASURES,
        GROUPWISE_FUNCTIONAL_SIMILARITY_MEASURES,
        PAIRWISE_FUNCTIONAL_SIMILARITY_MEASURES,
        PERFORMANCE_BASED_FUNCTIONAL_SIMILARITY_MEASURES,
    )
    from stability.measures.repsim import ND_SHAPE

    invalid_rep = sorted(set(args.representational_measures) - set(ALL_MEASURES.keys()))
    invalid_func = sorted(set(args.functional_measures) - set(ALL_FUNCSIM_MEASURES.keys()))
    if invalid_rep:
        raise ValueError(f"Unknown representational measures: {invalid_rep}")
    if invalid_func:
        raise ValueError(f"Unknown functional measures: {invalid_func}")

    selected_path = reports_dir / HYPERPARAMETER_SENSITIVITY_STAGE2_SELECTED_DIMENSIONS_FILE_NAME
    selected = {dimension for dimension in args.force_stage2_dimensions if dimension != args.anchor_dimension}
    if not args.skip_stage2_auto_selection:
        for row in comparison_rows:
            if int(row["dimension"]) == args.anchor_dimension:
                continue
            if row["params_changed"] or row["material_improvement"]:
                selected.add(int(row["dimension"]))
    selected_dimensions = [dimension for dimension in args.dimensions if dimension in selected]
    if not selected_dimensions:
        print("Stage 2 has no selected dimensions. Use --force_stage2_dimensions to force it.")
        _write_json(selected_path, _read_json_if_exists(selected_path, []))
        return

    dataset_params = _dataset_params(args.dataset)
    base_config = _load_base_config(args.algorithm, args.dataset)
    dimension_params = {
        int(row["dimension"]): json.loads(row["dimension_specific_params"])
        for row in _load_stage1_comparison(reports_dir)
    }

    existing_selected_dimensions = [int(dimension) for dimension in _read_json_if_exists(selected_path, [])]
    merged_selected_dimensions = sorted(set(existing_selected_dimensions).union(selected_dimensions))
    _write_json(selected_path, merged_selected_dimensions)

    embedding_rows: list[dict[str, Any]] = []
    downstream_rows: list[dict[str, Any]] = []
    tuning_report_path = reports_dir / HYPERPARAMETER_SENSITIVITY_STAGE2_DOWNSTREAM_TUNING_RESULTS_FILE_NAME
    downstream_report_path = reports_dir / HYPERPARAMETER_SENSITIVITY_STAGE2_DOWNSTREAM_PERFORMANCE_FILE_NAME
    embedding_report_path = reports_dir / HYPERPARAMETER_SENSITIVITY_STAGE2_EMBEDDING_RUNS_FILE_NAME
    rep_report_path = reports_dir / HYPERPARAMETER_SENSITIVITY_STAGE2_REPRESENTATIONAL_STABILITY_FILE_NAME
    func_report_path = reports_dir / HYPERPARAMETER_SENSITIVITY_STAGE2_FUNCTIONAL_STABILITY_FILE_NAME
    existing_embedding_rows: list[dict[str, Any]] = _keep_dimension_specific_stage2_rows(
        _read_csv_if_exists(embedding_report_path)
    )
    existing_downstream_rows: list[dict[str, Any]] = _keep_dimension_specific_stage2_rows(
        _read_csv_if_exists(downstream_report_path)
    )
    existing_tuning_rows: list[dict[str, Any]] = _keep_dimension_specific_stage2_rows(
        _read_csv_if_exists(tuning_report_path)
    )
    existing_rep_rows: list[dict[str, Any]] = _keep_dimension_specific_stage2_rows(_read_csv_if_exists(rep_report_path))
    existing_func_rows: list[dict[str, Any]] = _keep_dimension_specific_stage2_rows(_read_csv_if_exists(func_report_path))
    existing_rep_keys = (
        set()
        if args.overwrite
        else _existing_row_keys(
            existing_rep_rows,
            ["dataset", "algorithm", "dimension", "config_label", "measure", "seed_left", "seed_right"],
        )
    )
    existing_func_keys = (
        set()
        if args.overwrite or args.stage2_tune_logreg
        else {tuple(str(value) for value in _functional_row_key(row)) for row in existing_func_rows}
    )
    rep_rows: list[dict[str, Any]] = []
    func_rows: list[dict[str, Any]] = []
    prediction_outputs: dict[tuple[int, str, str, int], np.ndarray] = {}
    prediction_scores: dict[tuple[int, str, str, int], dict[str, float]] = {}
    downstream_tuning_rows: list[dict[str, Any]] = []

    for dimension in selected_dimensions:
        config_label = HYPERPARAMETER_SENSITIVITY_DIMENSION_SPECIFIC_LABEL
        params = dimension_params[dimension]
        config = copy.deepcopy(base_config)
        config[CONFIG_DIMENSION_KEY] = dimension
        config[CONFIG_ITERATIONS_KEY] = 1
        config.update(params)
        save_dir = _stage2_embedding_dir(output_dir, dimension, config_label)
        config_logs_dir = (
            logs_dir
            / HYPERPARAMETER_SENSITIVITY_STAGE2_STABILITY_DIR_NAME
            / DIMENSION_SUBDIR_NAME(dimension)
            / config_label
        )
        config_logs_dir.mkdir(parents=True, exist_ok=True)

        config_embedding_rows: list[dict[str, Any]] = []
        for embedding_index in range(args.num_stage2_embeddings):
            seed = args.seed_start + embedding_index
            embedding_path = save_dir / EMBEDDING_FILE_NAME(model_seed=seed)
            if embedding_path.is_file() and not args.overwrite:
                print(f"Reuse Stage 2 embedding {embedding_path}")
                emb_row = {
                    "dataset": args.dataset,
                    "algorithm": args.algorithm,
                    "dimension": dimension,
                    "config_label": config_label,
                    "embedding_index": embedding_index,
                    "seed": seed,
                    "status": "success",
                    "validation_score": None,
                    "embedding_path": str(embedding_path),
                }
            else:
                print(
                    f"Stage 2 train {args.algorithm}/{args.dataset}: "
                    f"dim={dimension}, config={config_label}, seed={seed}"
                )
                emb_row = _run_single_embedding_regular(
                    algorithm=args.algorithm,
                    dataset_params=dataset_params,
                    config=config,
                    seed=seed,
                    embedding_index=embedding_index,
                    save_dir=save_dir,
                    run_dir=output_dir,
                    logs_dir=config_logs_dir,
                    n_jobs=args.n_jobs,
                )
                emb_row["config_label"] = config_label
            embedding_rows.append(emb_row)
            config_embedding_rows.append(emb_row)

        classifier_params_by_classifier: dict[str, dict[str, Any]] = {}
        for classifier in args.classifiers:
            if args.stage2_tune_logreg and classifier == LOGISTIC_REGRESSION:
                classifier_params, tuning_rows = _tune_stage2_logreg_params(
                    args,
                    output_dir=output_dir,
                    dataset_params=dataset_params,
                    dimension=dimension,
                    config_label=config_label,
                    embedding_rows=config_embedding_rows,
                )
                classifier_params_by_classifier[classifier] = classifier_params
                downstream_tuning_rows.extend(tuning_rows)
            else:
                classifier_params_by_classifier[classifier] = _load_best_downstream_classifier_params(
                    dataset_params=dataset_params,
                    algorithm=args.algorithm,
                    dimension=dimension,
                    classifier=classifier,
                )

        for emb_row in config_embedding_rows:
            if emb_row["status"] != "success":
                continue
            seed = int(emb_row["seed"])
            embedding_index = int(emb_row["embedding_index"])
            for classifier in args.classifiers:
                prediction_dir = (
                    output_dir
                    / PREDICTIONS_DIR_NAME
                    / HYPERPARAMETER_SENSITIVITY_STAGE2_STABILITY_DIR_NAME
                    / DIMENSION_SUBDIR_NAME(dimension)
                    / config_label
                    / classifier
                )
                prediction_dir.mkdir(parents=True, exist_ok=True)
                output_path = prediction_dir / HYPERPARAMETER_SENSITIVITY_OUTPUTS_FILE_TEMPLATE.format(seed=seed)
                pred_path = prediction_dir / HYPERPARAMETER_SENSITIVITY_PREDICTIONS_FILE_TEMPLATE.format(seed=seed)
                scores_path = prediction_dir / HYPERPARAMETER_SENSITIVITY_SCORES_FILE_TEMPLATE.format(seed=seed)
                params_path = prediction_dir / HYPERPARAMETER_SENSITIVITY_CLASSIFIER_PARAMS_FILE_TEMPLATE.format(seed=seed)
                classifier_params = classifier_params_by_classifier[classifier]
                existing_params_match = (
                    params_path.is_file()
                    and _params_equal(_read_json(params_path), classifier_params)
                )

                if (
                    output_path.is_file()
                    and scores_path.is_file()
                    and existing_params_match
                    and not args.overwrite
                ):
                    outputs = np.load(output_path, mmap_mode="r")
                    with scores_path.open("r", encoding="utf-8") as f:
                        scores = json.load(f)
                else:
                    downstream_result = _evaluate_downstream_for_embedding(
                        dataset_params=dataset_params,
                        algorithm=args.algorithm,
                        dimension=dimension,
                        embedding_path=Path(emb_row["embedding_path"]),
                        classifier=classifier,
                        classifier_params=classifier_params,
                        measures=args.downstream_performance_measures,
                        seed=EXPERIMENTS_DEFAULT_SEED,
                        prediction_batch_size=args.prediction_batch_size,
                        lp_train_batch_size=args.lp_train_batch_size,
                        lp_cache_disable_dimension_threshold=args.lp_cache_disable_dimension_threshold,
                        lp_edge_feature_op=args.lp_edge_feature_op,
                        lp_logreg_train_pos_sample_ratio=args.lp_logreg_train_pos_sample_ratio,
                        lp_resample_negative_per_epoch=args.lp_resample_negative_per_epoch,
                    )
                    outputs = downstream_result["outputs"]
                    scores = downstream_result["scores"]
                    np.save(output_path, outputs)
                    np.save(
                        pred_path,
                        np.column_stack(
                            (
                                downstream_result["y_test"],
                                downstream_result["y_pred"],
                            )
                        ),
                    )
                    _write_json(scores_path, scores)
                    _write_json(params_path, classifier_params)

                prediction_outputs[(dimension, config_label, classifier, seed)] = np.asarray(outputs)
                prediction_scores[(dimension, config_label, classifier, seed)] = {
                    key: float(value) for key, value in scores.items()
                }
                row = {
                    "dataset": args.dataset,
                    "algorithm": args.algorithm,
                    "dimension": dimension,
                    "config_label": config_label,
                    "embedding_index": embedding_index,
                    "seed": seed,
                    "classifier": classifier,
                    "output_path": str(output_path),
                    "prediction_path": str(pred_path),
                    "classifier_params": _canonical_params(classifier_params),
                }
                row.update(scores)
                downstream_rows.append(row)

        save_dir = _stage2_embedding_dir(output_dir, dimension, config_label)
        seeds = [args.seed_start + i for i in range(args.num_stage2_embeddings)]
        seed_pairs = list(combinations(seeds, 2))
        for measure in args.representational_measures:
            for left_seed, right_seed in seed_pairs:
                rep_row = {
                    "dataset": args.dataset,
                    "algorithm": args.algorithm,
                    "dimension": dimension,
                    "config_label": config_label,
                    "measure": measure,
                    "seed_left": left_seed,
                    "seed_right": right_seed,
                }
                rep_key = tuple(
                    str(rep_row.get(column, ""))
                    for column in ["dataset", "algorithm", "dimension", "config_label", "measure", "seed_left", "seed_right"]
                )
                if rep_key in existing_rep_keys:
                    continue
                left = np.load(save_dir / EMBEDDING_FILE_NAME(model_seed=left_seed), mmap_mode="r")
                right = np.load(save_dir / EMBEDDING_FILE_NAME(model_seed=right_seed), mmap_mode="r")
                value = ALL_MEASURES[measure](left, right, shape=ND_SHAPE)
                rep_row["value"] = float(value)
                rep_rows.append(rep_row)

        for classifier in args.classifiers:
            for measure in args.functional_measures:
                if measure in GROUPWISE_FUNCTIONAL_SIMILARITY_MEASURES:
                    func_row = {
                        "dataset": args.dataset,
                        "algorithm": args.algorithm,
                        "dimension": dimension,
                        "config_label": config_label,
                        "classifier": classifier,
                        "measure": measure,
                        "seeds": ";".join(str(seed) for seed in seeds),
                        "num_outputs": len(seeds),
                    }
                    func_key = tuple(str(value) for value in _functional_row_key(func_row))
                    if func_key in existing_func_keys:
                        continue
                    outputs = [
                        prediction_outputs[(dimension, config_label, classifier, seed)]
                        for seed in seeds
                    ]
                    value = ALL_FUNCSIM_MEASURES[measure](outputs)
                    func_row["num_outputs"] = len(outputs)
                    func_row["value"] = float(value)
                    func_rows.append(func_row)
                    continue

                for left_seed, right_seed in seed_pairs:
                    func_row = {
                        "dataset": args.dataset,
                        "algorithm": args.algorithm,
                        "dimension": dimension,
                        "config_label": config_label,
                        "classifier": classifier,
                        "measure": measure,
                        "seed_left": left_seed,
                        "seed_right": right_seed,
                    }
                    func_key = tuple(str(value) for value in _functional_row_key(func_row))
                    if func_key in existing_func_keys:
                        continue
                    left_output = prediction_outputs[(dimension, config_label, classifier, left_seed)]
                    right_output = prediction_outputs[(dimension, config_label, classifier, right_seed)]
                    if measure in PAIRWISE_FUNCTIONAL_SIMILARITY_MEASURES:
                        value = ALL_FUNCSIM_MEASURES[measure](left_output, right_output)
                    elif measure in PERFORMANCE_BASED_FUNCTIONAL_SIMILARITY_MEASURES:
                        left_scores = prediction_scores[(dimension, config_label, classifier, left_seed)]
                        right_scores = prediction_scores[(dimension, config_label, classifier, right_seed)]
                        if ACCURACY_SCORE not in left_scores or ACCURACY_SCORE not in right_scores:
                            raise ValueError(
                                f"{measure} requires {ACCURACY_SCORE!r} in "
                                "--downstream_performance_measures."
                            )
                        value = ALL_FUNCSIM_MEASURES[measure](
                            left_output,
                            right_output,
                            left_scores[ACCURACY_SCORE],
                            right_scores[ACCURACY_SCORE],
                        )
                    else:
                        raise ValueError(f"Unsupported functional measure type: {measure}")
                    func_row["value"] = float(value)
                    func_rows.append(func_row)

    merged_embedding_rows = _merge_rows_by_key(
        existing_embedding_rows,
        embedding_rows,
        ["dataset", "algorithm", "dimension", "config_label", "embedding_index", "seed"],
    )
    merged_tuning_rows = _merge_rows_by_key(
        existing_tuning_rows,
        downstream_tuning_rows,
        ["dataset", "algorithm", "dimension", "config_label", "classifier", "tune_id"],
    )
    merged_downstream_rows = _merge_rows_by_key(
        existing_downstream_rows,
        downstream_rows,
        ["dataset", "algorithm", "dimension", "config_label", "embedding_index", "seed", "classifier"],
    )
    merged_rep_rows = _merge_rows_by_key(
        existing_rep_rows,
        rep_rows,
        ["dataset", "algorithm", "dimension", "config_label", "measure", "seed_left", "seed_right"],
    )
    merged_func_rows = _merge_functional_rows(existing_func_rows, func_rows)

    _write_csv(embedding_report_path, merged_embedding_rows)
    _write_csv(tuning_report_path, merged_tuning_rows)
    _write_csv(downstream_report_path, merged_downstream_rows)
    _write_csv(rep_report_path, merged_rep_rows)
    _write_csv(
        reports_dir / HYPERPARAMETER_SENSITIVITY_STAGE2_REPRESENTATIONAL_STABILITY_SUMMARY_FILE_NAME,
        _summarize_stability(merged_rep_rows),
    )
    _write_csv(func_report_path, merged_func_rows)
    _write_csv(
        reports_dir / HYPERPARAMETER_SENSITIVITY_STAGE2_FUNCTIONAL_STABILITY_SUMMARY_FILE_NAME,
        _summarize_stability(merged_func_rows),
    )

    _write_csv(
        reports_dir / HYPERPARAMETER_SENSITIVITY_STAGE2_DOWNSTREAM_PERFORMANCE_SUMMARY_FILE_NAME,
        _summarize_performance(merged_downstream_rows),
    )


def collect_existing_stage2_results(
    args: argparse.Namespace,
    output_dir: Path,
    reports_dir: Path,
) -> None:
    from stability.measures import (
        ALL_FUNCSIM_MEASURES,
        ALL_MEASURES,
        GROUPWISE_FUNCTIONAL_SIMILARITY_MEASURES,
        PAIRWISE_FUNCTIONAL_SIMILARITY_MEASURES,
        PERFORMANCE_BASED_FUNCTIONAL_SIMILARITY_MEASURES,
    )
    from stability.measures.repsim import ND_SHAPE

    invalid_rep = sorted(set(args.representational_measures) - set(ALL_MEASURES.keys()))
    invalid_func = sorted(set(args.functional_measures) - set(ALL_FUNCSIM_MEASURES.keys()))
    if invalid_rep:
        raise ValueError(f"Unknown representational measures: {invalid_rep}")
    if invalid_func:
        raise ValueError(f"Unknown functional measures: {invalid_func}")

    selected_path = reports_dir / HYPERPARAMETER_SENSITIVITY_STAGE2_SELECTED_DIMENSIONS_FILE_NAME
    embedding_report_path = reports_dir / HYPERPARAMETER_SENSITIVITY_STAGE2_EMBEDDING_RUNS_FILE_NAME
    tuning_report_path = reports_dir / HYPERPARAMETER_SENSITIVITY_STAGE2_DOWNSTREAM_TUNING_RESULTS_FILE_NAME
    downstream_report_path = reports_dir / HYPERPARAMETER_SENSITIVITY_STAGE2_DOWNSTREAM_PERFORMANCE_FILE_NAME
    rep_report_path = reports_dir / HYPERPARAMETER_SENSITIVITY_STAGE2_REPRESENTATIONAL_STABILITY_FILE_NAME
    func_report_path = reports_dir / HYPERPARAMETER_SENSITIVITY_STAGE2_FUNCTIONAL_STABILITY_FILE_NAME

    existing_embedding_rows: list[dict[str, Any]] = _keep_dimension_specific_stage2_rows(
        _read_csv_if_exists(embedding_report_path)
    )
    existing_tuning_rows: list[dict[str, Any]] = _keep_dimension_specific_stage2_rows(
        _read_csv_if_exists(tuning_report_path)
    )
    existing_downstream_rows: list[dict[str, Any]] = _keep_dimension_specific_stage2_rows(
        _read_csv_if_exists(downstream_report_path)
    )
    existing_rep_rows: list[dict[str, Any]] = _keep_dimension_specific_stage2_rows(_read_csv_if_exists(rep_report_path))
    existing_func_rows: list[dict[str, Any]] = _keep_dimension_specific_stage2_rows(_read_csv_if_exists(func_report_path))
    existing_rep_keys = _existing_row_keys(
        existing_rep_rows,
        ["dataset", "algorithm", "dimension", "config_label", "measure", "seed_left", "seed_right"],
    )

    embedding_rows: list[dict[str, Any]] = []
    tuning_rows: list[dict[str, Any]] = []
    downstream_rows: list[dict[str, Any]] = []
    rep_rows: list[dict[str, Any]] = []
    func_rows: list[dict[str, Any]] = []
    collected_dimensions: set[int] = set()
    seeds = [args.seed_start + i for i in range(args.num_stage2_embeddings)]
    seed_pairs = list(combinations(seeds, 2))

    for dimension in args.dimensions:
        if dimension == args.anchor_dimension:
            continue
        config_label = HYPERPARAMETER_SENSITIVITY_DIMENSION_SPECIFIC_LABEL
        save_dir = (
            output_dir
            / EMBEDDINGS_DIR_NAME
            / HYPERPARAMETER_SENSITIVITY_STAGE2_STABILITY_DIR_NAME
            / DIMENSION_SUBDIR_NAME(dimension)
            / config_label
        )
        available_embedding_seeds: set[int] = set()
        for embedding_index, seed in enumerate(seeds):
            embedding_path = save_dir / EMBEDDING_FILE_NAME(model_seed=seed)
            if not embedding_path.is_file():
                continue
            available_embedding_seeds.add(seed)
            collected_dimensions.add(dimension)
            embedding_rows.append(
                {
                    "dataset": args.dataset,
                    "algorithm": args.algorithm,
                    "dimension": dimension,
                    "config_label": config_label,
                    "embedding_index": embedding_index,
                    "seed": seed,
                    "status": "success",
                    "validation_score": None,
                    "embedding_path": str(embedding_path),
                }
            )

        for measure in args.representational_measures:
            for left_seed, right_seed in seed_pairs:
                if left_seed not in available_embedding_seeds or right_seed not in available_embedding_seeds:
                    continue
                rep_row = {
                    "dataset": args.dataset,
                    "algorithm": args.algorithm,
                    "dimension": dimension,
                    "config_label": config_label,
                    "measure": measure,
                    "seed_left": left_seed,
                    "seed_right": right_seed,
                }
                rep_key = tuple(
                    str(rep_row.get(column, ""))
                    for column in ["dataset", "algorithm", "dimension", "config_label", "measure", "seed_left", "seed_right"]
                )
                if rep_key in existing_rep_keys:
                    continue
                left = np.load(save_dir / EMBEDDING_FILE_NAME(model_seed=left_seed), mmap_mode="r")
                right = np.load(save_dir / EMBEDDING_FILE_NAME(model_seed=right_seed), mmap_mode="r")
                rep_row["value"] = float(ALL_MEASURES[measure](left, right, shape=ND_SHAPE))
                rep_rows.append(rep_row)

        for classifier in args.classifiers:
            if classifier == LOGISTIC_REGRESSION:
                tuning_summary_path = _stage2_downstream_tuning_path(
                    output_dir,
                    dimension,
                    config_label,
                    classifier,
                )
                tuning_summary = _read_json_if_exists(tuning_summary_path, {})
                for tune_key, summary in sorted(tuning_summary.items(), key=lambda item: int(item[0])):
                    tune_scores = summary.get(TUNING_SUMMARY_RESULTS_KEY, {})
                    accuracy_values = [
                        float(scores[ACCURACY_SCORE])
                        for scores in tune_scores.values()
                        if isinstance(scores, dict) and ACCURACY_SCORE in scores
                    ]
                    tuning_rows.append(
                        {
                            "dataset": args.dataset,
                            "algorithm": args.algorithm,
                            "dimension": dimension,
                            "config_label": config_label,
                            "classifier": classifier,
                            "tune_id": int(tune_key),
                            "params": _canonical_params(summary[TUNING_SUMMARY_PARAMS_KEY]),
                            "mean_validation_score": summary[TUNING_SUMMARY_SCORE_KEY],
                            "std_validation_score": _row_std(accuracy_values),
                            "num_embeddings": len(accuracy_values),
                        }
                    )

            prediction_dir = (
                output_dir
                / PREDICTIONS_DIR_NAME
                / HYPERPARAMETER_SENSITIVITY_STAGE2_STABILITY_DIR_NAME
                / DIMENSION_SUBDIR_NAME(dimension)
                / config_label
                / classifier
            )
            available_prediction_seeds: set[int] = set()
            prediction_outputs: dict[int, np.ndarray] = {}
            prediction_scores: dict[int, dict[str, float]] = {}
            for embedding_index, seed in enumerate(seeds):
                output_path = prediction_dir / HYPERPARAMETER_SENSITIVITY_OUTPUTS_FILE_TEMPLATE.format(seed=seed)
                pred_path = prediction_dir / HYPERPARAMETER_SENSITIVITY_PREDICTIONS_FILE_TEMPLATE.format(seed=seed)
                scores_path = prediction_dir / HYPERPARAMETER_SENSITIVITY_SCORES_FILE_TEMPLATE.format(seed=seed)
                params_path = prediction_dir / HYPERPARAMETER_SENSITIVITY_CLASSIFIER_PARAMS_FILE_TEMPLATE.format(seed=seed)
                if not output_path.is_file() or not scores_path.is_file():
                    continue
                scores = _read_json(scores_path)
                classifier_params = _read_json(params_path) if params_path.is_file() else None
                row = {
                    "dataset": args.dataset,
                    "algorithm": args.algorithm,
                    "dimension": dimension,
                    "config_label": config_label,
                    "embedding_index": embedding_index,
                    "seed": seed,
                    "classifier": classifier,
                    "output_path": str(output_path),
                    "prediction_path": str(pred_path),
                }
                if classifier_params is not None:
                    row["classifier_params"] = _canonical_params(classifier_params)
                row.update({key: float(value) for key, value in scores.items()})
                downstream_rows.append(row)
                available_prediction_seeds.add(seed)
                prediction_scores[seed] = {key: float(value) for key, value in scores.items()}
                prediction_outputs[seed] = np.load(output_path, mmap_mode="r")
                collected_dimensions.add(dimension)

            for measure in args.functional_measures:
                if measure in GROUPWISE_FUNCTIONAL_SIMILARITY_MEASURES:
                    if not all(seed in available_prediction_seeds for seed in seeds):
                        continue
                    outputs = [prediction_outputs[seed] for seed in seeds]
                    func_rows.append(
                        {
                            "dataset": args.dataset,
                            "algorithm": args.algorithm,
                            "dimension": dimension,
                            "config_label": config_label,
                            "classifier": classifier,
                            "measure": measure,
                            "seeds": ";".join(str(seed) for seed in seeds),
                            "num_outputs": len(outputs),
                            "value": float(ALL_FUNCSIM_MEASURES[measure](outputs)),
                        }
                    )
                    continue

                for left_seed, right_seed in seed_pairs:
                    if left_seed not in available_prediction_seeds or right_seed not in available_prediction_seeds:
                        continue
                    left_output = prediction_outputs[left_seed]
                    right_output = prediction_outputs[right_seed]
                    if measure in PAIRWISE_FUNCTIONAL_SIMILARITY_MEASURES:
                        value = ALL_FUNCSIM_MEASURES[measure](left_output, right_output)
                    elif measure in PERFORMANCE_BASED_FUNCTIONAL_SIMILARITY_MEASURES:
                        left_scores = prediction_scores[left_seed]
                        right_scores = prediction_scores[right_seed]
                        if ACCURACY_SCORE not in left_scores or ACCURACY_SCORE not in right_scores:
                            continue
                        value = ALL_FUNCSIM_MEASURES[measure](
                            left_output,
                            right_output,
                            left_scores[ACCURACY_SCORE],
                            right_scores[ACCURACY_SCORE],
                        )
                    else:
                        raise ValueError(f"Unsupported functional measure type: {measure}")
                    func_rows.append(
                        {
                            "dataset": args.dataset,
                            "algorithm": args.algorithm,
                            "dimension": dimension,
                            "config_label": config_label,
                            "classifier": classifier,
                            "measure": measure,
                            "seed_left": left_seed,
                            "seed_right": right_seed,
                            "value": float(value),
                        }
                    )

    merged_selected_dimensions = sorted(
        set(int(dimension) for dimension in _read_json_if_exists(selected_path, [])).union(collected_dimensions)
    )
    merged_embedding_rows = _merge_rows_by_key(
        existing_embedding_rows,
        embedding_rows,
        ["dataset", "algorithm", "dimension", "config_label", "embedding_index", "seed"],
    )
    merged_tuning_rows = _merge_rows_by_key(
        existing_tuning_rows,
        tuning_rows,
        ["dataset", "algorithm", "dimension", "config_label", "classifier", "tune_id"],
    )
    merged_downstream_rows = _merge_rows_by_key(
        existing_downstream_rows,
        downstream_rows,
        ["dataset", "algorithm", "dimension", "config_label", "embedding_index", "seed", "classifier"],
    )
    merged_rep_rows = _merge_rows_by_key(
        existing_rep_rows,
        rep_rows,
        ["dataset", "algorithm", "dimension", "config_label", "measure", "seed_left", "seed_right"],
    )
    merged_func_rows = _merge_functional_rows(existing_func_rows, func_rows)

    _write_json(selected_path, merged_selected_dimensions)
    _write_csv(embedding_report_path, merged_embedding_rows)
    _write_csv(tuning_report_path, merged_tuning_rows)
    _write_csv(downstream_report_path, merged_downstream_rows)
    _write_csv(rep_report_path, merged_rep_rows)
    _write_csv(
        reports_dir / HYPERPARAMETER_SENSITIVITY_STAGE2_REPRESENTATIONAL_STABILITY_SUMMARY_FILE_NAME,
        _summarize_stability(merged_rep_rows),
    )
    _write_csv(func_report_path, merged_func_rows)
    _write_csv(
        reports_dir / HYPERPARAMETER_SENSITIVITY_STAGE2_FUNCTIONAL_STABILITY_SUMMARY_FILE_NAME,
        _summarize_stability(merged_func_rows),
    )
    _write_csv(
        reports_dir / HYPERPARAMETER_SENSITIVITY_STAGE2_DOWNSTREAM_PERFORMANCE_SUMMARY_FILE_NAME,
        _summarize_performance(merged_downstream_rows),
    )

    print(f"Collected existing Stage 2 results for dimensions: {sorted(collected_dimensions)}")


def _summarize_performance(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["dataset"], row["algorithm"], row["dimension"], row["config_label"], row["classifier"])
        grouped.setdefault(key, []).append(row)
    summary_rows = []
    for (dataset, algorithm, dimension, config_label, classifier), group_rows in sorted(grouped.items()):
        row = {
            "dataset": dataset,
            "algorithm": algorithm,
            "dimension": dimension,
            "config_label": config_label,
            "classifier": classifier,
            "num_embeddings": len(group_rows),
        }
        for measure in DOWNSTREAM_PERFORMANCE_MEASURES:
            values = [float(item[measure]) for item in group_rows if item.get(measure) is not None]
            if values:
                row[f"{measure}_mean"] = _row_mean(values)
                row[f"{measure}_std"] = _row_std(values)
        summary_rows.append(row)
    return summary_rows


def _summarize_stability(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(
            row[col]
            for col in ["dataset", "algorithm", "dimension", "config_label", "classifier", "measure"]
            if col in row
        )
        grouped.setdefault(key, []).append(row)

    summary_rows = []
    for key, group_rows in sorted(grouped.items()):
        sample = group_rows[0]
        out = {
            "dataset": sample["dataset"],
            "algorithm": sample["algorithm"],
            "dimension": sample["dimension"],
            "config_label": sample["config_label"],
            "measure": sample["measure"],
            "num_pairs": len(group_rows),
            "value_mean": _row_mean([float(row["value"]) for row in group_rows]),
            "value_std": _row_std([float(row["value"]) for row in group_rows]),
        }
        if "classifier" in sample:
            out["classifier"] = sample["classifier"]
        summary_rows.append(out)
    return summary_rows


def main() -> None:
    args = parse_args()
    if args.num_tuning_seeds < 1:
        raise ValueError("--num_tuning_seeds must be at least 1.")
    if args.num_stage2_tuning_embeddings < 1:
        raise ValueError("--num_stage2_tuning_embeddings must be at least 1.")
    if args.num_stage2_embeddings < 2 and "stability" in args.stages:
        raise ValueError("--num_stage2_embeddings must be at least 2 for stability calculations.")
    if args.stage2_tune_logreg and args.num_stage2_tuning_embeddings > args.num_stage2_embeddings:
        print(
            "--num_stage2_tuning_embeddings exceeds --num_stage2_embeddings; "
            "using all successful Stage 2 embeddings for LogisticRegression tuning."
        )
    if args.prediction_batch_size <= 0:
        raise ValueError("--prediction_batch_size must be positive.")
    if args.lp_train_batch_size is not None and args.lp_train_batch_size < 0:
        raise ValueError("--lp_train_batch_size must be non-negative.")
    if not (0 < args.lp_logreg_train_pos_sample_ratio <= 1.0):
        raise ValueError("--lp_logreg_train_pos_sample_ratio must be in (0, 1].")

    dataset_params = _dataset_params(args.dataset)
    output_dir = (
        Path(HYPERPARAMETER_SENSITIVITY_CASE_STUDY_OUTPUT_DIR)
        / args.algorithm
        / dataset_params[CONFIG_DATASET_NAME_KEY]
        / BUILD_DATASET_DIR_NAME(dataset_params)
    )
    reports_dir = output_dir / REPORTS_DIR_NAME
    logs_dir = output_dir / LOGS_DIR_NAME
    reports_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    run_metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "algorithm": args.algorithm,
        "dataset": args.dataset,
        "dataset_variant": BUILD_DATASET_DIR_NAME(dataset_params),
        "output_dir": str(output_dir),
        "dimensions": args.dimensions,
        "anchor_dimension": args.anchor_dimension,
        "stages": args.stages,
        "num_tuning_seeds": args.num_tuning_seeds,
        "num_stage2_embeddings": args.num_stage2_embeddings,
        "n_jobs": args.n_jobs,
        "stage2_tune_logreg": args.stage2_tune_logreg,
        "num_stage2_tuning_embeddings": args.num_stage2_tuning_embeddings,
        "collect_existing_stage2_results": args.collect_existing_stage2_results,
        "classifiers": args.classifiers,
        "representational_measures": args.representational_measures,
        "functional_measures": args.functional_measures,
        "material_improvement": args.material_improvement,
        "overwrite": args.overwrite,
        "embedding_training_scope": HYPERPARAMETER_SENSITIVITY_EMBEDDING_TRAINING_SCOPE,
        "stage1_anchor_dimension_source": "main tuning summary from regular output/embeddings tree",
        "stage2_case_study_config_labels": [HYPERPARAMETER_SENSITIVITY_DIMENSION_SPECIFIC_LABEL],
        "stage2_reference_source": "regular output/ embeddings, downstream results, and stability results",
        "args": vars(args),
    }
    _write_json(reports_dir / RUN_METADATA_FILE_NAME, run_metadata)
    metadata_history_path = reports_dir / RUN_METADATA_HISTORY_FILE_NAME
    metadata_history_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_history_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(run_metadata, sort_keys=True) + "\n")

    comparison_rows: list[dict[str, Any]]
    if args.collect_existing_stage2_results:
        if "stability" not in args.stages:
            raise ValueError("--collect_existing_stage2_results requires the stability stage.")
        comparison_rows = []
    elif "retune" in args.stages:
        comparison_rows = run_stage1_retuning(args, output_dir, logs_dir, reports_dir)
    else:
        comparison_rows = _load_stage1_comparison(reports_dir)

    if "stability" in args.stages and args.collect_existing_stage2_results:
        collect_existing_stage2_results(args, output_dir, reports_dir)
    elif "stability" in args.stages:
        run_stage2_stability(args, output_dir, logs_dir, reports_dir, comparison_rows)

    print(f"Hyperparameter sensitivity reports written to {reports_dir}")


if __name__ == "__main__":
    main()
