from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

from paths_globals import *
from plotting.viz import (
    _identify_statistical_peak_and_plateau_dims,
    identify_peak_and_plateau_dims,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Bootstrap whether the most stable embedding dimension is among the performance-optimal or "
            "near-optimal dimensions."
        )
    )
    parser.add_argument(
        "-a",
        "--algorithm",
        "--algorithms",
        dest="algorithms",
        nargs="+",
        choices=EMBEDDING_ALGORITHM_LIST,
        required=True,
    )
    parser.add_argument(
        "-d",
        "--dataset",
        "--datasets",
        dest="datasets",
        nargs="+",
        choices=EMPIRICAL_DATASET_LIST,
        required=True,
    )
    parser.add_argument("-m", "--stability_measure", "--stability_measures", dest="stability_measures", nargs="+", required=True)
    parser.add_argument(
        "-c",
        "--classifier",
        "--classifiers",
        dest="classifiers",
        nargs="+",
        choices=DOWNSTREAM_CLASSIFIERS,
        default=[LOGISTIC_REGRESSION],
        help="Downstream classifier for performance, and also for functional stability.",
    )
    parser.add_argument(
        "--metric",
        choices=DOWNSTREAM_PERFORMANCE_MEASURES,
        default=ACCURACY_SCORE,
        help="Downstream performance metric used to define optimal dimensions.",
    )
    parser.add_argument(
        "-dim",
        "--dimensions",
        nargs="+",
        type=int,
        default=EXPERIMENTS_DIMENSIONS_LIST,
        help="Embedding dimensions to include.",
    )
    parser.add_argument("--train_seed", type=int, default=EXPERIMENTS_DEFAULT_SEED)
    parser.add_argument("--n_bootstraps", type=int, default=10000)
    parser.add_argument(
        "--sample_size",
        type=int,
        default=EXPERIMENTS_NUM_ITERATIONS,
        help="Number of embedding indexes sampled with replacement in each bootstrap.",
    )
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument(
        "--performance_criterion",
        choices=["strict_best", "threshold", "statistical"],
        default="threshold",
        help=(
            "Criterion used to identify performance dimensions in each bootstrap sample. "
            "'strict_best' only tests overlap with the single best-performance dimension."
        ),
    )
    parser.add_argument(
        "--absolute_tolerance",
        type=float,
        default=0.01,
        help="Absolute threshold below the best performance score for threshold near-optimality.",
    )
    parser.add_argument(
        "--relative_tolerance",
        type=float,
        default=None,
        help="Optional relative threshold below the best performance score.",
    )
    parser.add_argument(
        "--min_plateau_size",
        type=int,
        default=1,
        help="Minimum number of near-optimal dimensions for threshold mode.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Alpha level for statistical near-optimality using Welch t-tests with Holm correction.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(STABILITY_PERFORMANCE_BOOTSTRAP_CASE_STUDY_OUTPUT_DIR),
        help="Directory where summary JSON and bootstrap CSV are written.",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute bootstrap results even when the corresponding summary and CSV already exist.",
    )
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Continue batch execution when one algorithm/dataset/measure/classifier combination fails.",
    )
    return parser.parse_args()


def _as_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def _stability_objective_for_measure(
    stability_measure: str,
    stability_type: STABILITY_TYPE,
) -> Literal["max", "min"]:
    from stability.measures import ALL_FUNCSIM_MEASURES, ALL_MEASURES

    if stability_type == REPRESENTATIONAL:
        measure = ALL_MEASURES.get(stability_measure)
    elif stability_type == FUNCTIONAL:
        measure = ALL_FUNCSIM_MEASURES.get(stability_measure)
    else:
        measure = None
    if measure is None:
        raise KeyError(f"Unknown {stability_type} stability measure {stability_measure!r}.")
    return "max" if measure.larger_is_more_similar else "min"


def _read_json_if_exists(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


def _infer_stability_type(
    *,
    dataset_params: Dict[str, Any],
    algorithm: EMBEDDING_ALGORITHM,
    stability_measure: str,
    classifier: Optional[DOWNSTREAM_CLASSIFIER],
) -> STABILITY_TYPE:
    representational_path = Path(CREATE_STABILITY_RESULTS_PATH(dataset_params, algorithm)) / STABILITY_RESULTS_JSON_FILE_NAME(
        REPRESENTATIONAL
    )
    representational_raw = _read_json_if_exists(representational_path)
    is_representational = isinstance(representational_raw, dict) and stability_measure in representational_raw

    is_functional = False
    if classifier is not None:
        functional_path = Path(CREATE_STABILITY_RESULTS_PATH(dataset_params, algorithm)) / STABILITY_RESULTS_JSON_FILE_NAME(
            FUNCTIONAL
        )
        functional_raw = _read_json_if_exists(functional_path)
        if isinstance(functional_raw, dict):
            classifier_dict = functional_raw.get(classifier)
            is_functional = isinstance(classifier_dict, dict) and stability_measure in classifier_dict

    if is_representational and not is_functional:
        return REPRESENTATIONAL
    if is_functional and not is_representational:
        return FUNCTIONAL
    if is_representational and is_functional:
        raise ValueError(
            f"Stability measure {stability_measure!r} exists in both representational and functional results for "
            f"{algorithm}/{dataset_params[CONFIG_DATASET_NAME_KEY]}/{classifier}; automatic inference is ambiguous."
        )
    raise ValueError(
        f"Could not infer stability type for measure {stability_measure!r} in "
        f"{algorithm}/{dataset_params[CONFIG_DATASET_NAME_KEY]}/{classifier}; no matching result file entry was found."
    )


def _load_stability_matrices(
    *,
    dataset_params: Dict[str, Any],
    algorithm: EMBEDDING_ALGORITHM,
    stability_type: STABILITY_TYPE,
    stability_measure: str,
    classifier: Optional[DOWNSTREAM_CLASSIFIER],
    dimensions: List[int],
    stability_objective: Literal["max", "min"],
) -> Dict[int, np.ndarray]:
    json_path = Path(CREATE_STABILITY_RESULTS_PATH(dataset_params, algorithm)) / STABILITY_RESULTS_JSON_FILE_NAME(
        stability_type
    )
    if not json_path.exists():
        raise FileNotFoundError(f"Missing stability results file: {json_path}")

    with open(json_path, "r") as f:
        raw = json.load(f)

    if stability_type == REPRESENTATIONAL:
        measure_dict = raw.get(stability_measure)
        if measure_dict is None:
            raise KeyError(f"Measure {stability_measure!r} not found in {json_path}.")
    else:
        if classifier is None:
            raise ValueError("--classifier is required for functional stability results.")
        classifier_dict = raw.get(classifier)
        if classifier_dict is None:
            raise KeyError(f"Classifier {classifier!r} not found in {json_path}.")
        measure_dict = classifier_dict.get(stability_measure)
        if measure_dict is None:
            raise KeyError(f"Measure {stability_measure!r} not found for classifier {classifier!r} in {json_path}.")

    matrices: Dict[int, np.ndarray] = {}
    for dim in dimensions:
        values = measure_dict.get(str(dim), measure_dict.get(dim))
        if values is None:
            continue
        if not isinstance(values, list):
            raise ValueError(
                f"{stability_type} measure {stability_measure!r} at dim={dim} is not pairwise. "
                "Groupwise scalar stability measures cannot be bootstrapped as pair matrices."
            )
        arr = np.asarray(values, dtype=float).ravel()
        pair_count = int(arr.size)
        disc = 1 + 8 * pair_count
        root = int(math.isqrt(disc))
        if root * root != disc or (1 + root) % 2 != 0:
            raise ValueError(f"Cannot infer n from {pair_count} pairwise values.")
        n = (1 + root) // 2
        if n * (n - 1) // 2 != pair_count:
            raise ValueError(f"Cannot infer n from {pair_count} pairwise values.")

        matrix = np.full((n, n), np.nan, dtype=float)
        diagonal_value = 0.0 if stability_objective == "min" else 1.0
        np.fill_diagonal(matrix, diagonal_value)
        tri_i, tri_j = np.triu_indices(n, k=1)
        matrix[tri_i, tri_j] = arr
        matrix[tri_j, tri_i] = arr
        matrices[int(dim)] = matrix

    return matrices


def _load_performance_vectors(
    *,
    dataset_params: Dict[str, Any],
    algorithm: EMBEDDING_ALGORITHM,
    classifier: DOWNSTREAM_CLASSIFIER,
    metric: DOWNSTREAM_PERFORMANCE_MEASURE,
    train_seed: int,
    dimensions: List[int],
    num_embeddings: int,
) -> Dict[int, np.ndarray]:
    vectors: Dict[int, np.ndarray] = {}
    for dim in dimensions:
        results_dir = Path(
            CREATE_DOWNSTREAM_RESULTS_PATH(
                dataset_params=dataset_params,
                embedding_name=algorithm,
                embedding_dim=dim,
                clf_name=classifier,
            )
        )
        json_path = results_dir / DOWNSTREAM_PERFORMANCE_JSON_FILE_NAME
        if not json_path.exists():
            continue

        with open(json_path, "r") as f:
            raw = json.load(f)
        metric_dict = raw.get(metric, {})
        vector = np.full(num_embeddings, np.nan, dtype=float)
        for emb_idx in range(num_embeddings):
            seed_map = metric_dict.get(str(emb_idx), metric_dict.get(emb_idx))
            if not isinstance(seed_map, dict):
                continue
            value = seed_map.get(str(train_seed), seed_map.get(train_seed))
            fvalue = _as_float(value)
            if fvalue is not None:
                vector[emb_idx] = fvalue
        if np.isfinite(vector).any():
            vectors[int(dim)] = vector

    return vectors


def _identify_performance_dims(
    by_dim_values: Dict[int, np.ndarray],
    *,
    criterion: STABILITY_PERFORMANCE_BOOTSTRAP_PERFORMANCE_CRITERION,
    absolute_tolerance: float,
    relative_tolerance: Optional[float],
    min_plateau_size: int,
    alpha: float,
) -> Dict[str, Any]:
    mean_by_dim = {
        dim: float(np.nanmean(values))
        for dim, values in by_dim_values.items()
        if values.size > 0 and np.isfinite(values).any()
    }
    clean_mean_by_dim = {dim: score for dim, score in mean_by_dim.items() if np.isfinite(score)}
    if clean_mean_by_dim:
        _, best_score = max(clean_mean_by_dim.items(), key=lambda item: (item[1], -int(item[0])))
        best_dims = sorted(
            int(dim)
            for dim, score in clean_mean_by_dim.items()
            if np.isclose(
                float(score),
                float(best_score),
                rtol=STABILITY_PERFORMANCE_BOOTSTRAP_TIE_RTOL,
                atol=STABILITY_PERFORMANCE_BOOTSTRAP_TIE_ATOL,
            )
        )
        best_score = float(best_score)
    else:
        best_dims = []
        best_score = np.nan
    if criterion == "strict_best":
        if not best_dims:
            return {"best_dim": None, "best_score": np.nan, "plateau_dims": []}
        return {
            "best_dim": int(best_dims[0]),
            "best_dims": best_dims,
            "best_score": float(best_score),
            "plateau_dims": best_dims,
        }

    if criterion == "threshold":
        summary = identify_peak_and_plateau_dims(
            mean_by_dim,
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
            min_plateau_size=min_plateau_size,
        )
        summary["best_dims"] = best_dims
        return summary

    list_by_dim = {
        dim: [float(v) for v in values[np.isfinite(values)].tolist()]
        for dim, values in by_dim_values.items()
        if values.size > 0 and np.isfinite(values).any()
    }
    summary = _identify_statistical_peak_and_plateau_dims(list_by_dim, alpha=alpha)
    summary["best_dims"] = best_dims
    return summary


def _frequency(counter: Counter, n_total: int) -> Dict[str, Dict[str, float]]:
    return {
        str(key): {"count": int(count), "share": float(count / n_total) if n_total else np.nan}
        for key, count in sorted(counter.items())
    }


def run_bootstrap(args: argparse.Namespace) -> Dict[str, Any]:
    if args.n_bootstraps <= 0:
        raise ValueError("--n_bootstraps must be positive.")
    if args.sample_size < 2:
        raise ValueError("--sample_size must be at least 2 to average pairwise stability values.")
    if args.classifier is None:
        raise ValueError("--classifier is required because downstream performance is classifier-specific.")

    dimensions = sorted({int(dim) for dim in args.dimensions})
    dataset_params = {
        CONFIG_DATASET_NAME_KEY: args.dataset,
        CONFIG_DATA_SUBSAMPLING_INDICATOR_KEY: False,
    }
    stability_type = _infer_stability_type(
        dataset_params=dataset_params,
        algorithm=args.algorithm,
        stability_measure=args.stability_measure,
        classifier=args.classifier,
    )

    stability_objective = _stability_objective_for_measure(args.stability_measure, stability_type)
    diagonal_value = 0.0 if stability_objective == "min" else 1.0

    stability_matrices = _load_stability_matrices(
        dataset_params=dataset_params,
        algorithm=args.algorithm,
        stability_type=stability_type,
        stability_measure=args.stability_measure,
        classifier=args.classifier,
        dimensions=dimensions,
        stability_objective=stability_objective,
    )
    if not stability_matrices:
        raise ValueError("No matching stability dimensions found.")

    inferred_num_embeddings = sorted({matrix.shape[0] for matrix in stability_matrices.values()})
    if len(inferred_num_embeddings) != 1:
        raise ValueError(f"Stability matrices disagree on embedding count: {inferred_num_embeddings}")
    num_embeddings = inferred_num_embeddings[0]

    performance_vectors = _load_performance_vectors(
        dataset_params=dataset_params,
        algorithm=args.algorithm,
        classifier=args.classifier,
        metric=args.metric,
        train_seed=args.train_seed,
        dimensions=dimensions,
        num_embeddings=num_embeddings,
    )
    if not performance_vectors:
        raise ValueError("No matching downstream performance dimensions found.")

    common_dims = sorted(set(stability_matrices.keys()).intersection(performance_vectors.keys()))
    if not common_dims:
        raise ValueError("No dimensions have both stability and downstream performance results.")

    rng = np.random.default_rng(args.random_seed)
    rows: List[Dict[str, Any]] = []
    hit_counter = 0
    perf_best_counter: Counter = Counter()
    stability_best_counter: Counter = Counter()
    plateau_size_counter: Counter = Counter()
    perf_best_set_size_counter: Counter = Counter()
    stability_best_set_size_counter: Counter = Counter()

    for bootstrap_id in range(args.n_bootstraps):
        sample_by_dim = {
            dim: rng.integers(0, num_embeddings, size=args.sample_size)
            for dim in common_dims
        }

        perf_sample_by_dim = {
            dim: performance_vectors[dim][sample_by_dim[dim]]
            for dim in common_dims
        }
        perf_summary = _identify_performance_dims(
            perf_sample_by_dim,
            criterion=args.performance_criterion,
            absolute_tolerance=args.absolute_tolerance,
            relative_tolerance=args.relative_tolerance,
            min_plateau_size=args.min_plateau_size,
            alpha=args.alpha,
        )
        plateau_dims = [int(dim) for dim in perf_summary.get("plateau_dims", [])]
        best_perf_dims = sorted({int(dim) for dim in perf_summary.get("best_dims", [])})
        best_perf_dim = best_perf_dims[0] if best_perf_dims else perf_summary.get("best_dim")
        if best_perf_dim is not None and not best_perf_dims:
            best_perf_dims = [int(best_perf_dim)]

        stability_by_dim = {}
        for dim in common_dims:
            sample = sample_by_dim[dim]
            tri_i, tri_j = np.triu_indices(sample.size, k=1)
            left = sample[tri_i]
            right = sample[tri_j]
            non_self_pair_mask = left != right
            vals = stability_matrices[dim][left[non_self_pair_mask], right[non_self_pair_mask]]
            vals = vals[np.isfinite(vals)]
            stability_by_dim[dim] = np.nan if vals.size == 0 else float(np.mean(vals))

        clean_stability = {dim: score for dim, score in stability_by_dim.items() if np.isfinite(score)}
        if clean_stability:
            if stability_objective == "min":
                _, stability_score = min(clean_stability.items(), key=lambda item: (item[1], int(item[0])))
            else:
                _, stability_score = max(clean_stability.items(), key=lambda item: (item[1], -int(item[0])))
            stability_dims = sorted(
                int(dim)
                for dim, score in clean_stability.items()
                if np.isclose(
                    float(score),
                    float(stability_score),
                    rtol=STABILITY_PERFORMANCE_BOOTSTRAP_TIE_RTOL,
                    atol=STABILITY_PERFORMANCE_BOOTSTRAP_TIE_ATOL,
                )
            )
            stability_score = float(stability_score)
        else:
            stability_dims = []
            stability_score = np.nan
        stability_dim = stability_dims[0] if stability_dims else None
        hit = bool(set(stability_dims).intersection(plateau_dims))

        if hit:
            hit_counter += 1
        for dim in best_perf_dims:
            perf_best_counter[int(dim)] += 1
        for dim in stability_dims:
            stability_best_counter[int(dim)] += 1
        plateau_size_counter[len(plateau_dims)] += 1
        perf_best_set_size_counter[len(best_perf_dims)] += 1
        stability_best_set_size_counter[len(stability_dims)] += 1

        rows.append(
            {
                "bootstrap_id": bootstrap_id,
                "sample_indexes_by_dimension": json.dumps(
                    {
                        str(dim): [int(idx) for idx in sample_by_dim[dim].tolist()]
                        for dim in common_dims
                    },
                    separators=(",", ":"),
                ),
                "performance_best_dim": "" if best_perf_dim is None else int(best_perf_dim),
                "performance_best_dims": ";".join(str(dim) for dim in best_perf_dims),
                "performance_best_score": perf_summary.get("best_score", np.nan),
                "performance_near_optimal_dims": ";".join(str(dim) for dim in plateau_dims),
                "stability_best_dim": "" if stability_dim is None else stability_dim,
                "stability_best_dims": ";".join(str(dim) for dim in stability_dims),
                "stability_best_score": stability_score,
                "hit": int(hit),
            }
        )

    hit_rate = hit_counter / args.n_bootstraps
    summary = {
        "algorithm": args.algorithm,
        "dataset": args.dataset,
        "classifier": args.classifier,
        "metric": args.metric,
        "stability_type": stability_type,
        "stability_measure": args.stability_measure,
        "stability_objective": stability_objective,
        "diagonal_value": diagonal_value,
        "performance_criterion": args.performance_criterion,
        "absolute_tolerance": args.absolute_tolerance,
        "relative_tolerance": args.relative_tolerance,
        "min_plateau_size": args.min_plateau_size,
        "alpha": args.alpha,
        "train_seed": args.train_seed,
        "random_seed": args.random_seed,
        "num_bootstraps": args.n_bootstraps,
        "num_embeddings": num_embeddings,
        "sample_size": args.sample_size,
        "stability_self_pairs_excluded": True,
        "dimensions": common_dims,
        "hit_count": hit_counter,
        "hit_rate": hit_rate,
        "hit_rate_monte_carlo_se": float(math.sqrt(hit_rate * (1.0 - hit_rate) / args.n_bootstraps)),
        "performance_best_dim_frequency": _frequency(perf_best_counter, args.n_bootstraps),
        "stability_best_dim_frequency": _frequency(stability_best_counter, args.n_bootstraps),
        "performance_plateau_size_frequency": _frequency(plateau_size_counter, args.n_bootstraps),
        "performance_best_set_size_frequency": _frequency(perf_best_set_size_counter, args.n_bootstraps),
        "stability_best_set_size_frequency": _frequency(stability_best_set_size_counter, args.n_bootstraps),
    }
    return {"summary": summary, "rows": rows}


def _path_token(value: Any) -> str:
    return str(value).replace("/", "-").replace("\\", "-")


def _output_paths_for_stem(output_dir: Path, stem: str) -> tuple[Path, Path]:
    return (
        output_dir / STABILITY_PERFORMANCE_BOOTSTRAP_RUN_SUMMARY_FILE_TEMPLATE.format(stem=stem),
        output_dir / STABILITY_PERFORMANCE_BOOTSTRAP_RUN_BOOTSTRAPS_FILE_TEMPLATE.format(stem=stem),
    )


def _would_hit_path_length_limit(output_dir: Path, stem: str) -> bool:
    summary_path, rows_path = _output_paths_for_stem(output_dir, stem)
    return (
        max(len(str(summary_path.resolve())), len(str(rows_path.resolve())))
        >= STABILITY_PERFORMANCE_BOOTSTRAP_MAX_OUTPUT_PATH_LENGTH
    )


def _resolved_stability_type_for_args(args: argparse.Namespace) -> STABILITY_TYPE:
    dataset_params = {
        CONFIG_DATASET_NAME_KEY: args.dataset,
        CONFIG_DATA_SUBSAMPLING_INDICATOR_KEY: False,
    }
    return _infer_stability_type(
        dataset_params=dataset_params,
        algorithm=args.algorithm,
        stability_measure=args.stability_measure,
        classifier=args.classifier,
    )


def _compact_run_stem(
    args: argparse.Namespace,
    resolved_stability_type: STABILITY_TYPE,
    full_stem: str,
) -> str:
    digest = hashlib.sha1(full_stem.encode("utf-8")).hexdigest()[:10]
    measure = _path_token(args.stability_measure)
    if len(measure) > 24:
        measure = f"{measure[:24]}"
    parts = [
        _path_token(args.algorithm),
        _path_token(args.dataset),
        _path_token(args.classifier or STABILITY_PERFORMANCE_BOOTSTRAP_RUN_STEM_ALL_CLASSIFIERS_TOKEN),
        _path_token(args.metric),
        {
            REPRESENTATIONAL: "rep",
            FUNCTIONAL: "func",
        }.get(resolved_stability_type, resolved_stability_type),
        measure,
        {"strict_best": "best", "threshold": "thr", "statistical": "stat"}.get(
            args.performance_criterion,
            args.performance_criterion,
        ),
        f"b{args.n_bootstraps}",
        f"s{args.random_seed}",
        digest,
    ]
    return "_".join(parts)


def _output_paths_for_run(args: argparse.Namespace, result: Optional[Dict[str, Any]] = None) -> tuple[Path, Path]:
    if args.run_id is None:
        resolved_stability_type = _resolved_stability_type_for_args(args)
        if result is not None:
            resolved_stability_type = result.get("summary", {}).get("stability_type", resolved_stability_type)
        parts = [
            args.algorithm,
            args.dataset,
            args.classifier or STABILITY_PERFORMANCE_BOOTSTRAP_RUN_STEM_ALL_CLASSIFIERS_TOKEN,
            args.metric,
            resolved_stability_type,
            args.stability_measure,
            args.performance_criterion,
            f"b{args.n_bootstraps}",
            f"s{args.random_seed}",
        ]
        stem = "_".join(_path_token(part) for part in parts)
        if _would_hit_path_length_limit(args.output_dir, stem):
            stem = _compact_run_stem(args, resolved_stability_type, stem)
    else:
        stem = args.run_id
    if _would_hit_path_length_limit(args.output_dir, stem):
        resolved_stability_type = _resolved_stability_type_for_args(args)
        if result is not None:
            resolved_stability_type = result.get("summary", {}).get("stability_type", resolved_stability_type)
        stem = _compact_run_stem(args, resolved_stability_type, stem)
    return _output_paths_for_stem(args.output_dir, stem)


def write_outputs(args: argparse.Namespace, result: Dict[str, Any]) -> tuple[Path, Path]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path, rows_path = _output_paths_for_run(args, result)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    rows_path.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(result["summary"], f, indent=2)

    fieldnames = [
        "bootstrap_id",
        "sample_indexes_by_dimension",
        "performance_best_dim",
        "performance_best_dims",
        "performance_best_score",
        "performance_near_optimal_dims",
        "stability_best_dim",
        "stability_best_dims",
        "stability_best_score",
        "hit",
    ]
    with open(rows_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result["rows"])

    return summary_path, rows_path


def main() -> None:
    args = parse_args()
    combinations = [
        (algorithm, dataset, classifier, stability_measure)
        for algorithm in args.algorithms
        for dataset in args.datasets
        for classifier in args.classifiers
        for stability_measure in args.stability_measures
    ]
    single_runs: List[argparse.Namespace] = []
    for algorithm, dataset, classifier, stability_measure in combinations:
        single_args = copy.copy(args)
        single_args.algorithm = algorithm
        single_args.dataset = dataset
        single_args.classifier = classifier
        single_args.stability_measure = stability_measure
        if args.run_id is None:
            single_args.run_id = None
        elif len(combinations) <= 1:
            single_args.run_id = args.run_id
        else:
            run_id_parts = [
                args.run_id,
                single_args.algorithm,
                single_args.dataset,
                single_args.classifier,
                single_args.stability_measure,
            ]
            single_args.run_id = "_".join(
                str(part).replace("/", "-").replace("\\", "-")
                for part in run_id_parts
            )
        single_runs.append(single_args)
    failures = []

    for run_idx, single_args in enumerate(single_runs, start=1):
        label = (
            f"{single_args.algorithm}/{single_args.dataset}/{single_args.classifier}/"
            f"{single_args.stability_measure}"
        )
        print(f"[{run_idx}/{len(single_runs)}] Running {label}")
        try:
            summary_path, rows_path = _output_paths_for_run(single_args)
            if summary_path.is_file() and rows_path.is_file() and not args.overwrite:
                print(f"Skip existing bootstrap result for {label}")
                print(f"Summary exists: {summary_path}")
                print(f"Bootstrap rows exist: {rows_path}")
                continue

            result = run_bootstrap(single_args)
            summary_path, rows_path = write_outputs(single_args, result)
        except Exception as exc:
            failures.append((label, exc))
            print(f"FAILED {label}: {exc}")
            if not args.continue_on_error:
                raise
            continue

        summary = result["summary"]
        print(
            f"Hit rate: {summary['hit_rate']:.4f} "
            f"({summary['hit_count']}/{summary['num_bootstraps']}); "
            f"stability_type={summary['stability_type']}"
        )
        print(f"Summary written to: {summary_path}")
        print(f"Bootstrap rows written to: {rows_path}")

    if failures:
        print("Failed combinations:")
        for label, exc in failures:
            print(f"- {label}: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
