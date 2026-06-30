import json
import os.path as osp
from typing import Any, Iterable, List, Literal, Optional, Sequence

import pandas as pd

from paths_globals import (
    CONFIG_DATA_SAMPLING_SEED_KEY,
    CONFIG_DATASET_NAME_KEY,
    CONFIG_DATA_SUBSAMPLING_INDICATOR_KEY,
    CONFIG_SYNTH_DATA_DENSITY_KEY,
    CONFIG_SYNTH_DATA_NUM_NODES_KEY,
    BUILD_DATASET_DIR_NAME,
    CREATE_MODELS_PATH,
    CREATE_SYNTH_TUNING_RESULTS_PATH,
    DATASET_RENAME_DICT,
    DGI,
    DIMENSION_SUBDIR_NAME,
    DOWNSTREAM_RESULTS_DIR,
    EMBEDDING_ALGORITHM,
    EMPIRICAL_DATASET_LIST,
    MULTILAYER_PERCEPTRON,
    SYNTHETIC_DATASET_LIST,
    SYNTH_DATA_EXPERIMENTS_DEFAULT_DENSITY,
    SYNTH_DATA_EXPERIMENTS_DEFAULT_NUM_NODES,
    SYNTH_DATA_EXPERIMENTS_DENSITIES_LIST,
    SYNTH_DATA_EXPERIMENTS_NUM_NODES_LIST,
    TUNING_DEFAULT_DIMENSION,
    TUNING_PARAM_GRID_DICT,
    TUNING_SUMMARY_FILE_NAME,
    TUNING_SUMMARY_PARAMS_KEY,
    TUNING_SUMMARY_RESULTS_KEY,
    TUNING_SUMMARY_SCORE_KEY,
)

SYNTHETIC_SWEEP_MODE = Literal["both", "vary_size", "vary_density"]


def _normalize_datasets(datasets: Optional[Sequence[str]]) -> List[str]:
    if datasets is None:
        return list(EMPIRICAL_DATASET_LIST)
    return list(datasets)


def _normalize_hyperparameters(
    embedding_method: EMBEDDING_ALGORITHM,
    hyperparameters: Optional[Sequence[str]],
) -> List[str]:
    if hyperparameters is None:
        return list(TUNING_PARAM_GRID_DICT[embedding_method].keys())

    invalid_hyperparameters = sorted(set(hyperparameters) - set(TUNING_PARAM_GRID_DICT[embedding_method].keys()))
    if invalid_hyperparameters:
        raise ValueError(
            f"Invalid hyperparameters for {embedding_method}: {invalid_hyperparameters}. "
            f"Available options are {list(TUNING_PARAM_GRID_DICT[embedding_method].keys())}."
        )
    return list(hyperparameters)


def _tuning_summary_path(embedding_method: EMBEDDING_ALGORITHM, dataset: str, dimension: int) -> str:
    dataset_params = {
        CONFIG_DATASET_NAME_KEY: dataset,
        CONFIG_DATA_SUBSAMPLING_INDICATOR_KEY: False,
    }
    tune_dir = CREATE_MODELS_PATH(
        dataset_params=dataset_params,
        embedding_name=embedding_method,
        embedding_dim=dimension,
        b_tune=True,
    )
    return osp.join(tune_dir, TUNING_SUMMARY_FILE_NAME)


def load_empirical_tuning_results(
    embedding_method: EMBEDDING_ALGORITHM,
    datasets: Optional[Sequence[str]] = None,
    hyperparameters: Optional[Sequence[str]] = None,
    dimension: int = TUNING_DEFAULT_DIMENSION,
    raise_if_missing: bool = False,
) -> pd.DataFrame:
    """Load empirical tuning summaries into a flat dataframe.

    Each row corresponds to one tuning configuration for one dataset.
    """

    datasets = _normalize_datasets(datasets)
    hyperparameters = _normalize_hyperparameters(embedding_method, hyperparameters)

    rows: List[dict[str, Any]] = []
    missing_paths: List[str] = []

    for dataset in datasets:
        tuning_summary_path = _tuning_summary_path(
            embedding_method=embedding_method,
            dataset=dataset,
            dimension=dimension,
        )

        if not osp.isfile(tuning_summary_path):
            missing_paths.append(tuning_summary_path)
            continue

        with open(tuning_summary_path, "r") as file:
            tuning_summary = json.load(file)

        for tune_id, summary in tuning_summary.items():
            params = summary.get(TUNING_SUMMARY_PARAMS_KEY, {})
            results = summary.get(TUNING_SUMMARY_RESULTS_KEY, {})
            row = {
                "dataset": dataset,
                "dataset_label": DATASET_RENAME_DICT.get(dataset, dataset),
                "dimension": dimension,
                "tune_id": int(tune_id),
                "avg_accuracy": summary.get(TUNING_SUMMARY_SCORE_KEY),
                "num_runs": len(results),
            }
            for hyperparameter in hyperparameters:
                row[hyperparameter] = params.get(hyperparameter)
            rows.append(row)

    if missing_paths and raise_if_missing:
        missing_lines = "\n".join(missing_paths)
        raise FileNotFoundError(f"Missing tuning summaries:\n{missing_lines}")

    if not rows:
        columns = ["dataset", "dataset_label", "dimension", "tune_id", "avg_accuracy", "num_runs", *hyperparameters]
        return pd.DataFrame(columns=columns)

    sort_columns: List[str] = ["dataset_label"] + hyperparameters
    return pd.DataFrame(rows).sort_values(sort_columns).reset_index(drop=True)


def tuning_accuracy_table(
    embedding_method: EMBEDDING_ALGORITHM,
    datasets: Optional[Sequence[str]] = None,
    hyperparameters: Optional[Sequence[str]] = None,
    dimension: int = TUNING_DEFAULT_DIMENSION,
    raise_if_missing: bool = False,
) -> pd.DataFrame:
    """Build a table of tuning hyperparameters against aggregate accuracy.

    If exactly one hyperparameter is requested, a pivot table is returned:
    rows are datasets, columns are hyperparameter values, and cells are aggregate accuracy.

    If multiple hyperparameters are requested, a flat dataframe is returned with one
    row per dataset and tuning configuration.
    """

    resolved_hyperparameters = _normalize_hyperparameters(embedding_method, hyperparameters)
    tuning_results = load_empirical_tuning_results(
        embedding_method=embedding_method,
        datasets=datasets,
        hyperparameters=resolved_hyperparameters,
        dimension=dimension,
        raise_if_missing=raise_if_missing,
    )

    if len(resolved_hyperparameters) == 1:
        hyperparameter = resolved_hyperparameters[0]
        table = tuning_results.pivot(
            index="dataset_label",
            columns=hyperparameter,
            values="avg_accuracy",
        )
        best_runs = best_tuning_runs(
            embedding_method=embedding_method,
            datasets=datasets,
            hyperparameters=resolved_hyperparameters,
            dimension=dimension,
            raise_if_missing=raise_if_missing,
        ).set_index("dataset_label")
        table = table.sort_index(axis=0).sort_index(axis=1)
        table["best_{}".format(hyperparameter)] = best_runs[hyperparameter]
        table["best_tune_id"] = best_runs["tune_id"]
        table["best_avg_accuracy"] = best_runs["avg_accuracy"]
        return table

    result_columns = ["dataset", "dataset_label", *resolved_hyperparameters, "avg_accuracy", "num_runs", "tune_id"]
    return tuning_results.loc[:, result_columns].reset_index(drop=True)


def best_tuning_runs(
    embedding_method: EMBEDDING_ALGORITHM,
    datasets: Optional[Sequence[str]] = None,
    hyperparameters: Optional[Sequence[str]] = None,
    dimension: int = TUNING_DEFAULT_DIMENSION,
    raise_if_missing: bool = False,
) -> pd.DataFrame:
    """Return the best tuning configuration per empirical dataset."""

    resolved_hyperparameters = _normalize_hyperparameters(embedding_method, hyperparameters)
    tuning_results = load_empirical_tuning_results(
        embedding_method=embedding_method,
        datasets=datasets,
        hyperparameters=resolved_hyperparameters,
        dimension=dimension,
        raise_if_missing=raise_if_missing,
    )

    if tuning_results.empty:
        return tuning_results

    best_idx = tuning_results.groupby("dataset")["avg_accuracy"].idxmax()
    result_columns = ["dataset", "dataset_label", *resolved_hyperparameters, "avg_accuracy", "num_runs", "tune_id"]
    return tuning_results.loc[best_idx, result_columns].sort_values("dataset_label").reset_index(drop=True)


def available_empirical_tuning_datasets(
    embedding_method: EMBEDDING_ALGORITHM,
    datasets: Optional[Iterable[str]] = None,
    dimension: int = TUNING_DEFAULT_DIMENSION,
) -> List[str]:
    """Return empirical datasets for which a tuning summary currently exists."""

    resolved_datasets = _normalize_datasets(list(datasets) if datasets is not None else None)
    return [
        dataset
        for dataset in resolved_datasets
        if osp.isfile(_tuning_summary_path(embedding_method=embedding_method, dataset=dataset, dimension=dimension))
    ]


def _downstream_tuning_summary_path(
    embedding_method: EMBEDDING_ALGORITHM,
    dataset: str,
    classifier: str,
    dimension: int,
) -> str:
    dataset_params = {
        CONFIG_DATASET_NAME_KEY: dataset,
        CONFIG_DATA_SUBSAMPLING_INDICATOR_KEY: False,
    }
    return osp.join(
        DOWNSTREAM_RESULTS_DIR,
        embedding_method,
        dataset,
        BUILD_DATASET_DIR_NAME(dataset_params),
        classifier,
        DIMENSION_SUBDIR_NAME(dimension),
        TUNING_SUMMARY_FILE_NAME,
    )


def dgi_mlp_regularization_tuning_table(
    datasets: Optional[Sequence[str]] = None,
    dimension: int = TUNING_DEFAULT_DIMENSION,
    raise_if_missing: bool = False,
) -> pd.DataFrame:
    """Return DGI/MLP downstream tuning accuracy by regularization weight.

    Rows are MLP ``alpha`` values, columns are empirical datasets, and cells are
    mean validation accuracy from the corresponding downstream ``tuning_results.json``.
    """

    rows: List[dict[str, Any]] = []
    missing_paths: List[str] = []
    resolved_datasets = _normalize_datasets(datasets)

    for dataset in resolved_datasets:
        tuning_summary_path = _downstream_tuning_summary_path(
            embedding_method=DGI,
            dataset=dataset,
            classifier=MULTILAYER_PERCEPTRON,
            dimension=dimension,
        )

        if not osp.isfile(tuning_summary_path):
            missing_paths.append(tuning_summary_path)
            continue

        with open(tuning_summary_path, "r") as file:
            tuning_summary = json.load(file)

        for tune_id, summary in tuning_summary.items():
            params = summary.get(TUNING_SUMMARY_PARAMS_KEY, {})
            alpha = params.get("alpha")
            if alpha is None:
                continue

            results = summary.get(TUNING_SUMMARY_RESULTS_KEY, {})
            accuracies = results.get("accuracy", [])
            if accuracies:
                avg_accuracy = sum(accuracies) / len(accuracies)
                num_runs = len(accuracies)
            else:
                avg_accuracy = summary.get(TUNING_SUMMARY_SCORE_KEY)
                num_runs = len(results)

            rows.append(
                {
                    "dataset": dataset,
                    "dataset_label": DATASET_RENAME_DICT.get(dataset, dataset),
                    "dimension": dimension,
                    "alpha": alpha,
                    "avg_accuracy": avg_accuracy,
                    "num_runs": num_runs,
                    "tune_id": int(tune_id),
                }
            )

    if missing_paths and raise_if_missing:
        missing_lines = "\n".join(missing_paths)
        raise FileNotFoundError(f"Missing downstream tuning summaries:\n{missing_lines}")

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    table = result.pivot_table(
        index="alpha",
        columns="dataset_label",
        values="avg_accuracy",
        aggfunc="mean",
    )
    dataset_columns = [DATASET_RENAME_DICT.get(dataset, dataset) for dataset in resolved_datasets]
    table = table.reindex(columns=dataset_columns)
    return table.sort_index(axis=0).sort_index(axis=1)


def dgi_mlp_optimal_regularization_by_dimension_table(
    datasets: Optional[Sequence[str]] = None,
    dimensions: Optional[Sequence[int]] = None,
    raise_if_missing: bool = False,
) -> pd.DataFrame:
    """Return the best DGI/MLP regularization weight for each dataset and dimension.

    Rows are empirical datasets, columns are embedding dimensions, and cells are the
    MLP ``alpha`` value whose downstream tuning run has the highest mean accuracy.
    """

    resolved_datasets = _normalize_datasets(datasets)
    resolved_dimensions = list(dimensions) if dimensions is not None else [2**i for i in range(2, 13)]
    rows: List[dict[str, Any]] = []
    missing_paths: List[str] = []

    for dataset in resolved_datasets:
        for dimension in resolved_dimensions:
            tuning_summary_path = _downstream_tuning_summary_path(
                embedding_method=DGI,
                dataset=dataset,
                classifier=MULTILAYER_PERCEPTRON,
                dimension=dimension,
            )

            if not osp.isfile(tuning_summary_path):
                missing_paths.append(tuning_summary_path)
                continue

            with open(tuning_summary_path, "r") as file:
                tuning_summary = json.load(file)

            best_alpha = None
            best_accuracy = None
            best_tune_id = None

            for tune_id, summary in tuning_summary.items():
                params = summary.get(TUNING_SUMMARY_PARAMS_KEY, {})
                alpha = params.get("alpha")
                if alpha is None:
                    continue

                results = summary.get(TUNING_SUMMARY_RESULTS_KEY, {})
                accuracies = results.get("accuracy", [])
                if accuracies:
                    avg_accuracy = sum(accuracies) / len(accuracies)
                else:
                    avg_accuracy = summary.get(TUNING_SUMMARY_SCORE_KEY)

                if avg_accuracy is None:
                    continue
                if best_accuracy is None or avg_accuracy > best_accuracy:
                    best_alpha = alpha
                    best_accuracy = avg_accuracy
                    best_tune_id = int(tune_id)

            if best_alpha is not None:
                rows.append(
                    {
                        "dataset": dataset,
                        "dataset_label": DATASET_RENAME_DICT.get(dataset, dataset),
                        "dimension": dimension,
                        "best_alpha": best_alpha,
                        "best_avg_accuracy": best_accuracy,
                        "best_tune_id": best_tune_id,
                    }
                )

    if missing_paths and raise_if_missing:
        missing_lines = "\n".join(missing_paths)
        raise FileNotFoundError(f"Missing downstream tuning summaries:\n{missing_lines}")

    if not rows:
        return pd.DataFrame(index=[DATASET_RENAME_DICT.get(dataset, dataset) for dataset in resolved_datasets])

    result = pd.DataFrame(rows)
    table = result.pivot(
        index="dataset_label",
        columns="dimension",
        values="best_alpha",
    )
    dataset_index = [DATASET_RENAME_DICT.get(dataset, dataset) for dataset in resolved_datasets]
    table = table.reindex(index=dataset_index, columns=resolved_dimensions)
    return table.sort_index(axis=0).sort_index(axis=1)


def _format_density(density: float) -> str:
    return "{:g}".format(density)


def _synthetic_configuration_label(num_nodes: int, density: float) -> str:
    return "n={}; d={}".format(num_nodes, _format_density(density))


def _synthetic_sweep_configurations(
    sweep_mode: SYNTHETIC_SWEEP_MODE = "both",
    num_nodes_list: Optional[Sequence[int]] = None,
    density_list: Optional[Sequence[float]] = None,
    fixed_num_nodes: int = SYNTH_DATA_EXPERIMENTS_DEFAULT_NUM_NODES,
    fixed_density: float = SYNTH_DATA_EXPERIMENTS_DEFAULT_DENSITY,
) -> List[dict[str, Any]]:
    if sweep_mode not in ("both", "vary_size", "vary_density"):
        raise ValueError("sweep_mode must be one of ['both', 'vary_size', 'vary_density'].")

    resolved_num_nodes_list = list(num_nodes_list) if num_nodes_list is not None else list(SYNTH_DATA_EXPERIMENTS_NUM_NODES_LIST)
    resolved_density_list = list(density_list) if density_list is not None else list(SYNTH_DATA_EXPERIMENTS_DENSITIES_LIST)

    configurations: List[dict[str, Any]] = []
    seen_configurations = set()

    def add_configuration(num_nodes: int, density: float, sweep_type: str) -> None:
        key = (num_nodes, density)
        if key in seen_configurations:
            return
        seen_configurations.add(key)
        configurations.append(
            {
                CONFIG_SYNTH_DATA_NUM_NODES_KEY: num_nodes,
                CONFIG_SYNTH_DATA_DENSITY_KEY: density,
                "sweep_type": sweep_type,
                "configuration_label": _synthetic_configuration_label(num_nodes, density),
            }
        )

    if sweep_mode in ("both", "vary_size"):
        for num_nodes in resolved_num_nodes_list:
            add_configuration(num_nodes=num_nodes, density=fixed_density, sweep_type="vary_size")

    if sweep_mode in ("both", "vary_density"):
        for density in resolved_density_list:
            add_configuration(num_nodes=fixed_num_nodes, density=density, sweep_type="vary_density")

    if sweep_mode == "vary_size":
        configurations.sort(key=lambda row: row[CONFIG_SYNTH_DATA_NUM_NODES_KEY])
    elif sweep_mode == "vary_density":
        configurations.sort(key=lambda row: row[CONFIG_SYNTH_DATA_DENSITY_KEY])
    else:
        configurations.sort(key=lambda row: (row["sweep_type"], row[CONFIG_SYNTH_DATA_NUM_NODES_KEY], row[CONFIG_SYNTH_DATA_DENSITY_KEY]))

    return configurations


def _append_best_configuration_columns(
    table: pd.DataFrame,
    best_runs: pd.DataFrame,
    hyperparameter: str,
) -> pd.DataFrame:
    if table.empty:
        result = table.copy()
        result["best_{}".format(hyperparameter)] = pd.Series(dtype=object)
        result["best_tune_id"] = pd.Series(dtype="Int64")
        result["best_avg_accuracy"] = pd.Series(dtype=float)
        return result

    result = table.copy()
    aligned_best_runs = best_runs.reindex(result.index)
    result["best_{}".format(hyperparameter)] = aligned_best_runs[hyperparameter]
    result["best_tune_id"] = aligned_best_runs["tune_id"]
    result["best_avg_accuracy"] = aligned_best_runs["avg_accuracy"]
    return result


def best_synthetic_tuning_runs(
    embedding_method: EMBEDDING_ALGORITHM,
    dataset: str,
    hyperparameters: Optional[Sequence[str]] = None,
    sweep_mode: SYNTHETIC_SWEEP_MODE = "both",
    num_nodes_list: Optional[Sequence[int]] = None,
    density_list: Optional[Sequence[float]] = None,
    fixed_num_nodes: int = SYNTH_DATA_EXPERIMENTS_DEFAULT_NUM_NODES,
    fixed_density: float = SYNTH_DATA_EXPERIMENTS_DEFAULT_DENSITY,
    raise_if_missing: bool = False,
) -> pd.DataFrame:
    resolved_hyperparameters = _normalize_hyperparameters(embedding_method, hyperparameters)
    tuning_results = load_synthetic_tuning_results(
        embedding_method=embedding_method,
        dataset=dataset,
        hyperparameters=resolved_hyperparameters,
        sweep_mode=sweep_mode,
        num_nodes_list=num_nodes_list,
        density_list=density_list,
        fixed_num_nodes=fixed_num_nodes,
        fixed_density=fixed_density,
        raise_if_missing=raise_if_missing,
    )

    if tuning_results.empty:
        return tuning_results

    best_idx = tuning_results.groupby("configuration_label")["avg_accuracy"].idxmax()
    result_columns = [
        "dataset",
        "dataset_label",
        "configuration_label",
        "num_nodes",
        "density",
        "sweep_type",
    ] + resolved_hyperparameters + ["avg_accuracy", "num_runs", "tune_id"]
    best_runs = tuning_results.loc[best_idx, result_columns].reset_index(drop=True)

    if sweep_mode == "vary_size":
        return best_runs.sort_values("num_nodes").reset_index(drop=True)
    if sweep_mode == "vary_density":
        return best_runs.sort_values("density").reset_index(drop=True)
    return best_runs.sort_values(["sweep_type", "num_nodes", "density"]).reset_index(drop=True)


def load_synthetic_tuning_results(
    embedding_method: EMBEDDING_ALGORITHM,
    dataset: str,
    hyperparameters: Optional[Sequence[str]] = None,
    sweep_mode: SYNTHETIC_SWEEP_MODE = "both",
    num_nodes_list: Optional[Sequence[int]] = None,
    density_list: Optional[Sequence[float]] = None,
    fixed_num_nodes: int = SYNTH_DATA_EXPERIMENTS_DEFAULT_NUM_NODES,
    fixed_density: float = SYNTH_DATA_EXPERIMENTS_DEFAULT_DENSITY,
    raise_if_missing: bool = False,
) -> pd.DataFrame:
    """Load synthetic tuning summaries for one synthetic dataset family.

    Rows are generated from the union of the canonical vary-size and vary-density sweeps.
    Each row corresponds to one graph configuration and one tuning setup, using the
    aggregate score already stored in the tuning summary.
    """

    if dataset not in SYNTHETIC_DATASET_LIST:
        raise ValueError(
            "Synthetic tuning analysis expects one synthetic dataset. "
            "Choose one of {}.".format(SYNTHETIC_DATASET_LIST)
        )

    resolved_hyperparameters = _normalize_hyperparameters(embedding_method, hyperparameters)
    configurations = _synthetic_sweep_configurations(
        sweep_mode=sweep_mode,
        num_nodes_list=num_nodes_list,
        density_list=density_list,
        fixed_num_nodes=fixed_num_nodes,
        fixed_density=fixed_density,
    )

    rows: List[dict[str, Any]] = []
    missing_paths: List[str] = []

    for configuration in configurations:
        dataset_params = {
            CONFIG_DATASET_NAME_KEY: dataset,
            CONFIG_DATA_SUBSAMPLING_INDICATOR_KEY: False,
            CONFIG_SYNTH_DATA_NUM_NODES_KEY: configuration[CONFIG_SYNTH_DATA_NUM_NODES_KEY],
            CONFIG_SYNTH_DATA_DENSITY_KEY: configuration[CONFIG_SYNTH_DATA_DENSITY_KEY],
            CONFIG_DATA_SAMPLING_SEED_KEY: 0,
        }
        tuning_summary_path = osp.join(
            CREATE_SYNTH_TUNING_RESULTS_PATH(
                dataset_params=dataset_params,
                embedding_name=embedding_method,
            ),
            TUNING_SUMMARY_FILE_NAME,
        )

        if not osp.isfile(tuning_summary_path):
            missing_paths.append(tuning_summary_path)
            continue

        with open(tuning_summary_path, "r") as file:
            tuning_summary = json.load(file)

        for tune_id, summary in tuning_summary.items():
            params = summary.get(TUNING_SUMMARY_PARAMS_KEY, {})
            results = summary.get(TUNING_SUMMARY_RESULTS_KEY, {})
            base_row = {
                "dataset": dataset,
                "dataset_label": DATASET_RENAME_DICT.get(dataset, dataset),
                "num_nodes": configuration[CONFIG_SYNTH_DATA_NUM_NODES_KEY],
                "density": configuration[CONFIG_SYNTH_DATA_DENSITY_KEY],
                "configuration_label": configuration["configuration_label"],
                "sweep_type": configuration["sweep_type"],
                "tune_id": int(tune_id),
                "num_runs": len(results),
                "avg_accuracy": summary.get(TUNING_SUMMARY_SCORE_KEY),
            }
            for hyperparameter in resolved_hyperparameters:
                base_row[hyperparameter] = params.get(hyperparameter)
            rows.append(base_row)

    if missing_paths and raise_if_missing:
        missing_lines = "\n".join(missing_paths)
        raise FileNotFoundError(f"Missing tuning summaries:\n{missing_lines}")

    if not rows:
        columns = [
            "dataset",
            "dataset_label",
            "num_nodes",
            "density",
            "configuration_label",
            "sweep_type",
            "avg_accuracy",
            "num_runs",
            "tune_id",
        ] + resolved_hyperparameters
        return pd.DataFrame(columns=columns)

    return (
        pd.DataFrame(rows)
        .sort_values(["sweep_type", "num_nodes", "density", "tune_id"])
        .reset_index(drop=True)
    )


def synthetic_tuning_accuracy_table(
    embedding_method: EMBEDDING_ALGORITHM,
    dataset: str,
    hyperparameters: Optional[Sequence[str]] = None,
    sweep_mode: SYNTHETIC_SWEEP_MODE = "both",
    num_nodes_list: Optional[Sequence[int]] = None,
    density_list: Optional[Sequence[float]] = None,
    fixed_num_nodes: int = SYNTH_DATA_EXPERIMENTS_DEFAULT_NUM_NODES,
    fixed_density: float = SYNTH_DATA_EXPERIMENTS_DEFAULT_DENSITY,
    raise_if_missing: bool = False,
) -> pd.DataFrame:
    """Build a synthetic tuning table for one synthetic dataset family.

    If one hyperparameter is requested, returns a pivoted table with one row per graph
    configuration and hyperparameter values as columns.
    Otherwise returns a flat dataframe with one row per configuration and tuning setup.
    """

    resolved_hyperparameters = _normalize_hyperparameters(embedding_method, hyperparameters)
    tuning_results = load_synthetic_tuning_results(
        embedding_method=embedding_method,
        dataset=dataset,
        hyperparameters=resolved_hyperparameters,
        sweep_mode=sweep_mode,
        num_nodes_list=num_nodes_list,
        density_list=density_list,
        fixed_num_nodes=fixed_num_nodes,
        fixed_density=fixed_density,
        raise_if_missing=raise_if_missing,
    )

    if len(resolved_hyperparameters) == 1:
        hyperparameter = resolved_hyperparameters[0]
        table = tuning_results.pivot(
            index="configuration_label",
            columns=hyperparameter,
            values="avg_accuracy",
        )
        best_runs = best_synthetic_tuning_runs(
            embedding_method=embedding_method,
            dataset=dataset,
            hyperparameters=resolved_hyperparameters,
            sweep_mode=sweep_mode,
            num_nodes_list=num_nodes_list,
            density_list=density_list,
            fixed_num_nodes=fixed_num_nodes,
            fixed_density=fixed_density,
            raise_if_missing=raise_if_missing,
        ).set_index("configuration_label")

        if sweep_mode == "vary_size":
            ordered_index = (
                tuning_results.loc[:, ["configuration_label", "num_nodes"]]
                .drop_duplicates()
                .sort_values("num_nodes")["configuration_label"]
            )
        elif sweep_mode == "vary_density":
            ordered_index = (
                tuning_results.loc[:, ["configuration_label", "density"]]
                .drop_duplicates()
                .sort_values("density")["configuration_label"]
            )
        else:
            ordered_index = (
                tuning_results.loc[:, ["configuration_label", "sweep_type", "num_nodes", "density"]]
                .drop_duplicates()
                .sort_values(["sweep_type", "num_nodes", "density"])["configuration_label"]
            )

        table = table.reindex(ordered_index).sort_index(axis=1)
        return _append_best_configuration_columns(table, best_runs, hyperparameter)

    result_columns = [
        "dataset",
        "dataset_label",
        "configuration_label",
        "num_nodes",
        "density",
        "sweep_type",
    ] + resolved_hyperparameters + ["avg_accuracy", "num_runs", "tune_id"]
    return tuning_results.loc[:, result_columns].reset_index(drop=True)
