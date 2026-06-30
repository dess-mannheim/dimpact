from __future__ import annotations

import argparse
import time
from argparse import Namespace
from typing import Sequence

from paths_globals import *

from min_ge.feature_entropy import closest_zero_graph_entropy_dimension


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(description="Compute the MinGE estimate for an empirical dataset.")
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        choices=EMPIRICAL_DATASET_LIST,
        default=None,
        help="Empirical dataset to analyze.",
    )
    parser.add_argument(
        "--implementation",
        type=str,
        choices=["dense", "sparse"],
        default="sparse",
        help="MinGE implementation to run.",
    )
    parser.add_argument("--table", action="store_true", help="Build the default MinGE approximation-vs-scan table.")
    parser.add_argument(
        "--lambda_dimension_table",
        action="store_true",
        help="Build a dataset-by-lambda table containing the scanned optimal MinGE dimensions.",
    )
    parser.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=[1.0],
        help="Lambda values used with --lambda_dimension_table.",
    )
    parser.add_argument("--scan_min_dim", type=int, default=2)
    parser.add_argument("--scan_max_dim", type=int, default=512)
    parser.add_argument("--lambda_", type=float, default=1.0)
    return parser.parse_args()


def _load_empirical_edge_index(dataset_name: DATASET) -> Tuple[np.ndarray, int]:
    if dataset_name not in EMPIRICAL_DATASET_LIST:
        raise ValueError(f"MinGE expects an empirical dataset, got {dataset_name}.")

    from tools import data_utils

    dataset_params = {
        CONFIG_DATASET_NAME_KEY: dataset_name,
        CONFIG_DATA_SUBSAMPLING_INDICATOR_KEY: False,
    }
    data, _ = data_utils.load_dataset(dataset_params)
    edge_index = data.edge_index.detach().cpu().numpy()
    if edge_index.shape[0] != 2:
        raise ValueError(f"Expected edge_index shape (2, n_edges), got {edge_index.shape}.")
    return edge_index.astype(np.int64), int(data.num_nodes)


def structure_entropy_from_normalized_degree(v_degree_n: np.ndarray) -> float:
    """Compute Hs from the normalized-degree vector Dr."""

    v_degree_n = np.asarray(v_degree_n, dtype=np.float64).reshape(-1)
    z = np.sum(v_degree_n)
    if z <= 0:
        raise ValueError("Cannot compute structure entropy for a graph with zero normalized degree mass.")

    p = v_degree_n / z
    p = p[p > 0]
    return float(-np.dot(p, np.log(p)))


def min_ge_dimension_from_structure_entropy(num_nodes: int, structure_entropy: float, lambda_: float = 1.0) -> float:
    """Released MinGE approximation using hn(n) ~= -0.24 * n."""

    return float((np.log(num_nodes * num_nodes) + lambda_ * structure_entropy) / 0.24)


def _entropy_sample_size(v_degree_n: np.ndarray, num_nodes: int) -> float:
    h = structure_entropy_from_normalized_degree(v_degree_n)
    return min_ge_dimension_from_structure_entropy(num_nodes=num_nodes, structure_entropy=h)


def _print_result(n: float, start: float | None = None) -> None:
    print(n)
    if start is not None:
        print("runing time: %s Second" % (time.time() - start))


def min_ge(dataset_name: DATASET) -> float:
    """Compute the dense MinGE estimate for an empirical dataset."""

    start = time.time()
    h, num_nodes = dense_structure_entropy(dataset_name, return_num_nodes=True)
    n = min_ge_dimension_from_structure_entropy(num_nodes=num_nodes, structure_entropy=h)
    _print_result(n, start)
    return n


def dense_structure_entropy(dataset_name: DATASET, return_num_nodes: bool = False) -> float | Tuple[float, int]:
    """Compute the dense MinGE structure entropy Hs for an empirical dataset."""

    edge_index, num_nodes = _load_empirical_edge_index(dataset_name)
    adj_mtrx = np.eye(num_nodes, dtype=np.float32)
    adj_mtrx[edge_index[0], edge_index[1]] = 1.0
    adj_mtrx[edge_index[1], edge_index[0]] = 1.0

    v_degree = np.sum(adj_mtrx, axis=0) + 1.0
    second_order = adj_mtrx.dot(adj_mtrx)
    row_sums = np.sum(second_order, axis=1)
    if np.any(row_sums == 0):
        raise ValueError("Cannot normalize second-order adjacency with zero row sums.")

    normalized_second_order = second_order / row_sums[:, None]
    v_degree_n = normalized_second_order.dot(v_degree)
    h = structure_entropy_from_normalized_degree(v_degree_n)
    if return_num_nodes:
        return h, num_nodes
    return h


def sparse_min_ge(dataset_name: DATASET) -> float:
    """Compute the sparse MinGE estimate for an empirical dataset."""

    h, num_nodes = sparse_structure_entropy(dataset_name, return_num_nodes=True)
    n = min_ge_dimension_from_structure_entropy(num_nodes=num_nodes, structure_entropy=h)
    _print_result(n)
    return n


def sparse_structure_entropy(dataset_name: DATASET, return_num_nodes: bool = False) -> float | Tuple[float, int]:
    """Compute the sparse MinGE structure entropy Hs for an empirical dataset."""

    from scipy import sparse

    edge_index, num_nodes = _load_empirical_edge_index(dataset_name)
    rows = np.concatenate([edge_index[0], edge_index[1], np.arange(num_nodes)])
    cols = np.concatenate([edge_index[1], edge_index[0], np.arange(num_nodes)])
    values = np.ones(rows.shape[0], dtype=np.float32)
    adj_mtrx = sparse.csr_matrix((values, (rows, cols)), shape=(num_nodes, num_nodes))
    adj_mtrx.data[:] = 1.0

    v_degree = np.asarray(adj_mtrx.sum(axis=0)).reshape(-1) + 1.0
    second_order = adj_mtrx.dot(adj_mtrx).tocsr()
    row_sums = np.asarray(second_order.sum(axis=1)).reshape(-1)
    if np.any(row_sums == 0):
        raise ValueError("Cannot normalize second-order adjacency with zero row sums.")

    inv_row_sums = sparse.diags(1.0 / row_sums)
    normalized_second_order = inv_row_sums.dot(second_order)
    v_degree_n = normalized_second_order.dot(v_degree)
    h = structure_entropy_from_normalized_degree(v_degree_n)
    if return_num_nodes:
        return h, num_nodes
    return h


def min_ge_comparison_table(
    datasets: list[DATASET] | None = None,
    scan_min_dim: int = 2,
    scan_max_dim: int = 512,
    lambda_: float = 1.0,
):
    import pandas as pd

    rows = []
    for dataset in EMPIRICAL_DATASET_LIST if datasets is None else datasets:
        structure_entropy, num_nodes = sparse_structure_entropy(dataset, return_num_nodes=True)
        approx_dim = min_ge_dimension_from_structure_entropy(
            num_nodes=num_nodes,
            structure_entropy=structure_entropy,
            lambda_=lambda_,
        )
        scan_result = closest_zero_graph_entropy_dimension(
            num_nodes=num_nodes,
            structure_entropy=structure_entropy,
            candidate_dimensions=range(scan_min_dim, scan_max_dim + 1),
            lambda_=lambda_,
        )
        rows.append(
            {
                "dataset": dataset,
                "num_nodes": num_nodes,
                "structure_entropy": structure_entropy,
                "min_ge_0_24": approx_dim,
                "min_ge_scan": int(scan_result["embedding_dim"]),
                "scan_graph_entropy": scan_result["graph_entropy"],
                "difference": scan_result["embedding_dim"] - approx_dim,
            }
        )
    return pd.DataFrame(rows)


def min_ge_lambda_dimension_table(
    lambdas: Sequence[float],
    datasets: list[DATASET] | None = None,
    scan_min_dim: int = 2,
    scan_max_dim: int = 512,
):
    import pandas as pd

    rows = []
    for dataset in EMPIRICAL_DATASET_LIST if datasets is None else datasets:
        structure_entropy, num_nodes = sparse_structure_entropy(dataset, return_num_nodes=True)
        row = {"dataset": dataset}
        for lambda_ in lambdas:
            scan_result = closest_zero_graph_entropy_dimension(
                num_nodes=num_nodes,
                structure_entropy=structure_entropy,
                candidate_dimensions=range(scan_min_dim, scan_max_dim + 1),
                lambda_=lambda_,
            )
            row[f"lambda={lambda_:g}"] = int(scan_result["embedding_dim"])
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if args.table:
        table = min_ge_comparison_table(
            scan_min_dim=args.scan_min_dim,
            scan_max_dim=args.scan_max_dim,
            lambda_=args.lambda_,
        )
        print(table.to_string(index=False))
        return

    if args.lambda_dimension_table:
        table = min_ge_lambda_dimension_table(
            lambdas=args.lambdas,
            scan_min_dim=args.scan_min_dim,
            scan_max_dim=args.scan_max_dim,
        )
        print(table.to_string(index=False))
        return

    if args.dataset is None:
        raise ValueError("--dataset is required unless --table is set.")

    if args.implementation == "dense":
        min_ge(args.dataset)
    else:
        sparse_min_ge(args.dataset)


if __name__ == "__main__":
    main()
