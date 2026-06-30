from __future__ import annotations

import argparse
from argparse import Namespace
from typing import Iterable

import numpy as np
from scipy.special import gammaln, ive


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the MinGE feature-entropy expression.")
    parser.add_argument("-n", "--embedding_dim", type=float, default=None)
    parser.add_argument("--num_nodes", type=int, default=2708)
    parser.add_argument("--structure_entropy", type=float, default=7.8)
    parser.add_argument("--lambda_", type=float, default=1.0)
    parser.add_argument("--scan_min_dim", type=int, default=None)
    parser.add_argument("--scan_max_dim", type=int, default=None)
    return parser.parse_args()


def feature_entropy_offset(embedding_dim: float) -> float:
    """Compute the dimension-dependent feature entropy term hn(n)."""

    n = float(embedding_dim)
    if n <= 1:
        raise ValueError("embedding_dim must be greater than 1 for this feature entropy expression.")

    order = (n - 2.0) / 2.0
    bessel_scaled = ive(order, n)
    if bessel_scaled <= 0 or not np.isfinite(bessel_scaled):
        raise ValueError(f"Could not evaluate scaled Bessel term for embedding_dim={embedding_dim}.")

    log_e1 = gammaln(n / 2.0) + n + (n - 2.0) / 2.0 * np.log(2.0 / n) + np.log(bessel_scaled)
    e2_over_e1 = n * ive(order + 1.0, n) / bessel_scaled
    return float(log_e1 - e2_over_e1)


def graph_entropy_from_terms(
    num_nodes: int, embedding_dim: float, structure_entropy: float, lambda_: float = 1.0
) -> float:
    """Compute Hg(n) = log(N^2) + hn(n) + lambda * Hs."""

    return float(np.log(num_nodes**2) + feature_entropy_offset(embedding_dim) + lambda_ * structure_entropy)


def approximate_min_ge_dimension(num_nodes: int, structure_entropy: float, lambda_: float = 1.0) -> float:
    """Released MinGE approximation using hn(n) ~= -0.24 * n."""

    return float((np.log(num_nodes**2) + lambda_ * structure_entropy) / 0.24)


def closest_zero_graph_entropy_dimension(
    num_nodes: int,
    structure_entropy: float,
    candidate_dimensions: Iterable[int],
    lambda_: float = 1.0,
) -> dict[str, float]:
    """Evaluate Hg(n) on candidate dimensions and return the one closest to zero."""

    best_dimension = None
    best_graph_entropy = None
    best_feature_offset = None

    for embedding_dim in candidate_dimensions:
        try:
            feature_offset = feature_entropy_offset(embedding_dim)
        except ValueError:
            continue
        graph_entropy = float(np.log(num_nodes**2) + feature_offset + lambda_ * structure_entropy)
        if not np.isfinite(graph_entropy):
            continue
        if best_graph_entropy is None or abs(graph_entropy) < abs(best_graph_entropy):
            best_dimension = embedding_dim
            best_graph_entropy = graph_entropy
            best_feature_offset = feature_offset

    if best_dimension is None:
        raise ValueError("candidate_dimensions must contain at least one valid value.")

    return {
        "embedding_dim": float(best_dimension),
        "graph_entropy": float(best_graph_entropy),
        "feature_entropy_offset": float(best_feature_offset),
    }


def main() -> None:
    args = parse_args()
    if args.embedding_dim is not None:
        hn = feature_entropy_offset(args.embedding_dim)
        h = graph_entropy_from_terms(
            num_nodes=args.num_nodes,
            embedding_dim=args.embedding_dim,
            structure_entropy=args.structure_entropy,
            lambda_=args.lambda_,
        )
        print("hn =", hn)
        print("H =", h)

    approx_dim = approximate_min_ge_dimension(
        num_nodes=args.num_nodes,
        structure_entropy=args.structure_entropy,
        lambda_=args.lambda_,
    )
    print("approx_min_ge_dimension =", approx_dim)

    if args.scan_min_dim is not None or args.scan_max_dim is not None:
        if args.scan_min_dim is None or args.scan_max_dim is None:
            raise ValueError("--scan_min_dim and --scan_max_dim must be provided together.")
        best = closest_zero_graph_entropy_dimension(
            num_nodes=args.num_nodes,
            structure_entropy=args.structure_entropy,
            candidate_dimensions=range(args.scan_min_dim, args.scan_max_dim + 1),
            lambda_=args.lambda_,
        )
        print("closest_zero_scan =", best)


if __name__ == "__main__":
    main()
