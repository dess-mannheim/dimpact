"""Visualization data crawlers and plotting helpers for plots_tables.ipynb.

This module centralizes notebook plotting logic so the notebook can stay focused on
exploration and rendering while implementation details live here.
"""

from __future__ import annotations

import csv
import __main__
import json
import re
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from matplotlib.patches import Rectangle

from paths_globals import *

try:
    from stability.measures.funcsim import StableCore
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise

    class StableCore:
        """Fallback used only for StableCore.__name__ plotting defaults."""

        pass


# ---------------------------------------------------------------------------
# Global Constants
# ---------------------------------------------------------------------------


EMPIRICAL_DATASET_PALETTE = sns.color_palette("Dark2", len(EMPIRICAL_DATASET_LIST))


EMPIRICAL_DATASET_COLORS = {
    dataset: EMPIRICAL_DATASET_PALETTE[i] for i, dataset in enumerate(EMPIRICAL_DATASET_LIST)
}

BOOTSTRAP_ALIGNMENT_FILE_STEM = "bootstrap_alignment"
BOOTSTRAP_HIT_RATE_PLOT_DIR_NAME = "bootstrap_hit_rates"
SYNTH_DATA_PLOT_DIR_NAME = "synth_data"
BOOTSTRAP_HIT_RATE_DEFAULT_OUTPUT_DIR = Path(PLOTS_DIR) / BOOTSTRAP_HIT_RATE_PLOT_DIR_NAME
HYPERPARAMETER_SENSITIVITY_GRID_FILE_STEM = "hyperparameter_sensitivity"
HYPERPARAMETER_SENSITIVITY_DEFAULT_OUTPUT_DIR = Path(PLOTS_DIR) / HYPERPARAMETER_SENSITIVITY_GRID_FILE_STEM

STABILITY_PERFORMANCE_BOOTSTRAP_RESULTS_DIR = Path(STABILITY_PERFORMANCE_BOOTSTRAP_CASE_STUDY_OUTPUT_DIR)
EMBEDDING_COST_RESULTS_DIRS = [
    Path(EMBEDDING_COSTS_CASE_STUDY_OUTPUT_DIR),
]
HYPERPARAMETER_SENSITIVITY_RESULTS_DIR = Path(HYPERPARAMETER_SENSITIVITY_CASE_STUDY_OUTPUT_DIR)
BOOTSTRAP_OPTIMUM_RELATION_MARKERS: Dict[str, Dict[str, Any]] = {
    "higher": {"marker": ">", "label": "Stability optimum higher"},
    "lower": {"marker": "<", "label": "Stability optimum lower"},
    "same": {"marker": "o", "label": "Same optimum"},
    "mixed": {"marker": "D", "label": "Stability optima on both sides"},
}

SYMLOG_TICK_BASE = 10.0
SYMLOG_LINEAR_TICK_DISTANCE_RATIO = 0.5
SYMLOG_DEFAULT_LINSCALE = SYMLOG_LINEAR_TICK_DISTANCE_RATIO * (1.0 - SYMLOG_TICK_BASE**-1)
CONFIDENCE_BAND_Z_VALUE = 1.96
CONFIDENCE_BAND_ALPHA = 0.18
CONFIDENCE_BAND_DEFAULT_MODE = "std"
CONFIDENCE_BAND_VALID_MODES = {"sem", "std", "2std", "quantile", "iqr"}

LINEPLOT_PAPER_STYLE_DEFAULT: Dict[str, Any] = {
    "max_cols": 3,
    "figure_width": 6.5,
    "golden_ratio": (1 + 5**0.5) / 2,
    "subplot_width": 2.4,
    "subplot_height": None,
    "row_height_pad": 0.55,
    "tick_label_size": 7,
    "axis_label_size": 8,
    "title_size": 8,
    "title_pad": 1.5,
    "title_y": None,
    "title_enumerate": True,
    "title_enum_start": "a",
    "legend_font_size": 7,
    "y_tick_step": 0.1,
    "x_tick_rotation": 45,
    "x_tick_ha": "right",
    "x_tick_rotation_mode": "anchor",
    "tick_direction": "out",
    "tick_length": 5.2,
    "tick_width": 0.9,
    "tick_label_pad": 1.5,
    "x_label_pad": 2.0,
    "y_label_pad": 5.0,
    "line_width": 1.35,
    "marker_size": 3.3,
    "line_alpha": 0.9,
    "grid_alpha": 0.35,
    "grid_color": "#9a9a9a",
    "text_color": "black",
    "spine_color": "black",
    "legend_ncol_max": 7,
    "legend_loc": "lower center",
    "legend_bbox_y": -0.006,
    "tight_layout_bottom": 0.10,
    "tight_layout_top": 0.90,
    "dpi": 300,
    "figure_facecolor": "white",
    "axes_facecolor": "white",
}


SYNTH_LINE_COLOR_PRESETS: Dict[str, Dict[str, Any]] = {
    # Recommended default: robust contrast without very pale lines.
    "cividis_mid_dark": {"palette": "cividis", "start": 0.30, "stop": 0.95},
    # Perceptually uniform, slightly more vivid than cividis.
    "viridis_mid_dark": {"palette": "viridis", "start": 0.30, "stop": 0.95},
    # Blue-centric sequential with clipped pale end.
    "blues_mid_dark": {"palette": "Blues", "start": 0.35, "stop": 0.95},
    # Warm sequential option.
    "magma_mid_dark": {"palette": "magma", "start": 0.25, "stop": 0.90},
    # High-contrast warm map, strong in print.
    "inferno_mid_dark": {"palette": "inferno", "start": 0.25, "stop": 0.90},
    # Grayscale-friendly option for B/W printouts.
    "greys_mid_dark": {"palette": "Greys", "start": 0.35, "stop": 0.92},
    # Smooth sequential with strong monotonic luminance.
    "cubehelix_mid_dark": {"palette": "cubehelix", "start": 0.30, "stop": 0.92},
    # Legacy single-hue tint behavior (adjustable with base_color).
    "single_hue_mid_dark": {"palette": "single_hue", "start": 0.22, "stop": 0.88},
}


PERFORMANCE_BEST_MARKER_STYLE: Dict[str, Any] = {
    "label": "Best-performance dimension",
    "marker": "*",
    "legend_color": "marker_color",
    "legend_markerfacecolor": "marker_color",
    "legend_markeredgecolor": "marker_color",
    "plot_color": "marker_color",
    "plot_edgecolor": "black",
    "plot_linewidth": 0.6,
    "plot_size_min": 44.0,
    "plot_size_offset": 1.9,
    "plot_zorder": 6,
}


PERFORMANCE_THRESHOLD_MARKER_STYLE: Dict[str, Any] = {
    "label": "Near-best-performance dimension",
    "marker": "o",
    "legend_color": "marker_color",
    "legend_markerfacecolor": "none",
    "legend_markeredgecolor": "marker_color",
    "plot_facecolors": "none",
    "plot_edgecolors": "black",
    "plot_linewidth": 1.2,
    "plot_size_min": 20.0,
    "plot_size_offset": 0.3,
    "plot_zorder": 6,
    "underlay_color": "marker_color",
    "underlay_size_min": 12.0,
    "underlay_zorder": 5.8,
}


PERFORMANCE_STATISTICAL_MARKER_STYLE: Dict[str, Any] = {
    **PERFORMANCE_THRESHOLD_MARKER_STYLE,
    "label": "Near-best performing dimension",
}


# ---------------------------------------------------------------------------
# Private Helpers
# ---------------------------------------------------------------------------


def _validate_axis_mode(value: str, param_name: str = "algorithm_axis") -> str:
    """Validate whether a subplot-axis selector is 'rows' or 'columns'."""
    valid = {"rows", "columns"}
    if value not in valid:
        raise ValueError(f"{param_name} must be one of {sorted(valid)}, got: {value!r}")
    return value


def _display_repsim_measure_name(raw_name: Any) -> str:
    """Resolve representational measure display name from rename dictionaries when available."""
    name = str(raw_name)
    local_map = globals().get("REPSIM_MEASURE_RENAME_DICT", None)
    if isinstance(local_map, dict) and name in local_map:
        return str(local_map[name])
    main_map = getattr(__main__, "REPSIM_MEASURE_RENAME_DICT", None)
    if isinstance(main_map, dict) and name in main_map:
        return str(main_map[name])
    return name


def _display_funcsim_measure_name(raw_name: Any) -> str:
    """Resolve functional measure display name from rename dictionaries when available."""
    name = str(raw_name)
    local_map = globals().get("FUNCSIM_MEASURE_RENAME_DICT", None)
    if isinstance(local_map, dict) and name in local_map:
        return str(local_map[name])
    main_map = getattr(__main__, "FUNCSIM_MEASURE_RENAME_DICT", None)
    if isinstance(main_map, dict) and name in main_map:
        return str(main_map[name])
    return name


def _apply_legend_frame(legend: Any) -> None:
    frame = legend.get_frame()
    frame.set_facecolor("white")
    frame.set_alpha(1.0)
    frame.set_edgecolor("#d0d0d0")
    frame.set_linewidth(0.8)


def _legend_text_color() -> str:
    label_color = plt.rcParams.get("legend.labelcolor", None)
    if isinstance(label_color, str) and label_color not in {
        "None",
        "none",
        "inherit",
        "linecolor",
        "markerfacecolor",
    }:
        return label_color
    return str(plt.rcParams.get("text.color", "black"))


def _set_legend_text_color(legend: Any, color: Any) -> None:
    for txt in legend.get_texts():
        txt.set_color(color)


def _blank_legend_handle() -> Any:
    return plt.Line2D([0], [0], linestyle="", alpha=0.0, marker="", label="")


def _compose_balanced_legend_handles(
    handles: List[Any],
    *,
    max_cols: int = 7,
) -> Tuple[List[Any], int, int]:
    """Return handles ordered for balanced visual rows under Matplotlib's column-major legend packing."""
    items = list(handles)
    if not items:
        return [], 1, 0

    max_cols_i = max(1, int(max_cols))
    num_items = len(items)
    num_rows = max(1, (num_items + max_cols_i - 1) // max_cols_i)
    num_cols = max(1, (num_items + num_rows - 1) // num_rows)

    base_row_len, extra = divmod(num_items, num_rows)
    row_lengths = [base_row_len + (1 if row_idx < extra else 0) for row_idx in range(num_rows)]

    rows: List[List[Any]] = []
    cursor = 0
    for row_len in row_lengths:
        row = items[cursor : cursor + row_len]
        cursor += row_len
        missing = num_cols - len(row)
        left = missing // 2
        right = missing - left
        rows.append(([_blank_legend_handle()] * left) + row + ([_blank_legend_handle()] * right))

    ordered: List[Any] = []
    for col_idx in range(num_cols):
        for row_idx in range(num_rows):
            ordered.append(rows[row_idx][col_idx])

    return ordered, num_cols, num_rows


def _draw_combined_legend_frame(
    fig: Any,
    legends: List[Any],
    *,
    pad: float = 0.004,
    edgecolor: str = "#d0d0d0",
    linewidth: float = 0.8,
) -> None:
    """Draw one frame around multiple legends in figure coordinates."""
    if not legends:
        return

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bboxes = [lg.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted()) for lg in legends]
    x0 = max(0.0, min(bb.x0 for bb in bboxes) - pad)
    y0 = max(0.0, min(bb.y0 for bb in bboxes) - pad)
    x1 = min(1.0, max(bb.x1 for bb in bboxes) + pad)
    y1 = min(1.0, max(bb.y1 for bb in bboxes) + pad)

    frame = Rectangle(
        (x0, y0),
        max(0.0, x1 - x0),
        max(0.0, y1 - y0),
        transform=fig.transFigure,
        facecolor="white",
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=2,
        clip_on=False,
    )
    fig.add_artist(frame)
    for lg in legends:
        lg.set_zorder(3)


def _dataset_color_map(datasets: List[str]) -> Dict[str, Any]:
    """Return deterministic colors for datasets with fixed empirical mappings."""
    dataset_colors = {ds: EMPIRICAL_DATASET_COLORS[ds] for ds in datasets if ds in EMPIRICAL_DATASET_COLORS}
    remaining = [ds for ds in datasets if ds not in dataset_colors]
    if remaining:
        fallback_palette = sns.color_palette("tab20", len(remaining))
        for i, dataset in enumerate(remaining):
            dataset_colors[dataset] = fallback_palette[i]
    return dataset_colors


def _dataset_display_sort_key(dataset: Any) -> str:
    return str(DATASET_RENAME_DICT.get(dataset, dataset)).lower()


def _optional_filter_set(values: Any) -> Optional[Set[str]]:
    if values is None:
        return None
    if isinstance(values, str):
        return {values}
    try:
        return {str(value) for value in values}
    except TypeError:
        return {str(values)}


def _algorithm_alpha_sort_key(algorithm: Any) -> str:
    algo = str(algorithm)
    return str(EMBEDDING_ALGORITHM_RENAME_DICT.get(algo, algo)).lower()


def _dataset_alpha_sort_key(dataset: Any) -> str:
    return _dataset_display_sort_key(dataset)


def _resolve_algorithm_datasets(
    algorithms: List[str],
    available_by_algorithm: Dict[str, List[str]],
    datasets: Any,
) -> Tuple[Dict[str, List[str]], List[str]]:
    """Resolve flat or per-algorithm dataset selectors into plot and legend datasets."""
    available_union = sorted(
        {dataset for algo in algorithms for dataset in available_by_algorithm.get(algo, [])},
        key=_dataset_display_sort_key,
    )

    if datasets is not None:
        try:
            dataset_items = [] if isinstance(datasets, str) else list(datasets)
        except TypeError:
            dataset_items = []
        is_nested_selector = any(
            isinstance(item, (list, tuple, set)) and not isinstance(item, str)
            for item in dataset_items
        )
    else:
        is_nested_selector = False

    if datasets is None:
        by_algorithm = {
            algo: sorted(available_by_algorithm.get(algo, []), key=_dataset_display_sort_key) for algo in algorithms
        }
    elif is_nested_selector:
        per_algorithm = list(datasets)
        if len(per_algorithm) != len(algorithms):
            raise ValueError(
                "When datasets is a list of lists, it must have one inner list per plotted algorithm. "
                f"Got {len(per_algorithm)} dataset lists for {len(algorithms)} algorithms: {algorithms}"
            )

        by_algorithm = {}
        for algo, selected in zip(algorithms, per_algorithm):
            if isinstance(selected, str) or not isinstance(selected, (list, tuple, set)):
                raise ValueError(
                    "When datasets is a list of lists, every item must be a dataset list. "
                    f"Invalid entry for algorithm {algo!r}: {selected!r}"
                )
            selected_list = list(selected)
            available = set(available_by_algorithm.get(algo, []))
            unknown = [dataset for dataset in selected_list if dataset not in available]
            if unknown:
                raise ValueError(
                    f"Datasets {unknown!r} are not available for algorithm {algo!r}. "
                    f"Available datasets: {sorted(available, key=_dataset_display_sort_key)}"
                )
            by_algorithm[algo] = sorted(selected_list, key=_dataset_display_sort_key)
    else:
        if isinstance(datasets, str):
            selected = [datasets]
        else:
            selected = list(datasets)
        selected_set = set(selected)
        by_algorithm = {
            algo: [
                dataset
                for dataset in sorted(available_by_algorithm.get(algo, []), key=_dataset_display_sort_key)
                if dataset in selected_set
            ]
            for algo in algorithms
        }

    legend_datasets = sorted(
        {dataset for algo in algorithms for dataset in by_algorithm.get(algo, [])},
        key=_dataset_display_sort_key,
    )
    if datasets is None and not legend_datasets:
        legend_datasets = available_union
    return by_algorithm, legend_datasets


def _apply_bounded_yaxis(ax: Any, y_max: float = 1.0) -> None:
    """Set bounded y-axis with 0.1-spaced ticks."""
    ymax = float(y_max)
    if ymax <= 0:
        ymax = 1.0
    ax.set_ylim(0.0, ymax)
    ax.set_yticks(np.arange(0.0, ymax + 0.001, 0.1))


def _functional_yaxis_max(measure: str) -> float:
    return 1.0 if str(measure).strip() == StableCore.__name__ else 0.6


def _resolve_lineplot_paper_style(style_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    style = dict(LINEPLOT_PAPER_STYLE_DEFAULT)
    if style_overrides:
        style.update(style_overrides)
    phi = float(style.get("golden_ratio", (1 + 5**0.5) / 2))
    if phi <= 0:
        phi = (1 + 5**0.5) / 2
    if style.get("subplot_height") in {None, 0}:
        style["subplot_height"] = float(style["subplot_width"]) / phi
    return style


def _subplot_enum_label(index: int, start: str = "a") -> str:
    base = ord("a")
    start_char = (str(start).strip().lower() or "a")[0]
    offset = max(0, ord(start_char) - base) + max(0, int(index))
    letters: List[str] = []
    while True:
        offset, remainder = divmod(offset, 26)
        letters.append(chr(base + remainder))
        if offset == 0:
            break
        offset -= 1
    return "".join(reversed(letters))


def _sanitize_filename_token(value: Any) -> str:
    token = re.sub(r"[^A-Za-z0-9]+", "_", str(value).strip()).strip("_").lower()
    return token or "unknown"


def _resolve_synth_line_colors(
    num_lines: int,
    *,
    color_scheme: str = "cividis_mid_dark",
    base_color: str = "#1f77b4",
) -> List[Any]:
    """Resolve ordered line colors for synthetic plots from named presets."""
    n = max(1, int(num_lines))
    preset = SYNTH_LINE_COLOR_PRESETS.get(str(color_scheme), SYNTH_LINE_COLOR_PRESETS["cividis_mid_dark"])
    start = float(preset.get("start", 0.30))
    stop = float(preset.get("stop", 0.95))
    if stop < start:
        start, stop = stop, start
    samples = np.linspace(start, stop, n)

    palette_name = str(preset.get("palette", "cividis"))
    if palette_name == "single_hue":
        cmap = sns.light_palette(base_color, as_cmap=True)
        colors = [cmap(float(v)) for v in samples]
    else:
        cmap = plt.get_cmap(palette_name)
        colors = [cmap(float(v)) for v in samples]

    def _luminance(rgba: Any) -> float:
        r, g, b = float(rgba[0]), float(rgba[1]), float(rgba[2])
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    # Ascending config values should map to light -> dark.
    return sorted(colors, key=_luminance, reverse=True)


def _resolve_performance_marker_style(marker_mode: str) -> Dict[str, Any]:
    marker_mode = _validate_performance_marker_mode(marker_mode)
    if marker_mode == "statistical":
        return dict(PERFORMANCE_STATISTICAL_MARKER_STYLE)
    return dict(PERFORMANCE_THRESHOLD_MARKER_STYLE)


def _resolve_marker_style_value(value: Any, marker_color: Any) -> Any:
    return marker_color if value == "marker_color" else value


def _overlay_performance_markers(
    ax: Any,
    dim_map: Dict[int, float],
    summary: Optional[Dict[str, Any]],
    *,
    color: Any,
    line_marker_size: float,
    marker_mode: str = "threshold",
) -> bool:
    """Overlay best and near-best downstream dimensions for one dataset curve."""
    if not summary:
        return False

    best_dim = summary.get("best_dim")
    plateau_dims = set(summary.get("plateau_dims", []))
    best_style = PERFORMANCE_BEST_MARKER_STYLE
    plateau_style = _resolve_performance_marker_style(marker_mode)
    has_marker = False

    if best_dim in dim_map:
        best_size = max(
            float(best_style.get("plot_size_min", 44.0)),
            (line_marker_size + float(best_style.get("plot_size_offset", 1.9))) ** 2,
        )
        ax.scatter(
            [best_dim],
            [dim_map[best_dim]],
            marker=str(best_style.get("marker", "*")),
            s=best_size,
            color=_resolve_marker_style_value(best_style.get("plot_color", "marker_color"), color),
            edgecolor=_resolve_marker_style_value(best_style.get("plot_edgecolor", "black"), color),
            linewidth=float(best_style.get("plot_linewidth", 0.6)),
            zorder=float(best_style.get("plot_zorder", 6)),
            clip_on=False,
        )
        has_marker = True

    for pdim in sorted(plateau_dims):
        if pdim == best_dim or pdim not in dim_map:
            continue
        # Repaint the marked curve's own point above overlapping curves before
        # drawing the hollow significance marker. Otherwise a transparent
        # marker can show whichever curve happened to be drawn last.
        underlay_size = max(float(plateau_style.get("underlay_size_min", 12.0)), line_marker_size**2)
        plateau_size = max(
            float(plateau_style.get("plot_size_min", 20.0)),
            (line_marker_size + float(plateau_style.get("plot_size_offset", 0.3))) ** 2,
        )
        ax.scatter(
            [pdim],
            [dim_map[pdim]],
            marker=str(plateau_style.get("marker", "o")),
            s=underlay_size,
            color=_resolve_marker_style_value(plateau_style.get("underlay_color", "marker_color"), color),
            edgecolors="none",
            linewidth=0.0,
            zorder=float(plateau_style.get("underlay_zorder", 5.8)),
            clip_on=False,
        )
        ax.scatter(
            [pdim],
            [dim_map[pdim]],
            marker=str(plateau_style.get("marker", "o")),
            s=plateau_size,
            facecolors=_resolve_marker_style_value(plateau_style.get("plot_facecolors", "none"), color),
            edgecolors=_resolve_marker_style_value(plateau_style.get("plot_edgecolors", "black"), color),
            linewidth=float(plateau_style.get("plot_linewidth", 1.2)),
            zorder=float(plateau_style.get("plot_zorder", 6)),
            clip_on=False,
        )
        has_marker = True

    return has_marker


def _sorted_finite_dim_points(dim_map: Dict[Any, Any]) -> List[Tuple[int, float]]:
    """Return sorted dimension/value pairs, excluding missing and non-finite values."""
    clean_by_dim: Dict[int, float] = {}
    for dim, score in dim_map.items():
        try:
            dim_i = int(dim)
            score_f = float(score)
        except (TypeError, ValueError):
            continue
        if np.isfinite(score_f):
            clean_by_dim[dim_i] = score_f
    return sorted(clean_by_dim.items(), key=lambda x: x[0])


def _split_dim_points_at_missing_dimensions(
    clean_points: List[Tuple[int, float]],
) -> List[List[Tuple[int, float]]]:
    """Break line segments where configured intermediate dimensions are absent."""
    if len(clean_points) <= 1:
        return [clean_points] if clean_points else []

    point_dims = {dim for dim, _ in clean_points}
    configured_dims = set()
    for dim in globals().get("EXPERIMENTS_DIMENSIONS_LIST", []):
        try:
            configured_dims.add(int(dim))
        except (TypeError, ValueError):
            continue
    expected_dims = sorted(configured_dims | point_dims)
    expected_idx = {dim: idx for idx, dim in enumerate(expected_dims)}

    segments: List[List[Tuple[int, float]]] = []
    current: List[Tuple[int, float]] = [clean_points[0]]
    for prev, curr in zip(clean_points, clean_points[1:]):
        prev_dim, curr_dim = prev[0], curr[0]
        missing_between = expected_idx[curr_dim] - expected_idx[prev_dim] > 1
        if missing_between:
            segments.append(current)
            current = [curr]
        else:
            current.append(curr)
    segments.append(current)
    return segments


def _plot_dimension_line(
    ax: Any,
    dim_map: Dict[Any, Any],
    *,
    marker: str = "o",
    linewidth: float = 2.0,
    markersize: Optional[float] = None,
    alpha: float = 0.9,
    label: Optional[str] = None,
    color: Any = None,
    clip_on: Optional[bool] = None,
    zorder: Optional[float] = None,
) -> Dict[int, float]:
    """Plot finite points, without drawing lines through missing configured dimensions."""
    clean_points = _sorted_finite_dim_points(dim_map)
    plot_kwargs = {
        "marker": marker,
        "linewidth": linewidth,
        "alpha": alpha,
        "label": label,
        "color": color,
    }
    if markersize is not None:
        plot_kwargs["markersize"] = markersize
    if clip_on is not None:
        plot_kwargs["clip_on"] = clip_on
    if zorder is not None:
        plot_kwargs["zorder"] = zorder

    first_segment = True
    for segment in _split_dim_points_at_missing_dimensions(clean_points):
        dims = [d for d, _ in segment]
        means = [m for _, m in segment]
        if not first_segment:
            plot_kwargs["label"] = None
        ax.plot(dims, means, **plot_kwargs)
        first_segment = False

    return dict(clean_points)


def _validate_confidence_band_mode(value: str) -> str:
    if value not in CONFIDENCE_BAND_VALID_MODES:
        raise ValueError(
            f"confidence_band_mode must be one of {sorted(CONFIDENCE_BAND_VALID_MODES)}, got: {value!r}"
        )
    return value


def _confidence_band_points(
    raw_dim_map: Dict[Any, Any],
    *,
    confidence_band_mode: str = CONFIDENCE_BAND_DEFAULT_MODE,
    z_value: float = CONFIDENCE_BAND_Z_VALUE,
    lower_bound_min: float = 0.0,
) -> List[Tuple[int, float, float]]:
    mode = _validate_confidence_band_mode(confidence_band_mode)
    points: List[Tuple[int, float, float]] = []
    for dim, values in raw_dim_map.items():
        try:
            dim_i = int(dim)
        except (TypeError, ValueError):
            continue

        vals = _sanitize_numeric_list(values)
        if not vals:
            continue

        arr = np.asarray(vals, dtype=float)
        mean = float(np.mean(arr))
        if mode == "quantile":
            lower, upper = (float(v) for v in np.percentile(arr, [2.5, 97.5]))
        elif mode == "iqr":
            lower, upper = (float(v) for v in np.percentile(arr, [25.0, 75.0]))
        else:
            if arr.size <= 1:
                spread = 0.0
            else:
                std = float(np.std(arr, ddof=1))
                if mode == "sem":
                    spread = float(z_value) * std / float(np.sqrt(arr.size))
                elif mode == "2std":
                    spread = 2.0 * std
                else:
                    spread = std
            lower = mean - spread
            upper = mean + spread
        lower = max(float(lower_bound_min), lower)
        upper = max(float(upper), lower)
        if np.isfinite(lower) and np.isfinite(upper):
            points.append((dim_i, lower, upper))

    return sorted(points, key=lambda item: item[0])


def _plot_confidence_band(
    ax: Any,
    raw_dim_map: Dict[Any, Any],
    *,
    color: Any,
    confidence_band_mode: str = CONFIDENCE_BAND_DEFAULT_MODE,
    alpha: float = CONFIDENCE_BAND_ALPHA,
    zorder: float = 1.0,
    lower_bound_min: float = 0.0,
) -> List[float]:
    """Draw an uncertainty/spread band and return finite bounds for y-limit calculations."""
    band_points = _confidence_band_points(
        raw_dim_map,
        confidence_band_mode=confidence_band_mode,
        lower_bound_min=lower_bound_min,
    )
    if not band_points:
        return []

    bound_values: List[float] = []
    mean_points = [(dim, (lower + upper) / 2.0) for dim, lower, upper in band_points]
    bounds_by_dim = {dim: (lower, upper) for dim, lower, upper in band_points}

    for segment in _split_dim_points_at_missing_dimensions(mean_points):
        dims = [dim for dim, _ in segment]
        lower = [bounds_by_dim[dim][0] for dim in dims]
        upper = [bounds_by_dim[dim][1] for dim in dims]
        ax.fill_between(dims, lower, upper, color=color, alpha=alpha, linewidth=0.0, zorder=zorder)
        bound_values.extend(lower)
        bound_values.extend(upper)

    return [float(v) for v in bound_values if np.isfinite(v)]


def _functional_embedding_raw_dim_map(
    functional_raw_results: Optional[dict],
    clf_name: str,
    algo: str,
    measure: str,
    dataset: str,
) -> Dict[int, List[float]]:
    if functional_raw_results is None:
        return {}
    grouped = functional_raw_results.get(clf_name, {}).get(algo, {}).get(measure, {}).get(dataset, {})
    return {
        int(dim): _sanitize_numeric_list(source_map.get("embedding", []))
        for dim, source_map in grouped.items()
        if isinstance(source_map, dict)
    }


def _validate_performance_marker_mode(value: str) -> str:
    valid = {"none", "threshold", "statistical"}
    if value not in valid:
        raise ValueError(f"performance_marker_mode must be one of {sorted(valid)}, got: {value!r}")
    return value


def _performance_marker_legend_handles(marker_mode: str = "threshold", marker_color: Any = "black") -> List[Any]:
    marker_mode = _validate_performance_marker_mode(marker_mode)
    if marker_mode == "none":
        return []
    best_style = PERFORMANCE_BEST_MARKER_STYLE
    plateau_style = _resolve_performance_marker_style(marker_mode)
    return [
        plt.Line2D(
            [0],
            [0],
            marker=str(best_style.get("marker", "*")),
            linestyle="",
            color=_resolve_marker_style_value(best_style.get("legend_color", "marker_color"), marker_color),
            markerfacecolor=_resolve_marker_style_value(
                best_style.get("legend_markerfacecolor", "marker_color"), marker_color
            ),
            markeredgecolor=_resolve_marker_style_value(
                best_style.get("legend_markeredgecolor", "marker_color"), marker_color
            ),
            label=str(best_style.get("label", "Best performance dim")),
        ),
        plt.Line2D(
            [0],
            [0],
            marker=str(plateau_style.get("marker", "o")),
            linestyle="",
            color=_resolve_marker_style_value(plateau_style.get("legend_color", "marker_color"), marker_color),
            markerfacecolor=_resolve_marker_style_value(
                plateau_style.get("legend_markerfacecolor", "none"), marker_color
            ),
            markeredgecolor=_resolve_marker_style_value(
                plateau_style.get("legend_markeredgecolor", "marker_color"), marker_color
            ),
            label=str(plateau_style.get("label", "Near-best performance dim")),
        ),
    ]


def _validate_legend_position(value: str) -> str:
    valid = {"bottom", "top", "none"}
    if value not in valid:
        raise ValueError(f"legend_position must be one of {sorted(valid)}, got: {value!r}")
    return value


def _add_overview_legend(
    fig: Any,
    legend_handles: List[Any],
    *,
    marker_handles: Optional[List[Any]] = None,
    legend_position: str = "bottom",
    max_cols: int = 7,
    top: float = 1.0,
) -> None:
    legend_position = _validate_legend_position(legend_position)
    marker_handles = list(marker_handles or [])
    if legend_position == "none" or (not legend_handles and not marker_handles):
        plt.tight_layout(rect=[0, 0, 1, top])
        return

    loc = "lower center" if legend_position == "bottom" else "upper center"
    bbox_y = 0.01 if legend_position == "bottom" else 0.995
    ordered_dataset_handles, dataset_ncol, dataset_rows = _compose_balanced_legend_handles(
        legend_handles, max_cols=max_cols
    )
    has_marker_row = bool(marker_handles)
    row_gap = 0.028
    layout_pad = 0.04 + 0.03 * max(1, dataset_rows) + (0.025 if has_marker_row else 0.0)
    layout_rect = (
        [0, layout_pad, 1, top] if legend_position == "bottom" else [0, 0, 1, min(top, max(0.0, 1.0 - layout_pad))]
    )
    text_color = _legend_text_color()

    legends: List[Any] = []
    dataset_bbox_y = bbox_y + row_gap if legend_position == "bottom" and has_marker_row else bbox_y
    if legend_position == "top" and has_marker_row:
        dataset_bbox_y = bbox_y - row_gap

    if ordered_dataset_handles:
        legend = fig.legend(
            handles=ordered_dataset_handles,
            loc=loc,
            bbox_to_anchor=(0.5, dataset_bbox_y),
            ncol=dataset_ncol,
            frameon=False if has_marker_row else True,
            fontsize=9,
            handlelength=1.8,
            columnspacing=1.2,
            borderaxespad=0.0,
        )
        _set_legend_text_color(legend, text_color)
        legends.append(legend)

    if marker_handles:
        marker_bbox_y = bbox_y if legend_position == "bottom" else bbox_y
        legend_markers = fig.legend(
            handles=marker_handles,
            loc=loc,
            bbox_to_anchor=(0.5, marker_bbox_y),
            ncol=len(marker_handles),
            frameon=False,
            fontsize=9,
            handlelength=1.8,
            columnspacing=1.2,
            borderaxespad=0.0,
        )
        _set_legend_text_color(legend_markers, text_color)
        legends.append(legend_markers)

    if has_marker_row:
        _draw_combined_legend_frame(fig, legends)
    elif legends:
        _apply_legend_frame(legends[0])

    plt.tight_layout(rect=layout_rect)


def _add_paper_legend(
    fig: Any,
    dataset_handles: List[Any],
    *,
    cfg: Dict[str, Any],
    marker_handles: Optional[List[Any]] = None,
) -> None:
    marker_handles = list(marker_handles or [])
    ordered_dataset_handles, dataset_ncol, dataset_rows = _compose_balanced_legend_handles(
        dataset_handles,
        max_cols=int(cfg["legend_ncol_max"]),
    )
    base_y = float(cfg["legend_bbox_y"])
    row_gap = 0.028
    text_color = str(cfg["text_color"])
    legends: List[Any] = []

    dataset_bbox_y = base_y + row_gap if marker_handles else base_y
    if ordered_dataset_handles:
        legend_datasets = fig.legend(
            handles=ordered_dataset_handles,
            loc=str(cfg["legend_loc"]),
            bbox_to_anchor=(0.5, dataset_bbox_y),
            ncol=dataset_ncol,
            frameon=False if marker_handles else True,
            fontsize=float(cfg["legend_font_size"]),
        )
        for txt in legend_datasets.get_texts():
            txt.set_color(text_color)
        legends.append(legend_datasets)

    if marker_handles:
        legend_markers = fig.legend(
            handles=marker_handles,
            loc=str(cfg["legend_loc"]),
            bbox_to_anchor=(0.5, base_y),
            ncol=len(marker_handles),
            frameon=False,
            fontsize=float(cfg["legend_font_size"]),
        )
        for txt in legend_markers.get_texts():
            txt.set_color(text_color)
        legends.append(legend_markers)

    if marker_handles:
        _draw_combined_legend_frame(fig, legends)
    elif legends:
        _apply_legend_frame(legends[0])

    extra_rows = max(0, dataset_rows - 1)
    if marker_handles or extra_rows:
        cfg["tight_layout_bottom"] = (
            float(cfg["tight_layout_bottom"]) + 0.028 * extra_rows + (0.02 if marker_handles else 0.0)
        )


def _build_marker_map_for_classifier(
    *,
    marker_mode: str,
    perf_peak_plateau_map: Optional[dict] = None,
    perf_results: Optional[dict] = None,
    classifier_name: Optional[str] = None,
    train_seed: int = EXPERIMENTS_DEFAULT_SEED,
    metric: str = ACCURACY_SCORE,
    relative_tolerance: Optional[float] = None,
    absolute_tolerance: float = 0.01,
    min_plateau_size: int = 1,
    alpha: float = 0.05,
) -> Optional[dict]:
    marker_mode = _validate_performance_marker_mode(marker_mode)
    if marker_mode == "none":
        return None
    if perf_peak_plateau_map is not None:
        return perf_peak_plateau_map
    if marker_mode == "threshold":
        return build_downstream_peak_plateau_map(
            perf_results=perf_results,
            train_seed=train_seed,
            classifier_name=classifier_name,
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
            min_plateau_size=min_plateau_size,
        )
    return build_downstream_peak_plateau_map_statistical(
        train_seed=train_seed,
        classifier_name=classifier_name,
        metric=metric,
        alpha=alpha,
    )


def _resolve_functional_marker_map_override(
    perf_peak_plateau_map: Optional[dict],
    *,
    classifier_name: str,
    available_classifiers: List[str],
    performance_marker_classifier_name: Optional[str] = None,
) -> Optional[dict]:
    """Use precomputed functional marker maps only when they match the functional classifier context."""
    if perf_peak_plateau_map is None:
        return None

    if performance_marker_classifier_name is not None:
        return perf_peak_plateau_map

    if classifier_name in perf_peak_plateau_map:
        return perf_peak_plateau_map[classifier_name]

    if any(clf in perf_peak_plateau_map for clf in available_classifiers):
        raise ValueError(
            f"perf_peak_plateau_map is keyed by classifier, but has no entry for functional classifier "
            f"{classifier_name!r}."
        )

    return None


def _resolve_manual_y_limits(
    y_axis_limits: Any,
    num_algorithms: int,
) -> List[Optional[Tuple[float, float]]]:
    """Normalize manual y-axis input to one (ymin, ymax) tuple per panel."""
    if y_axis_limits is None:
        raise ValueError("y_axis_limits must be provided when y_axis_mode='manual'.")

    if (
        isinstance(y_axis_limits, (tuple, list))
        and len(y_axis_limits) == 2
        and all(isinstance(v, (int, float)) for v in y_axis_limits)
    ):
        ymin, ymax = float(y_axis_limits[0]), float(y_axis_limits[1])
        return [(ymin, ymax)] * num_algorithms

    if not isinstance(y_axis_limits, list):
        raise ValueError("y_axis_limits must be either a (ymin, ymax) tuple or a list of (ymin, ymax) tuples.")

    if len(y_axis_limits) != num_algorithms:
        raise ValueError(f"Manual y_axis_limits list must have length {num_algorithms}, got {len(y_axis_limits)}.")

    limits_by_panel: List[Optional[Tuple[float, float]]] = []
    for idx, item in enumerate(y_axis_limits):
        if (
            not isinstance(item, (tuple, list))
            or len(item) != 2
            or not all(isinstance(v, (int, float)) for v in item)
        ):
            raise ValueError(f"Manual y_axis_limits entry {idx} must be a (ymin, ymax) tuple of numbers.")
        ymin, ymax = float(item[0]), float(item[1])
        limits_by_panel.append((ymin, ymax))

    return limits_by_panel


def _resolve_panel_y_limits(
    *,
    y_axis_mode: str,
    default_y_axis_max: float,
    y_axis_padding: float,
    all_plot_values: List[float],
    manual_y_limits_by_panel: Optional[List[Optional[Tuple[float, float]]]],
    panel_idx: int,
) -> Tuple[float, float]:
    """Resolve panel y-limits for fixed, zoom, or manual modes."""
    padding = max(0.0, float(y_axis_padding))
    hard_lower_bound = 0.0
    hard_upper_bound = float(default_y_axis_max)
    enforce_hard_bounds = hard_upper_bound <= 1.0 + 1e-12

    if y_axis_mode == "manual":
        if manual_y_limits_by_panel is None:
            raise ValueError("manual_y_limits_by_panel must be provided when y_axis_mode='manual'.")
        ymin, ymax = manual_y_limits_by_panel[panel_idx]
        if ymax <= ymin:
            raise ValueError(f"Invalid manual y-limits for panel {panel_idx}: {(ymin, ymax)!r}")
        return ymin, ymax

    if y_axis_mode == "zoom":
        clean_values = [float(v) for v in all_plot_values if np.isfinite(v)]
        if clean_values:
            ymin = min(clean_values) - padding
            ymax = max(clean_values) + padding

            # Keep canonical similarity boundaries visible when values are near full [0, 1].
            if min(clean_values) <= 0.02:
                ymin = min(ymin, 0.0)
            if max(clean_values) >= 0.98:
                ymax = max(ymax, 1.0)

            if ymax <= ymin:
                center = clean_values[0]
                ymin = center - max(padding, 0.01)
                ymax = center + max(padding, 0.01)

            # Snap to readable axis bounds to avoid awkward high-precision tick values.
            zoom_span = ymax - ymin
            snap_step = _nice_tick_step(zoom_span, target_ticks=8)
            if snap_step > 0:
                ymin = snap_step * np.floor((ymin - 1e-12) / snap_step)
                ymax = snap_step * np.ceil((ymax + 1e-12) / snap_step)
                if ymax <= ymin:
                    ymax = ymin + max(snap_step, 1e-3)

            if enforce_hard_bounds:
                ymin = max(hard_lower_bound, ymin)
                ymax = min(hard_upper_bound, ymax)
                if ymax <= ymin:
                    # Degenerate fallback under hard bounds: preserve small visible span.
                    span = max(1e-3, (hard_upper_bound - hard_lower_bound) / 20.0)
                    center = min(max(clean_values[0], hard_lower_bound), hard_upper_bound)
                    ymin = max(hard_lower_bound, center - span / 2.0)
                    ymax = min(hard_upper_bound, center + span / 2.0)
                    if ymax <= ymin:
                        ymin, ymax = hard_lower_bound, hard_upper_bound
            return ymin, ymax

    if enforce_hard_bounds:
        ymin = hard_lower_bound
        ymax = hard_upper_bound
    else:
        ymin = 0.0 - padding
        ymax = float(default_y_axis_max) + padding
    if ymax <= ymin:
        ymax = ymin + 1.0
    return ymin, ymax


def _validate_y_axis_scale(value: str) -> str:
    valid = {"linear", "log", "symlog"}
    if value not in valid:
        raise ValueError(f"y_axis_scale must be one of {sorted(valid)}, got: {value!r}")
    return value


def _validate_y_axis_symlog_linthresh(value: float) -> float:
    linthresh = float(value)
    if linthresh <= 0:
        raise ValueError(f"y_axis_symlog_linthresh must be > 0, got: {value!r}")
    return linthresh


def _resolve_symlog_linthresh_values(value: Any, algorithms: List[str]) -> List[float]:
    """Normalize symlog thresholds to one value per plotted algorithm."""
    if isinstance(value, dict):
        missing = [algo for algo in algorithms if algo not in value]
        if missing:
            raise ValueError(f"y_axis_symlog_linthresh is missing thresholds for algorithms: {missing}")
        return [_validate_y_axis_symlog_linthresh(value[algo]) for algo in algorithms]

    if isinstance(value, (list, tuple)):
        if len(value) != len(algorithms):
            raise ValueError(
                "y_axis_symlog_linthresh list must have one value per plotted algorithm "
                f"({len(algorithms)}), got {len(value)}."
            )
        return [_validate_y_axis_symlog_linthresh(item) for item in value]

    linthresh = _validate_y_axis_symlog_linthresh(value)
    return [linthresh] * len(algorithms)


def _validate_y_axis_symlog_linscale(value: float) -> float:
    linscale = float(value)
    if linscale <= 0:
        raise ValueError(f"y_axis_symlog_linscale must be > 0, got: {value!r}")
    return linscale


def _resolve_log_y_limits(ymin: float, ymax: float, values: List[float]) -> Tuple[float, float]:
    """Ensure y-limits are positive and ordered for log-scaled axes."""
    positive_values = [float(v) for v in values if np.isfinite(v) and float(v) > 0]
    min_positive = min(positive_values) if positive_values else 1e-6
    if ymin <= 0:
        ymin = min_positive * 0.8
    if ymax <= ymin:
        ymax = min_positive * 10.0
    return ymin, ymax


def _resolve_symlog_y_limits(
    ymin: float,
    ymax: float,
    *,
    default_y_axis_max: float = 1.0,
    y_axis_mode: str = "fixed",
) -> Tuple[float, float]:
    """Ensure symlog limits include zero and use 1.0 as the automatic upper default."""
    ymin = min(float(ymin), 0.0)
    if y_axis_mode != "manual":
        ymax = max(float(ymax), float(default_y_axis_max), 1.0)
    if ymax <= ymin:
        ymax = max(1.0, ymin + 1.0)
    return ymin, ymax


def _apply_symlog_yaxis(
    ax: Any,
    ymin: float,
    ymax: float,
    *,
    linthresh: float,
    linscale: float,
) -> None:
    ax.set_yscale("symlog", linthresh=linthresh, linscale=linscale, base=SYMLOG_TICK_BASE)
    ax.set_ylim(ymin, ymax)
    yticks: List[float] = []
    eps = 1e-12
    if ymin <= 0 <= ymax:
        yticks.append(0.0)

    tick = float(linthresh)
    while tick <= ymax * (1.0 + eps):
        if tick >= ymin * (1.0 - eps):
            yticks.append(tick)
        tick *= SYMLOG_TICK_BASE

    if ymax >= 1.0 and not any(np.isclose(tick, 1.0, rtol=1e-9, atol=1e-12) for tick in yticks):
        yticks.append(1.0)

    yticks = sorted(set(round(float(tick), 12) for tick in yticks if ymin - eps <= tick <= ymax + eps))
    tick_labels = []
    for tick in yticks:
        tick_f = float(tick)
        if np.isclose(tick_f, 0.0, atol=1e-15):
            tick_labels.append("0")
        else:
            label = f"{tick_f:.12f}".rstrip("0").rstrip(".")
            tick_labels.append(label if label not in {"-0", ""} else "0")
    ax.set_yticks(yticks)
    ax.set_yticklabels(tick_labels)


def _format_axis_ticklabels(values: List[float]) -> List[str]:
    """Format y-ticks with consistent, minimal decimals across all labels."""
    if not values:
        return []

    cleaned = []
    for value in values:
        v = float(value)
        if abs(v) < 5e-7:
            v = 0.0
        cleaned.append(v)

    arr = np.asarray(cleaned, dtype=float)
    decimals = 0
    for dec in range(0, 7):
        rounded = np.round(arr, dec)
        if np.all(np.isclose(arr, rounded, atol=10 ** (-(dec + 2)), rtol=0.0)):
            decimals = dec
            break
    else:
        decimals = 6

    labels: List[str] = []
    for value in np.round(arr, decimals):
        label = f"{float(value):.{decimals}f}".rstrip("0").rstrip(".")
        labels.append(label if label else "0")
    return labels


def _resolve_y_ticks(
    *,
    ymin: float,
    ymax: float,
    y_step: float,
    y_axis_mode: str,
) -> List[float]:
    """Build readable y-ticks with nice steps and total count constrained to 5..11."""
    if y_step <= 0:
        y_step = 0.1

    target_min_ticks = 5
    target_max_ticks = 11
    y0, y1 = float(ymin), float(ymax)
    span = y1 - y0
    if span <= 0:
        return [y0]

    def _build_ticks_with_step(step: float) -> List[float]:
        if step <= 0:
            return []
        start = step * np.ceil((y0 - 1e-12) / step)
        end = step * np.floor((y1 + 1e-12) / step)
        if end < start:
            return []
        count = int(np.floor((end - start) / step)) + 1
        return [float(start + i * step) for i in range(max(0, count))]

    def _ensure_reference_ticks(ticks: List[float]) -> List[float]:
        out = list(ticks)
        if y0 - 1e-12 <= 0.0 <= y1 + 1e-12 and not any(np.isclose(t, 0.0, atol=1e-12, rtol=0.0) for t in out):
            out.append(0.0)
        if y0 - 1e-12 <= 1.0 <= y1 + 1e-12 and not any(np.isclose(t, 1.0, atol=1e-12, rtol=0.0) for t in out):
            out.append(1.0)
        return sorted({float(np.round(t, 12)) for t in out})

    candidate_steps = sorted(
        {
            float(y_step),
            *[_nice_tick_step(span, target_ticks=n - 1) for n in range(target_min_ticks, target_max_ticks + 1)],
            *[_nice_tick_step(span, target_ticks=n) for n in range(target_min_ticks, target_max_ticks + 1)],
        }
    )
    candidate_steps = [s for s in candidate_steps if s > 0]

    candidates: List[Tuple[bool, int, float, List[float]]] = []
    for step in candidate_steps:
        ticks = _ensure_reference_ticks(_build_ticks_with_step(step))
        count = len(ticks)
        in_range = target_min_ticks <= count <= target_max_ticks
        candidates.append((in_range, count, step, ticks))

    selected: Optional[List[float]] = None
    valid = [c for c in candidates if c[0]]
    if valid:
        # Prefer denser readable grids (closer to 11 ticks), then larger step for cleaner labels.
        valid.sort(key=lambda x: (x[1], -x[2]), reverse=True)
        selected = valid[0][3]
    elif candidates:
        # Nearest feasible count around [5, 11].
        candidates.sort(key=lambda x: (min(abs(x[1] - target_min_ticks), abs(x[1] - target_max_ticks)), -x[1], -x[2]))
        selected = candidates[0][3]

    if not selected:
        selected = [float(t) for t in np.linspace(y0, y1, num=target_min_ticks)]

    # Final hard cap in case adding 0/1 exceeded limit; keep endpoints.
    if len(selected) > target_max_ticks:
        stride = int(np.ceil(len(selected) / target_max_ticks))
        selected = selected[::stride]
        if selected[-1] != max(selected):
            selected[-1] = max(selected)

    return sorted({float(np.round(t, 12)) for t in selected})


def _nice_tick_step(span: float, target_ticks: int = 5) -> float:
    """Choose a readable tick step for the given span."""
    clean_span = max(float(span), 1e-9)
    approx = clean_span / max(int(target_ticks), 1)
    exponent = np.floor(np.log10(approx))
    fraction = approx / (10**exponent)

    if fraction <= 1:
        nice_fraction = 1
    elif fraction <= 2:
        nice_fraction = 2
    elif fraction <= 2.5:
        nice_fraction = 2.5
    elif fraction <= 5:
        nice_fraction = 5
    else:
        nice_fraction = 10

    return float(nice_fraction * (10**exponent))


def _create_centered_algorithm_axes(num_algorithms: int, style: Dict[str, Any]) -> Tuple[Any, List[List[Any]]]:
    if num_algorithms <= 0:
        raise ValueError("num_algorithms must be > 0")

    max_cols = max(1, int(style["max_cols"]))
    n_rows = int(np.ceil(num_algorithms / max_cols))
    slot_cols = max_cols * 2
    subplot_w = float(style["subplot_width"])
    subplot_h = float(style["subplot_height"])
    row_height_pad = float(style["row_height_pad"])
    fig_width = float(style.get("figure_width", subplot_w * max_cols))
    if fig_width <= 0:
        fig_width = subplot_w * max_cols

    fig = plt.figure(figsize=(fig_width, subplot_h * n_rows + row_height_pad))
    fig.patch.set_facecolor(str(style["figure_facecolor"]))
    gs = fig.add_gridspec(n_rows, slot_cols)
    row_axes: List[List[Any]] = []

    for row_idx in range(n_rows):
        remaining = num_algorithms - row_idx * max_cols
        if remaining <= 0:
            break

        cols_in_row = min(max_cols, remaining)
        start_col = (slot_cols - cols_in_row * 2) // 2
        curr_row_axes: List[Any] = []
        for col_idx in range(cols_in_row):
            col_start = start_col + col_idx * 2
            ax = fig.add_subplot(gs[row_idx, col_start : col_start + 2])
            ax.set_facecolor(str(style["axes_facecolor"]))
            curr_row_axes.append(ax)
        row_axes.append(curr_row_axes)

    return fig, row_axes


def _plot_line_grid_by_algorithm_paper(
    algorithm_to_dataset_dim_map: Dict[str, Dict[str, Dict[int, float]]],
    *,
    datasets: Any,
    y_label: str,
    measure_name: str,
    classifier_name: Optional[str] = None,
    y_axis_max: float = 1.0,
    style: Optional[Dict[str, Any]] = None,
    output_dir: Any = PLOTS_DIR,
    show: bool = True,
    show_performance_markers: bool = False,
    perf_peak_plateau_map: Optional[dict] = None,
    performance_marker_mode: str = "threshold",
    y_axis_mode: str = "fixed",
    y_axis_limits: Any = None,
    y_axis_padding: float = 0.02,
    y_axis_scale: str = "linear",
    y_axis_symlog_linthresh: Any = 0.0001,
    y_axis_symlog_linscale: float = SYMLOG_DEFAULT_LINSCALE,
    raw_algorithm_to_dataset_dim_map: Optional[Dict[str, Dict[str, Dict[int, List[float]]]]] = None,
    include_confidence_bands: bool = False,
    confidence_band_mode: str = CONFIDENCE_BAND_DEFAULT_MODE,
) -> Optional[Path]:
    algorithms = sorted(algorithm_to_dataset_dim_map.keys())
    available_by_algorithm = {
        algo: sorted(algorithm_to_dataset_dim_map.get(algo, {}).keys(), key=_dataset_display_sort_key)
        for algo in algorithms
    }
    algorithm_datasets, legend_datasets = _resolve_algorithm_datasets(
        algorithms,
        available_by_algorithm,
        datasets,
    )
    if len(algorithms) == 0 or len(legend_datasets) == 0:
        print("No results found for the selected row.")
        return None

    valid_y_axis_modes = {"fixed", "zoom", "manual"}
    if y_axis_mode not in valid_y_axis_modes:
        raise ValueError(f"y_axis_mode must be one of {sorted(valid_y_axis_modes)}, got: {y_axis_mode!r}")
    y_axis_scale = _validate_y_axis_scale(y_axis_scale)
    y_axis_symlog_linthresh_values = _resolve_symlog_linthresh_values(y_axis_symlog_linthresh, algorithms)
    y_axis_symlog_linscale = _validate_y_axis_symlog_linscale(y_axis_symlog_linscale)
    confidence_band_mode = _validate_confidence_band_mode(confidence_band_mode)
    performance_marker_mode = _validate_performance_marker_mode(performance_marker_mode)
    show_performance_markers = bool(show_performance_markers and performance_marker_mode != "none")

    cfg = _resolve_lineplot_paper_style(style)
    dataset_colors = _dataset_color_map(legend_datasets)
    fig, row_axes = _create_centered_algorithm_axes(len(algorithms), cfg)
    manual_y_limits_by_panel = (
        _resolve_manual_y_limits(y_axis_limits, len(algorithms)) if y_axis_mode == "manual" else None
    )
    all_plot_values: List[float] = []
    for algo in algorithms:
        for dataset in algorithm_datasets.get(algo, []):
            dim_map = algorithm_to_dataset_dim_map.get(algo, {}).get(dataset, {})
            for score in dim_map.values():
                try:
                    score_f = float(score)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(score_f):
                    all_plot_values.append(score_f)
            if include_confidence_bands and raw_algorithm_to_dataset_dim_map is not None:
                for _, lower, upper in _confidence_band_points(
                    raw_algorithm_to_dataset_dim_map.get(algo, {}).get(dataset, {}),
                    confidence_band_mode=confidence_band_mode,
                    lower_bound_min=1e-12 if y_axis_scale == "log" else 0.0,
                ):
                    all_plot_values.extend([lower, upper])
    algo_idx = 0

    for axes_in_row in row_axes:
        for col_idx, ax in enumerate(axes_in_row):
            panel_idx = algo_idx
            algo = algorithms[algo_idx]
            algo_idx += 1
            dims_seen: Set[int] = set()
            has_data = False

            for dataset in algorithm_datasets.get(algo, []):
                dim_map = algorithm_to_dataset_dim_map.get(algo, {}).get(dataset, {})
                if not dim_map:
                    continue

                if include_confidence_bands and raw_algorithm_to_dataset_dim_map is not None:
                    _plot_confidence_band(
                        ax,
                        raw_algorithm_to_dataset_dim_map.get(algo, {}).get(dataset, {}),
                        color=dataset_colors[dataset],
                        confidence_band_mode=confidence_band_mode,
                        lower_bound_min=1e-12 if y_axis_scale == "log" else 0.0,
                        zorder=2.0,
                    )

                clean_dim_map = _plot_dimension_line(
                    ax,
                    dim_map,
                    marker="o",
                    linewidth=float(cfg["line_width"]),
                    markersize=float(cfg["marker_size"]),
                    alpha=float(cfg["line_alpha"]),
                    color=dataset_colors[dataset],
                    label=dataset,
                    clip_on=False,
                    zorder=4,
                )
                if not clean_dim_map:
                    continue

                dims = sorted(clean_dim_map.keys())
                dims_seen.update(dims)
                has_data = True

                if show_performance_markers:
                    _overlay_performance_markers(
                        ax,
                        clean_dim_map,
                        perf_peak_plateau_map.get(algo, {}).get(dataset, {}) if perf_peak_plateau_map else None,
                        color=dataset_colors[dataset],
                        line_marker_size=float(cfg["marker_size"]),
                        marker_mode=performance_marker_mode,
                    )

            display_algo = EMBEDDING_ALGORITHM_RENAME_DICT.get(algo, str(algo))
            if bool(cfg.get("title_enumerate", True)):
                enum = _subplot_enum_label(panel_idx, start=str(cfg.get("title_enum_start", "a")))
                title_text = f"({enum}) {display_algo}"
            else:
                title_text = str(display_algo)
            title_kwargs = {
                "fontsize": float(cfg["title_size"]),
                "color": str(cfg["text_color"]),
                "pad": float(cfg["title_pad"]),
            }
            title_y = cfg.get("title_y")
            if title_y is not None:
                title_kwargs["y"] = float(title_y)
            ax.set_title(str(title_text), **title_kwargs)
            ax.set_xscale("log", base=2)

            ymin, ymax = _resolve_panel_y_limits(
                y_axis_mode=y_axis_mode,
                default_y_axis_max=y_axis_max,
                y_axis_padding=y_axis_padding,
                all_plot_values=all_plot_values,
                manual_y_limits_by_panel=manual_y_limits_by_panel,
                panel_idx=panel_idx,
            )
            y_step = float(cfg["y_tick_step"])
            if y_step <= 0:
                y_step = 0.1
            if y_axis_mode == "zoom":
                y_step = _nice_tick_step(ymax - ymin)
            if y_axis_scale == "log":
                ymin, ymax = _resolve_log_y_limits(ymin, ymax, all_plot_values)
                ax.set_yscale("log")
            elif y_axis_scale == "symlog":
                panel_linthresh = y_axis_symlog_linthresh_values[panel_idx]
                ymin, ymax = _resolve_symlog_y_limits(
                    ymin,
                    ymax,
                    default_y_axis_max=y_axis_max,
                    y_axis_mode=y_axis_mode,
                )
                _apply_symlog_yaxis(
                    ax,
                    ymin,
                    ymax,
                    linthresh=panel_linthresh,
                    linscale=y_axis_symlog_linscale,
                )
            else:
                ax.set_yscale("linear")
                yticks = _resolve_y_ticks(
                    ymin=ymin,
                    ymax=ymax,
                    y_step=y_step,
                    y_axis_mode=y_axis_mode,
                )
                ax.set_yticks(yticks)
                ax.set_yticklabels(_format_axis_ticklabels(list(yticks)))
            if y_axis_scale != "symlog":
                ax.set_ylim(ymin, ymax)

            ax.grid(
                True,
                which="major",
                linestyle="--",
                alpha=float(cfg["grid_alpha"]),
                color=str(cfg["grid_color"]),
            )
            ax.tick_params(
                axis="both",
                labelsize=float(cfg["tick_label_size"]),
                colors=str(cfg["text_color"]),
                direction=str(cfg["tick_direction"]),
                length=float(cfg["tick_length"]),
                width=float(cfg["tick_width"]),
                pad=float(cfg["tick_label_pad"]),
                bottom=True,
                left=True,
                top=False,
                right=False,
            )
            ax.xaxis.set_ticks_position("bottom")
            ax.yaxis.set_ticks_position("left")
            for spine in ax.spines.values():
                spine.set_color(str(cfg["spine_color"]))

            if col_idx == 0:
                ax.set_ylabel(
                    y_label,
                    fontsize=float(cfg["axis_label_size"]),
                    color=str(cfg["text_color"]),
                    labelpad=float(cfg["y_label_pad"]),
                )
            else:
                ax.set_ylabel("")
            ax.set_xlabel(
                "Embedding Dimension",
                fontsize=float(cfg["axis_label_size"]),
                color=str(cfg["text_color"]),
                labelpad=float(cfg["x_label_pad"]),
            )

            if dims_seen:
                xticks = sorted(dims_seen)
                ax.set_xticks(xticks)
                ax.set_xticklabels(
                    [str(x) for x in xticks],
                    rotation=float(cfg["x_tick_rotation"]),
                    ha=str(cfg["x_tick_ha"]),
                    rotation_mode=str(cfg["x_tick_rotation_mode"]),
                )
            elif not has_data:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=str(cfg["text_color"]),
                )

    dataset_legend_handles = [
        plt.Line2D([0], [0], color=dataset_colors[d], label=DATASET_RENAME_DICT.get(d, d)) for d in legend_datasets
    ]
    marker_handles: List[Any] = []
    if show_performance_markers:
        marker_handles = _performance_marker_legend_handles(
            performance_marker_mode, marker_color=str(cfg["text_color"])
        )
    _add_paper_legend(fig, dataset_legend_handles, cfg=cfg, marker_handles=marker_handles)
    fig.tight_layout(rect=[0, float(cfg["tight_layout_bottom"]), 1, float(cfg["tight_layout_top"])])

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"lineplot_{_sanitize_filename_token(measure_name)}"
    if classifier_name:
        stem += f"_{_sanitize_filename_token(classifier_name)}"
    out_path = out_dir / f"{stem}.pdf"
    fig.savefig(
        out_path,
        format="pdf",
        bbox_inches="tight",
        dpi=int(cfg["dpi"]),
        facecolor=str(cfg["figure_facecolor"]),
        edgecolor="none",
        transparent=False,
    )

    if show:
        plt.show()
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path


def _parse_synth_parent_dir_name(dirname: str) -> Optional[Tuple[int, float]]:
    """Parse synthetic config directory name: graphs_n<num_nodes>_d<density_token>."""
    match = re.match(r"^graphs_n(?P<n>\d+)_d(?P<d>\d+)$", str(dirname))
    if not match:
        return None
    try:
        num_nodes = int(match.group("n"))
        density = float(f"0.{match.group('d')}")
    except (TypeError, ValueError):
        return None
    return num_nodes, density


def _resolve_density_key(
    density_map: Dict[float, Any], target_density: float, atol: float = 1e-12
) -> Optional[float]:
    """Return the matching density key (with tolerance) from a density->... mapping."""
    for density in sorted(density_map.keys()):
        if np.isclose(float(density), float(target_density), atol=float(atol), rtol=0.0):
            return float(density)
    return None


def _format_density_label(density: float) -> str:
    label = f"{float(density):.3f}".rstrip("0").rstrip(".")
    return label if label else "0"


def _identify_statistical_peak_and_plateau_dims(
    dim_to_values: Dict[int, List[float]],
    *,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Return best dimension and dimensions not significantly worse than best."""
    clean_values: Dict[int, np.ndarray] = {}
    for dim, values in dim_to_values.items():
        try:
            d = int(dim)
        except (TypeError, ValueError):
            continue
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        clean_values[d] = arr

    if not clean_values:
        return {
            "best_dim": None,
            "best_score": np.nan,
            "plateau_dims": [],
            "raw_pvalues": {},
            "corrected_pvalues": {},
            "alpha": float(alpha),
            "plateau_method": "welch_ttest_holm",
        }

    mean_by_dim = {dim: float(np.mean(vals)) for dim, vals in clean_values.items()}
    best_dim, best_score = max(mean_by_dim.items(), key=lambda item: item[1])
    best_values = clean_values[best_dim]

    raw_pvalues: Dict[int, float] = {}
    for dim in EXPERIMENTS_DIMENSIONS_LIST:
        if dim == best_dim or dim not in clean_values:
            continue
        other_values = clean_values[dim]
        if best_values.size < 2 or other_values.size < 2:
            raw_pvalues[dim] = np.nan
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            test_result = ttest_ind(best_values, other_values, equal_var=False, alternative="greater")
        pvalue = float(test_result.pvalue)
        if not np.isfinite(pvalue):
            best_mean = mean_by_dim[best_dim]
            other_mean = mean_by_dim[dim]
            pvalue = 1.0 if np.isclose(best_mean, other_mean) else 0.0
        raw_pvalues[dim] = pvalue

    finite_pvalues = {dim: pval for dim, pval in raw_pvalues.items() if np.isfinite(pval)}
    ordered_pvalues = sorted(finite_pvalues.items(), key=lambda item: item[1])
    num_pvalues = len(ordered_pvalues)
    corrected_pvalues: Dict[int, float] = {}
    running_max = 0.0
    for idx, (dim, pval) in enumerate(ordered_pvalues):
        adjusted = min(1.0, float(pval) * (num_pvalues - idx))
        running_max = max(running_max, adjusted)
        corrected_pvalues[dim] = running_max
    for dim, pval in raw_pvalues.items():
        if dim not in corrected_pvalues:
            corrected_pvalues[dim] = float(pval)

    plateau_dims = [best_dim]
    for dim in EXPERIMENTS_DIMENSIONS_LIST:
        if dim == best_dim or dim not in clean_values:
            continue
        corrected = corrected_pvalues.get(dim, np.nan)
        if np.isfinite(corrected) and corrected >= alpha:
            plateau_dims.append(dim)

    plateau_dims = sorted(set(plateau_dims))
    return {
        "best_dim": best_dim,
        "best_score": best_score,
        "plateau_dims": plateau_dims,
        "raw_pvalues": raw_pvalues,
        "corrected_pvalues": corrected_pvalues,
        "alpha": float(alpha),
        "plateau_method": "welch_ttest_holm",
    }


def _sanitize_numeric_list(values: Any) -> List[float]:
    arr = np.asarray(values, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    return [float(v) for v in arr.tolist()]


def _collect_numeric_recursive(obj: Any) -> List[float]:
    if isinstance(obj, dict):
        out = []
        for v in obj.values():
            out.extend(_collect_numeric_recursive(v))
        return out
    if isinstance(obj, (list, tuple)):
        out = []
        for v in obj:
            out.extend(_collect_numeric_recursive(v))
        return out
    try:
        x = float(obj)
    except (TypeError, ValueError):
        return []
    return [x] if np.isfinite(x) else []


def _ingest_functional_file(path: Path, source_key: str, storage: dict, algo: str, dataset: str) -> None:
    if not path.exists():
        return
    with open(path) as f:
        raw = json.load(f)

    for clf_name, measure_dict in raw.items():
        for measure, dim_dict in measure_dict.items():
            for dim, values in dim_dict.items():
                vals = _sanitize_numeric_list(_collect_numeric_recursive(values))
                if not vals:
                    continue
                dim_i = int(dim)
                slot = storage[clf_name][algo][measure][dataset].setdefault(dim_i, {})
                slot.setdefault(source_key, [])
                slot[source_key].extend(vals)


# ---------------------------------------------------------------------------
# Public Plotting API
# ---------------------------------------------------------------------------


def plot_representational_mean_lines_paper(
    results: Optional[dict] = None,
    measure_name: str = "",
    datasets: Any = None,
    style: Optional[Dict[str, Any]] = None,
    output_dir: Any = PLOTS_DIR,
    show: bool = True,
    show_performance_markers: bool = False,
    perf_peak_plateau_map: Optional[dict] = None,
    train_seed: int = EXPERIMENTS_DEFAULT_SEED,
    relative_tolerance: Optional[float] = None,
    absolute_tolerance: float = 0.01,
    min_plateau_size: int = 1,
    performance_marker_mode: str = "threshold",
    performance_marker_alpha: float = 0.05,
    performance_marker_classifier_name: Optional[str] = None,
    performance_marker_metric: str = ACCURACY_SCORE,
    y_axis_mode: str = "fixed",
    y_axis_limits: Any = None,
    y_axis_padding: float = 0.02,
    y_axis_scale: str = "linear",
    y_axis_symlog_linthresh: Any = 0.0001,
    y_axis_symlog_linscale: float = SYMLOG_DEFAULT_LINSCALE,
    include_confidence_bands: bool = False,
    confidence_band_mode: str = CONFIDENCE_BAND_DEFAULT_MODE,
) -> Optional[Path]:
    """Plot one representational-measure row as paper-ready algorithm panels and save as PDF."""
    if not str(measure_name).strip():
        raise ValueError("measure_name must be provided (e.g., 'Jaccard').")

    results = results or crawl_results()
    algorithms = sorted(results.keys())
    available_measures = sorted({m for algo_map in results.values() for m in algo_map.keys()})
    if measure_name not in available_measures:
        raise ValueError(f"Unknown measure_name {measure_name!r}. Available: {available_measures}")

    performance_marker_mode = _validate_performance_marker_mode(performance_marker_mode)
    show_performance_markers = bool(show_performance_markers and performance_marker_mode != "none")

    if show_performance_markers and perf_peak_plateau_map is None:
        if performance_marker_mode == "threshold":
            perf_peak_plateau_map = build_downstream_peak_plateau_map(
                train_seed=train_seed,
                classifier_name=performance_marker_classifier_name,
                relative_tolerance=relative_tolerance,
                absolute_tolerance=absolute_tolerance,
                min_plateau_size=min_plateau_size,
            )
        else:
            perf_peak_plateau_map = build_downstream_peak_plateau_map_statistical(
                train_seed=train_seed,
                classifier_name=performance_marker_classifier_name,
                metric=performance_marker_metric,
                alpha=performance_marker_alpha,
            )

    algorithm_to_dataset_dim_map = {algo: results.get(algo, {}).get(measure_name, {}) for algo in algorithms}
    raw_algorithm_to_dataset_dim_map = None
    if include_confidence_bands:
        raw_results = crawl_representational_raw_results()
        raw_algorithm_to_dataset_dim_map = {
            algo: raw_results.get(algo, {}).get(measure_name, {}) for algo in algorithms
        }
    return _plot_line_grid_by_algorithm_paper(
        algorithm_to_dataset_dim_map=algorithm_to_dataset_dim_map,
        datasets=datasets,
        y_label=_display_repsim_measure_name(measure_name),
        measure_name=measure_name,
        classifier_name=None,
        y_axis_max=1.0,
        style=style,
        output_dir=output_dir,
        show=show,
        show_performance_markers=show_performance_markers,
        perf_peak_plateau_map=perf_peak_plateau_map,
        performance_marker_mode=performance_marker_mode,
        y_axis_mode=y_axis_mode,
        y_axis_limits=y_axis_limits,
        y_axis_padding=y_axis_padding,
        y_axis_scale=y_axis_scale,
        y_axis_symlog_linthresh=y_axis_symlog_linthresh,
        y_axis_symlog_linscale=y_axis_symlog_linscale,
        raw_algorithm_to_dataset_dim_map=raw_algorithm_to_dataset_dim_map,
        include_confidence_bands=include_confidence_bands,
        confidence_band_mode=confidence_band_mode,
    )


def plot_downstream_mean_accuracy_lines_paper(
    perf_results: Optional[dict] = None,
    classifier_name: str = MULTILAYER_PERCEPTRON,
    train_seed: int = EXPERIMENTS_DEFAULT_SEED,
    datasets: Any = None,
    metric_name: str = ACCURACY_SCORE,
    style: Optional[Dict[str, Any]] = None,
    output_dir: Any = PLOTS_DIR,
    show: bool = True,
    show_performance_markers: bool = False,
    perf_peak_plateau_map: Optional[dict] = None,
    relative_tolerance: Optional[float] = None,
    absolute_tolerance: float = 0.01,
    min_plateau_size: int = 1,
    performance_marker_mode: str = "none",
    performance_marker_alpha: float = 0.05,
    y_axis_mode: str = "fixed",
    y_axis_limits: Any = None,
    y_axis_padding: float = 0.02,
    y_axis_scale: str = "linear",
    y_axis_symlog_linthresh: Any = 0.0001,
    y_axis_symlog_linscale: float = SYMLOG_DEFAULT_LINSCALE,
    include_confidence_bands: bool = False,
    confidence_band_mode: str = CONFIDENCE_BAND_DEFAULT_MODE,
) -> Optional[Path]:
    """Plot one downstream-classifier row as paper-ready algorithm panels and save as PDF."""
    perf_results = perf_results or crawl_downstream_accuracy_results(train_seed=train_seed, metric=metric_name)
    algorithms = sorted(perf_results.keys())

    available_classifiers = sorted({clf_name for algo_map in perf_results.values() for clf_name in algo_map.keys()})
    if classifier_name not in available_classifiers:
        raise ValueError(f"Unknown classifier_name {classifier_name!r}. Available: {available_classifiers}")

    performance_marker_mode = _validate_performance_marker_mode(performance_marker_mode)
    show_performance_markers = bool(show_performance_markers and performance_marker_mode != "none")

    if show_performance_markers and perf_peak_plateau_map is None:
        if performance_marker_mode == "threshold":
            perf_peak_plateau_map = build_downstream_peak_plateau_map(
                perf_results=perf_results,
                train_seed=train_seed,
                classifier_name=classifier_name,
                relative_tolerance=relative_tolerance,
                absolute_tolerance=absolute_tolerance,
                min_plateau_size=min_plateau_size,
            )
        else:
            perf_peak_plateau_map = build_downstream_peak_plateau_map_statistical(
                train_seed=train_seed,
                classifier_name=classifier_name,
                metric=metric_name,
                alpha=performance_marker_alpha,
            )

    algorithm_to_dataset_dim_map = {algo: perf_results.get(algo, {}).get(classifier_name, {}) for algo in algorithms}
    raw_algorithm_to_dataset_dim_map = None
    if include_confidence_bands:
        perf_raw_results = crawl_downstream_accuracy_raw_results(train_seed=train_seed, metric=metric_name)
        raw_algorithm_to_dataset_dim_map = {
            algo: perf_raw_results.get(algo, {}).get(classifier_name, {}) for algo in algorithms
        }
    return _plot_line_grid_by_algorithm_paper(
        algorithm_to_dataset_dim_map=algorithm_to_dataset_dim_map,
        datasets=datasets,
        y_label="Mean Accuracy",
        measure_name=metric_name,
        classifier_name=classifier_name,
        y_axis_max=1.0,
        style=style,
        output_dir=output_dir,
        show=show,
        show_performance_markers=show_performance_markers,
        perf_peak_plateau_map=perf_peak_plateau_map,
        performance_marker_mode=performance_marker_mode,
        y_axis_mode=y_axis_mode,
        y_axis_limits=y_axis_limits,
        y_axis_padding=y_axis_padding,
        y_axis_scale=y_axis_scale,
        y_axis_symlog_linthresh=y_axis_symlog_linthresh,
        y_axis_symlog_linscale=y_axis_symlog_linscale,
        raw_algorithm_to_dataset_dim_map=raw_algorithm_to_dataset_dim_map,
        include_confidence_bands=include_confidence_bands,
        confidence_band_mode=confidence_band_mode,
    )


def plot_functional_mean_lines_paper(
    functional_results: Optional[dict] = None,
    classifier_name: str = MULTILAYER_PERCEPTRON,
    measure_name: str = StableCore.__name__,
    datasets: Any = None,
    style: Optional[Dict[str, Any]] = None,
    output_dir: Any = PLOTS_DIR,
    show: bool = True,
    show_performance_markers: bool = False,
    perf_peak_plateau_map: Optional[dict] = None,
    train_seed: int = EXPERIMENTS_DEFAULT_SEED,
    relative_tolerance: Optional[float] = None,
    absolute_tolerance: float = 0.01,
    min_plateau_size: int = 1,
    performance_marker_mode: str = "threshold",
    performance_marker_alpha: float = 0.05,
    performance_marker_classifier_name: Optional[str] = None,
    performance_marker_metric: str = ACCURACY_SCORE,
    y_axis_mode: str = "fixed",
    y_axis_limits: Any = None,
    y_axis_padding: float = 0.02,
    y_axis_scale: str = "linear",
    y_axis_symlog_linthresh: Any = 0.0001,
    y_axis_symlog_linscale: float = SYMLOG_DEFAULT_LINSCALE,
    include_confidence_bands: bool = False,
    confidence_band_mode: str = CONFIDENCE_BAND_DEFAULT_MODE,
) -> Optional[Path]:
    """Plot one functional row (classifier + measure) as paper-ready algorithm panels and save as PDF."""
    functional_results = functional_results or crawl_functional_results()

    available_classifiers = sorted(functional_results.keys())
    if classifier_name not in available_classifiers:
        raise ValueError(f"Unknown classifier_name {classifier_name!r}. Available: {available_classifiers}")

    algorithms = sorted(functional_results[classifier_name].keys())
    available_measures = sorted(
        {measure for algo_map in functional_results[classifier_name].values() for measure in algo_map.keys()}
    )
    if measure_name not in available_measures:
        raise ValueError(f"Unknown measure_name {measure_name!r}. Available: {available_measures}")

    performance_marker_mode = _validate_performance_marker_mode(performance_marker_mode)
    show_performance_markers = bool(show_performance_markers and performance_marker_mode != "none")

    functional_perf_peak_plateau_map = _resolve_functional_marker_map_override(
        perf_peak_plateau_map,
        classifier_name=classifier_name,
        available_classifiers=available_classifiers,
        performance_marker_classifier_name=performance_marker_classifier_name,
    )

    if show_performance_markers and functional_perf_peak_plateau_map is None:
        marker_classifier_name = performance_marker_classifier_name or classifier_name
        if performance_marker_mode == "threshold":
            functional_perf_peak_plateau_map = build_downstream_peak_plateau_map(
                train_seed=train_seed,
                classifier_name=marker_classifier_name,
                relative_tolerance=relative_tolerance,
                absolute_tolerance=absolute_tolerance,
                min_plateau_size=min_plateau_size,
            )
        else:
            functional_perf_peak_plateau_map = build_downstream_peak_plateau_map_statistical(
                train_seed=train_seed,
                classifier_name=marker_classifier_name,
                metric=performance_marker_metric,
                alpha=performance_marker_alpha,
            )

    algorithm_to_dataset_dim_map = {
        algo: functional_results[classifier_name].get(algo, {}).get(measure_name, {}) for algo in algorithms
    }
    raw_algorithm_to_dataset_dim_map = None
    if include_confidence_bands:
        functional_raw_results = crawl_functional_grouped_raw_results()
        raw_algorithm_to_dataset_dim_map = {
            algo: {
                dataset: _functional_embedding_raw_dim_map(
                    functional_raw_results,
                    classifier_name,
                    algo,
                    measure_name,
                    dataset,
                )
                for dataset in functional_results[classifier_name].get(algo, {}).get(measure_name, {}).keys()
            }
            for algo in algorithms
        }
    return _plot_line_grid_by_algorithm_paper(
        algorithm_to_dataset_dim_map=algorithm_to_dataset_dim_map,
        datasets=datasets,
        y_label=_display_funcsim_measure_name(measure_name),
        measure_name=measure_name,
        classifier_name=classifier_name,
        y_axis_max=1.0,
        style=style,
        output_dir=output_dir,
        show=show,
        show_performance_markers=show_performance_markers,
        perf_peak_plateau_map=functional_perf_peak_plateau_map,
        performance_marker_mode=performance_marker_mode,
        y_axis_mode=y_axis_mode,
        y_axis_limits=y_axis_limits,
        y_axis_padding=y_axis_padding,
        y_axis_scale=y_axis_scale,
        y_axis_symlog_linthresh=y_axis_symlog_linthresh,
        y_axis_symlog_linscale=y_axis_symlog_linscale,
        raw_algorithm_to_dataset_dim_map=raw_algorithm_to_dataset_dim_map,
        include_confidence_bands=include_confidence_bands,
        confidence_band_mode=confidence_band_mode,
    )


def plot_synth_representational_mean_lines_paper(
    synth_results: Optional[dict] = None,
    dataset: DATASET = WATTS_STROGATZ,
    measure_name: str = "JaccardSimilarity",
    vary_size: bool = True,
    vary_density: bool = False,
    fixed_num_nodes: int = SYNTH_DATA_EXPERIMENTS_DEFAULT_NUM_NODES,
    fixed_density: float = SYNTH_DATA_EXPERIMENTS_DEFAULT_DENSITY,
    algorithms: Optional[List[EMBEDDING_ALGORITHM]] = None,
    color_scheme: str = "cividis_mid_dark",
    base_color: str = "#1f77b4",
    style: Optional[Dict[str, Any]] = None,
    output_dir: Any = PLOTS_DIR,
    show: bool = True,
    y_axis_mode: str = "fixed",
    y_axis_limits: Any = None,
    y_axis_padding: float = 0.02,
    include_confidence_bands: bool = False,
    confidence_band_mode: str = CONFIDENCE_BAND_DEFAULT_MODE,
) -> Optional[Path]:
    """Plot synthetic representational stability over dimensions, with tonal lines for size/density configs."""
    if bool(vary_size) == bool(vary_density):
        raise ValueError("Set exactly one of vary_size=True or vary_density=True.")
    if not str(measure_name).strip():
        raise ValueError("measure_name must be provided (e.g., 'JaccardSimilarity').")
    confidence_band_mode = _validate_confidence_band_mode(confidence_band_mode)

    synth_results = synth_results or crawl_synth_results()
    synth_raw_results = crawl_synth_raw_results() if include_confidence_bands else None
    vary_mode = "size" if vary_size else "density"

    available_algorithms = sorted(
        [algo for algo in synth_results.keys() if dataset in synth_results.get(algo, {}).get(measure_name, {})]
    )
    if len(available_algorithms) == 0:
        print(f"No synthetic representational results found for dataset={dataset!r}, measure={measure_name!r}.")
        return None

    if algorithms is None:
        selected_algorithms = available_algorithms[:5]
    else:
        selected_algorithms = [algo for algo in algorithms if algo in available_algorithms]

    if len(selected_algorithms) == 0:
        print("No selected algorithms have data for the requested synthetic setup.")
        return None

    def _curve_map_for_algo(algo: str) -> Dict[float, Dict[int, float]]:
        by_nodes = synth_results.get(algo, {}).get(measure_name, {}).get(dataset, {})
        if vary_mode == "size":
            curves = {}
            for num_nodes, density_map in by_nodes.items():
                density_key = _resolve_density_key(density_map, fixed_density)
                if density_key is None:
                    continue
                curves[float(int(num_nodes))] = density_map[density_key]
            return curves

        density_map = by_nodes.get(int(fixed_num_nodes), {})
        return {float(d): dim_map for d, dim_map in density_map.items()}

    def _raw_curve_map_for_algo(algo: str) -> Dict[float, Dict[int, List[float]]]:
        if synth_raw_results is None:
            return {}
        by_nodes = synth_raw_results.get(algo, {}).get(measure_name, {}).get(dataset, {})
        if vary_mode == "size":
            curves = {}
            for num_nodes, density_map in by_nodes.items():
                density_key = _resolve_density_key(density_map, fixed_density)
                if density_key is None:
                    continue
                curves[float(int(num_nodes))] = density_map[density_key]
            return curves

        density_map = by_nodes.get(int(fixed_num_nodes), {})
        return {float(d): dim_map for d, dim_map in density_map.items()}

    legend_values = sorted({v for algo in selected_algorithms for v in _curve_map_for_algo(algo).keys()})
    if len(legend_values) == 0:
        print("No matching synthetic configurations found for the requested vary/fixed parameters.")
        return None

    if str(color_scheme) not in SYNTH_LINE_COLOR_PRESETS:
        available = sorted(SYNTH_LINE_COLOR_PRESETS.keys())
        raise ValueError(f"Unknown color_scheme {color_scheme!r}. Available presets: {available}")

    tone_colors = _resolve_synth_line_colors(
        len(legend_values),
        color_scheme=str(color_scheme),
        base_color=base_color,
    )
    color_by_value = {legend_values[i]: tone_colors[i] for i in range(len(legend_values))}

    valid_y_axis_modes = {"fixed", "zoom", "manual"}
    if y_axis_mode not in valid_y_axis_modes:
        raise ValueError(f"y_axis_mode must be one of {sorted(valid_y_axis_modes)}, got: {y_axis_mode!r}")

    cfg = _resolve_lineplot_paper_style(style)
    fig, row_axes = _create_centered_algorithm_axes(len(selected_algorithms), cfg)
    manual_y_limits_by_panel = (
        _resolve_manual_y_limits(y_axis_limits, len(selected_algorithms)) if y_axis_mode == "manual" else None
    )

    all_plot_values: List[float] = []
    for algo in selected_algorithms:
        for dim_map in _curve_map_for_algo(algo).values():
            for score in dim_map.values():
                try:
                    score_f = float(score)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(score_f):
                    all_plot_values.append(score_f)
        if include_confidence_bands:
            for raw_dim_map in _raw_curve_map_for_algo(algo).values():
                for _, lower, upper in _confidence_band_points(
                    raw_dim_map,
                    confidence_band_mode=confidence_band_mode,
                ):
                    all_plot_values.extend([lower, upper])

    algo_idx = 0
    for axes_in_row in row_axes:
        for _, ax in enumerate(axes_in_row):
            panel_idx = algo_idx
            algo = selected_algorithms[algo_idx]
            algo_idx += 1

            curve_map = _curve_map_for_algo(algo)
            raw_curve_map = _raw_curve_map_for_algo(algo)
            dims_seen: Set[int] = set()
            has_data = False
            panel_plot_values: List[float] = []

            for curve_value in legend_values:
                dim_map = curve_map.get(curve_value, {})
                if not dim_map:
                    continue

                if include_confidence_bands:
                    panel_plot_values.extend(
                        _plot_confidence_band(
                            ax,
                            raw_curve_map.get(curve_value, {}),
                            color=color_by_value[curve_value],
                            confidence_band_mode=confidence_band_mode,
                            zorder=2.0,
                        )
                    )

                clean_dim_map = _plot_dimension_line(
                    ax,
                    dim_map,
                    marker="o",
                    linewidth=float(cfg["line_width"]),
                    markersize=float(cfg["marker_size"]),
                    alpha=float(cfg["line_alpha"]),
                    color=color_by_value[curve_value],
                    clip_on=False,
                    zorder=4,
                )
                if not clean_dim_map:
                    continue

                dims = sorted(clean_dim_map.keys())
                dims_seen.update(dims)
                panel_plot_values.extend(clean_dim_map.values())
                has_data = True

            display_algo = EMBEDDING_ALGORITHM_RENAME_DICT.get(algo, str(algo))
            if bool(cfg.get("title_enumerate", True)):
                enum = _subplot_enum_label(panel_idx, start=str(cfg.get("title_enum_start", "a")))
                title_text = f"({enum}) {display_algo}"
            else:
                title_text = str(display_algo)
            title_kwargs = {
                "fontsize": float(cfg["title_size"]),
                "color": str(cfg["text_color"]),
                "pad": float(cfg["title_pad"]),
            }
            title_y = cfg.get("title_y")
            if title_y is not None:
                title_kwargs["y"] = float(title_y)
            ax.set_title(str(title_text), **title_kwargs)
            ax.set_xscale("log", base=2)

            ymin, ymax = _resolve_panel_y_limits(
                y_axis_mode=y_axis_mode,
                default_y_axis_max=1.0,
                y_axis_padding=y_axis_padding,
                all_plot_values=panel_plot_values if y_axis_mode == "zoom" else all_plot_values,
                manual_y_limits_by_panel=manual_y_limits_by_panel,
                panel_idx=panel_idx,
            )
            y_step = float(cfg["y_tick_step"])
            if y_step <= 0:
                y_step = 0.1
            if y_axis_mode == "zoom":
                y_step = _nice_tick_step(ymax - ymin)
            yticks = _resolve_y_ticks(
                ymin=ymin,
                ymax=ymax,
                y_step=y_step,
                y_axis_mode=y_axis_mode,
            )
            ax.set_ylim(ymin, ymax)
            ax.set_yticks(yticks)
            ax.set_yticklabels(
                _format_axis_ticklabels(yticks),
                fontsize=float(cfg["tick_label_size"]),
                color=str(cfg["text_color"]),
            )
            ax.grid(True, linestyle="--", alpha=float(cfg["grid_alpha"]), color=str(cfg["grid_color"]))

            for spine in ax.spines.values():
                spine.set_color(str(cfg["spine_color"]))
                spine.set_zorder(1)
                spine.set_linewidth(float(cfg["tick_width"]))

            ax.tick_params(
                axis="x",
                direction=str(cfg["tick_direction"]),
                length=float(cfg["tick_length"]),
                width=float(cfg["tick_width"]),
                colors=str(cfg["text_color"]),
                labelsize=float(cfg["tick_label_size"]),
                pad=float(cfg["tick_label_pad"]),
                bottom=True,
                left=False,
                top=False,
                right=False,
            )
            ax.tick_params(
                axis="y",
                direction=str(cfg["tick_direction"]),
                length=float(cfg["tick_length"]),
                width=float(cfg["tick_width"]),
                colors=str(cfg["text_color"]),
                labelsize=float(cfg["tick_label_size"]),
                pad=float(cfg["tick_label_pad"]),
                bottom=False,
                left=True,
                top=False,
                right=False,
            )
            ax.xaxis.set_ticks_position("bottom")
            ax.yaxis.set_ticks_position("left")

            if panel_idx % int(cfg["max_cols"]) == 0:
                ax.set_ylabel(
                    _display_repsim_measure_name(measure_name),
                    fontsize=float(cfg["axis_label_size"]),
                    color=str(cfg["text_color"]),
                )
            else:
                ax.set_ylabel("")

            ax.set_xlabel(
                "Embedding Dimension",
                fontsize=float(cfg["axis_label_size"]),
                color=str(cfg["text_color"]),
                labelpad=float(cfg["x_label_pad"]),
            )

            if dims_seen:
                xticks = sorted(dims_seen)
                ax.set_xticks(xticks)
                ax.set_xticklabels(
                    [str(x) for x in xticks],
                    rotation=float(cfg["x_tick_rotation"]),
                    ha=str(cfg["x_tick_ha"]),
                    rotation_mode=str(cfg["x_tick_rotation_mode"]),
                )
            elif not has_data:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=str(cfg["text_color"]),
                )

    if vary_mode == "size":
        legend_handles = [plt.Line2D([0], [0], color=color_by_value[v], label=f"n={int(v)}") for v in legend_values]
    else:
        legend_handles = [
            plt.Line2D([0], [0], color=color_by_value[v], label=f"d={_format_density_label(v)}")
            for v in legend_values
        ]

    legend = fig.legend(
        handles=legend_handles,
        loc=str(cfg["legend_loc"]),
        bbox_to_anchor=(0.5, float(cfg["legend_bbox_y"])),
        ncol=max(1, len(legend_handles)),
        frameon=True,
        fontsize=float(cfg["legend_font_size"]),
    )
    _apply_legend_frame(legend)
    for txt in legend.get_texts():
        txt.set_color(str(cfg["text_color"]))
    fig.tight_layout(rect=[0, float(cfg["tight_layout_bottom"]), 1, float(cfg["tight_layout_top"])])

    out_dir = Path(output_dir) / SYNTH_DATA_PLOT_DIR_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    fixed_token = (
        f"d{_sanitize_filename_token(_format_density_label(fixed_density))}"
        if vary_mode == "size"
        else f"n{_sanitize_filename_token(fixed_num_nodes)}"
    )
    out_path = out_dir / (
        f"lineplot_synth_{_sanitize_filename_token(dataset)}_{_sanitize_filename_token(measure_name)}"
        f"_{vary_mode}_{fixed_token}.pdf"
    )
    fig.savefig(
        out_path,
        format="pdf",
        bbox_inches="tight",
        dpi=int(cfg["dpi"]),
        facecolor=str(cfg["figure_facecolor"]),
        edgecolor="none",
        transparent=False,
    )

    if show:
        plt.show()
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path


def plot_downstream_mean_accuracy_lines(
    perf_results: Optional[dict] = None,
    train_seed: int = EXPERIMENTS_DEFAULT_SEED,
    datasets: Any = None,
    algorithm_axis: str = "columns",
    perf_peak_plateau_map: Optional[dict] = None,
    relative_tolerance: Optional[float] = None,
    absolute_tolerance: float = 0.01,
    min_plateau_size: int = 1,
    performance_marker_mode: str = "none",
    performance_marker_alpha: float = 0.05,
    performance_marker_metric: str = ACCURACY_SCORE,
    performance_marker_classifier_name: Optional[str] = None,
    legend_position: str = "bottom",
    y_axis_mode: str = "fixed",
    y_axis_limits: Any = None,
    y_axis_padding: float = 0.02,
    include_confidence_bands: bool = False,
    confidence_band_mode: str = CONFIDENCE_BAND_DEFAULT_MODE,
) -> None:
    """Plot mean downstream accuracy across embedding dimensions."""
    perf_results = perf_results or crawl_downstream_accuracy_results(train_seed=train_seed)
    perf_raw_results = (
        crawl_downstream_accuracy_raw_results(train_seed=train_seed, metric=ACCURACY_SCORE)
        if include_confidence_bands
        else None
    )
    algorithm_axis = _validate_axis_mode(algorithm_axis)
    confidence_band_mode = _validate_confidence_band_mode(confidence_band_mode)
    performance_marker_mode = _validate_performance_marker_mode(performance_marker_mode)
    legend_position = _validate_legend_position(legend_position)
    valid_y_axis_modes = {"fixed", "zoom", "manual"}
    if y_axis_mode not in valid_y_axis_modes:
        raise ValueError(f"y_axis_mode must be one of {sorted(valid_y_axis_modes)}, got: {y_axis_mode!r}")
    show_performance_markers = performance_marker_mode != "none"

    perf_algorithms = sorted(perf_results.keys())
    perf_classifiers = sorted({clf for algo_data in perf_results.values() for clf in algo_data.keys()})
    available_by_algorithm = {
        algo: sorted(
            {ds for clf_data in perf_results.get(algo, {}).values() for ds in clf_data.keys()},
            key=_dataset_display_sort_key,
        )
        for algo in perf_algorithms
    }
    algorithm_datasets, perf_datasets = _resolve_algorithm_datasets(
        perf_algorithms,
        available_by_algorithm,
        datasets,
    )

    if len(perf_algorithms) == 0 or len(perf_classifiers) == 0 or len(perf_datasets) == 0:
        print("No downstream performance results found.")
        return

    num_panels = len(perf_algorithms) * len(perf_classifiers)
    manual_y_limits_by_panel = (
        _resolve_manual_y_limits(y_axis_limits, num_panels) if y_axis_mode == "manual" else None
    )
    all_plot_values: List[float] = []
    for algo in perf_algorithms:
        for clf_name in perf_classifiers:
            for dataset in algorithm_datasets.get(algo, []):
                for score in perf_results.get(algo, {}).get(clf_name, {}).get(dataset, {}).values():
                    try:
                        score_f = float(score)
                    except (TypeError, ValueError):
                        continue
                    if np.isfinite(score_f):
                        all_plot_values.append(score_f)
                if include_confidence_bands and perf_raw_results is not None:
                    for _, lower, upper in _confidence_band_points(
                        perf_raw_results.get(algo, {}).get(clf_name, {}).get(dataset, {}),
                        confidence_band_mode=confidence_band_mode,
                    ):
                        all_plot_values.extend([lower, upper])

    perf_dataset_colors = _dataset_color_map(perf_datasets)
    marker_maps_by_classifier: Dict[str, Optional[dict]] = {}
    if show_performance_markers:
        marker_classifiers = (
            [performance_marker_classifier_name] if performance_marker_classifier_name else perf_classifiers
        )
        for marker_clf in marker_classifiers:
            marker_maps_by_classifier[marker_clf] = _build_marker_map_for_classifier(
                marker_mode=performance_marker_mode,
                perf_peak_plateau_map=perf_peak_plateau_map,
                perf_results=perf_results,
                classifier_name=marker_clf,
                train_seed=train_seed,
                metric=performance_marker_metric,
                relative_tolerance=relative_tolerance,
                absolute_tolerance=absolute_tolerance,
                min_plateau_size=min_plateau_size,
                alpha=performance_marker_alpha,
            )

    if algorithm_axis == "columns":
        n_rows, n_cols = len(perf_classifiers), len(perf_algorithms)
    else:
        n_rows, n_cols = len(perf_algorithms), len(perf_classifiers)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
        sharex=True,
        sharey=y_axis_mode != "manual",
    )

    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    if n_cols == 1:
        axes = np.expand_dims(axes, axis=1)

    for algo_i, algo in enumerate(perf_algorithms):
        for clf_i, clf_name in enumerate(perf_classifiers):
            r, c = (clf_i, algo_i) if algorithm_axis == "columns" else (algo_i, clf_i)
            ax = axes[r][c]
            dims_seen = set()
            panel_idx = (
                clf_i * len(perf_algorithms) + algo_i
                if algorithm_axis == "columns"
                else algo_i * len(perf_classifiers) + clf_i
            )

            for dataset in algorithm_datasets.get(algo, []):
                dim_map = perf_results[algo].get(clf_name, {}).get(dataset, {})
                if not dim_map:
                    continue

                if include_confidence_bands and perf_raw_results is not None:
                    all_plot_values.extend(
                        _plot_confidence_band(
                            ax,
                            perf_raw_results.get(algo, {}).get(clf_name, {}).get(dataset, {}),
                            color=perf_dataset_colors[dataset],
                            confidence_band_mode=confidence_band_mode,
                        )
                    )

                clean_dim_map = _plot_dimension_line(
                    ax,
                    dim_map,
                    marker="o",
                    linewidth=2,
                    alpha=0.9,
                    label=dataset,
                    color=perf_dataset_colors[dataset],
                )
                if not clean_dim_map:
                    continue

                dims = sorted(clean_dim_map.keys())
                dims_seen.update(dims)
                if show_performance_markers:
                    marker_clf = performance_marker_classifier_name or clf_name
                    marker_map = marker_maps_by_classifier.get(marker_clf)
                    _overlay_performance_markers(
                        ax,
                        clean_dim_map,
                        marker_map.get(algo, {}).get(dataset, {}) if marker_map else None,
                        color=perf_dataset_colors[dataset],
                        line_marker_size=6.0,
                        marker_mode=performance_marker_mode,
                    )

            ax.set_title(f"{algo} — {clf_name}", pad=6)
            ax.set_xlabel("Embedding Dimension")
            ax.set_ylabel("Mean Accuracy")
            ymin, ymax = _resolve_panel_y_limits(
                y_axis_mode=y_axis_mode,
                default_y_axis_max=1.0,
                y_axis_padding=y_axis_padding,
                all_plot_values=all_plot_values,
                manual_y_limits_by_panel=manual_y_limits_by_panel,
                panel_idx=panel_idx,
            )
            y_step = _nice_tick_step(ymax - ymin) if y_axis_mode == "zoom" else 0.1
            yticks = _resolve_y_ticks(
                ymin=ymin,
                ymax=ymax,
                y_step=y_step,
                y_axis_mode=y_axis_mode,
            )
            ax.set_ylim(ymin, ymax)
            ax.set_yticks(yticks)
            ax.set_yticklabels(_format_axis_ticklabels(list(yticks)))
            ax.set_xscale("log", base=2)
            ax.grid(True, which="major", linestyle="--", alpha=0.4)

            if dims_seen:
                xticks = sorted(dims_seen)
                ax.set_xticks(xticks)
                ax.set_xticklabels([str(x) for x in xticks])

    legend_handles = [
        plt.Line2D([0], [0], color=perf_dataset_colors[d], label=DATASET_RENAME_DICT.get(d, d)) for d in perf_datasets
    ]
    marker_handles: List[Any] = []
    if show_performance_markers:
        marker_handles = _performance_marker_legend_handles(
            performance_marker_mode, marker_color=_legend_text_color()
        )
    fig.suptitle(f"Downstream Mean Accuracy over Embedding Dimension (train seed = {train_seed})", y=0.995)
    fig.subplots_adjust(top=0.82)
    _add_overview_legend(
        fig, legend_handles, marker_handles=marker_handles, legend_position=legend_position, top=0.94
    )
    plt.show()


def plot_functional_mean_lines(
    functional_results: Optional[dict] = None,
    datasets: Any = None,
    algorithm_axis: str = "columns",
    perf_peak_plateau_map: Optional[dict] = None,
    train_seed: int = EXPERIMENTS_DEFAULT_SEED,
    relative_tolerance: Optional[float] = None,
    absolute_tolerance: float = 0.01,
    min_plateau_size: int = 1,
    performance_marker_mode: str = "none",
    performance_marker_alpha: float = 0.05,
    performance_marker_metric: str = ACCURACY_SCORE,
    performance_marker_classifier_name: Optional[str] = None,
    legend_position: str = "bottom",
    y_axis_scale: str = "linear",
    y_axis_symlog_linthresh: Any = 0.0001,
    y_axis_symlog_linscale: float = SYMLOG_DEFAULT_LINSCALE,
    include_confidence_bands: bool = False,
    confidence_band_mode: str = CONFIDENCE_BAND_DEFAULT_MODE,
) -> None:
    """Plot mean functional similarity across embedding dimensions."""
    functional_results = functional_results or crawl_functional_results()
    functional_raw_results = crawl_functional_grouped_raw_results() if include_confidence_bands else None
    algorithm_axis = _validate_axis_mode(algorithm_axis)
    confidence_band_mode = _validate_confidence_band_mode(confidence_band_mode)
    performance_marker_mode = _validate_performance_marker_mode(performance_marker_mode)
    legend_position = _validate_legend_position(legend_position)
    y_axis_scale = _validate_y_axis_scale(y_axis_scale)
    y_axis_symlog_linscale = _validate_y_axis_symlog_linscale(y_axis_symlog_linscale)
    show_performance_markers = performance_marker_mode != "none"

    functional_classifiers = sorted(functional_results.keys())
    functional_algorithms = sorted({algo for clf_data in functional_results.values() for algo in clf_data.keys()})
    y_axis_symlog_linthresh_values = _resolve_symlog_linthresh_values(
        y_axis_symlog_linthresh,
        functional_algorithms,
    )
    functional_measures = sorted(
        {
            measure
            for clf_data in functional_results.values()
            for algo_data in clf_data.values()
            for measure in algo_data.keys()
        }
    )
    available_by_algorithm = {
        algo: sorted(
            {
                dataset
                for clf_data in functional_results.values()
                for measure_data in clf_data.get(algo, {}).values()
                for dataset in measure_data.keys()
            },
            key=_dataset_display_sort_key,
        )
        for algo in functional_algorithms
    }
    algorithm_datasets, functional_datasets = _resolve_algorithm_datasets(
        functional_algorithms,
        available_by_algorithm,
        datasets,
    )

    if (
        len(functional_classifiers) == 0
        or len(functional_algorithms) == 0
        or len(functional_measures) == 0
        or len(functional_datasets) == 0
    ):
        print("No functional stability results found.")
        return

    func_dataset_colors = _dataset_color_map(functional_datasets)
    marker_maps_by_classifier: Dict[str, Optional[dict]] = {}
    if show_performance_markers:
        marker_classifiers = (
            [performance_marker_classifier_name] if performance_marker_classifier_name else functional_classifiers
        )
        for marker_clf in marker_classifiers:
            functional_perf_peak_plateau_map = _resolve_functional_marker_map_override(
                perf_peak_plateau_map,
                classifier_name=marker_clf,
                available_classifiers=functional_classifiers,
                performance_marker_classifier_name=performance_marker_classifier_name,
            )
            marker_maps_by_classifier[marker_clf] = _build_marker_map_for_classifier(
                marker_mode=performance_marker_mode,
                perf_peak_plateau_map=functional_perf_peak_plateau_map,
                classifier_name=marker_clf,
                train_seed=train_seed,
                metric=performance_marker_metric,
                relative_tolerance=relative_tolerance,
                absolute_tolerance=absolute_tolerance,
                min_plateau_size=min_plateau_size,
                alpha=performance_marker_alpha,
            )

    for clf_name in functional_classifiers:
        if algorithm_axis == "columns":
            n_rows, n_cols = len(functional_measures), len(functional_algorithms)
        else:
            n_rows, n_cols = len(functional_algorithms), len(functional_measures)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharex=True, sharey=False)

        if n_rows == 1:
            axes = np.expand_dims(axes, axis=0)
        if n_cols == 1:
            axes = np.expand_dims(axes, axis=1)

        for algo_i, algo in enumerate(functional_algorithms):
            for measure_i, measure in enumerate(functional_measures):
                r, c = (measure_i, algo_i) if algorithm_axis == "columns" else (algo_i, measure_i)
                ax = axes[r][c]
                dims_seen = set()
                panel_plot_values: List[float] = []

                for dataset in algorithm_datasets.get(algo, []):
                    dim_map = functional_results[clf_name].get(algo, {}).get(measure, {}).get(dataset, {})
                    if not dim_map:
                        continue

                    if include_confidence_bands:
                        panel_plot_values.extend(
                            _plot_confidence_band(
                                ax,
                                _functional_embedding_raw_dim_map(
                                    functional_raw_results,
                                    clf_name,
                                    algo,
                                    measure,
                                    dataset,
                                ),
                                color=func_dataset_colors[dataset],
                                confidence_band_mode=confidence_band_mode,
                                lower_bound_min=1e-12 if y_axis_scale == "log" else 0.0,
                            )
                        )

                    clean_dim_map = _plot_dimension_line(
                        ax,
                        dim_map,
                        marker="o",
                        linewidth=2,
                        alpha=0.9,
                        label=dataset,
                        color=func_dataset_colors[dataset],
                    )
                    if not clean_dim_map:
                        continue

                    dims = sorted(clean_dim_map.keys())
                    dims_seen.update(dims)
                    panel_plot_values.extend(clean_dim_map.values())
                    if show_performance_markers:
                        marker_clf = performance_marker_classifier_name or clf_name
                        marker_map = marker_maps_by_classifier.get(marker_clf)
                        _overlay_performance_markers(
                            ax,
                            clean_dim_map,
                            marker_map.get(algo, {}).get(dataset, {}) if marker_map else None,
                            color=func_dataset_colors[dataset],
                            line_marker_size=6.0,
                            marker_mode=performance_marker_mode,
                        )

                ax.set_title(f"{algo} — {_display_funcsim_measure_name(measure)}", pad=6)
                ax.set_xlabel("Embedding Dimension")
                ax.set_ylabel(_display_funcsim_measure_name(measure))
                if y_axis_scale == "log":
                    ymin, ymax = _resolve_log_y_limits(0.0, _functional_yaxis_max(measure), panel_plot_values)
                    ax.set_yscale("log")
                    ax.set_ylim(ymin, ymax)
                elif y_axis_scale == "symlog":
                    panel_linthresh = y_axis_symlog_linthresh_values[algo_i]
                    ymin, ymax = _resolve_symlog_y_limits(
                        0.0,
                        _functional_yaxis_max(measure),
                        default_y_axis_max=1.0,
                    )
                    _apply_symlog_yaxis(
                        ax,
                        ymin,
                        ymax,
                        linthresh=panel_linthresh,
                        linscale=y_axis_symlog_linscale,
                    )
                else:
                    ax.set_yscale("linear")
                    _apply_bounded_yaxis(ax, y_max=_functional_yaxis_max(measure))
                ax.set_xscale("log", base=2)
                ax.grid(True, which="major", linestyle="--", alpha=0.4)

                if dims_seen:
                    xticks = sorted(dims_seen)
                    ax.set_xticks(xticks)
                    ax.set_xticklabels([str(x) for x in xticks])

        handles = [
            plt.Line2D([0], [0], color=func_dataset_colors[d], label=DATASET_RENAME_DICT.get(d, d))
            for d in functional_datasets
        ]
        marker_handles: List[Any] = []
        if show_performance_markers:
            marker_handles = _performance_marker_legend_handles(
                performance_marker_mode, marker_color=_legend_text_color()
            )
        fig.suptitle(f"Functional Similarity over Embedding Dimension ({clf_name})", y=0.995)
        fig.subplots_adjust(top=0.82)
        _add_overview_legend(fig, handles, marker_handles=marker_handles, legend_position=legend_position, top=0.94)
        plt.show()


def plot_representational_stability_with_performance_markers(
    results: Optional[dict] = None,
    perf_peak_plateau_map: Optional[dict] = None,
    *,
    train_seed: int = EXPERIMENTS_DEFAULT_SEED,
    relative_tolerance: Optional[float] = None,
    absolute_tolerance: float = 0.01,
    min_plateau_size: int = 1,
    performance_marker_mode: str = "threshold",
    performance_marker_alpha: float = 0.05,
    performance_marker_classifier_name: Optional[str] = None,
    performance_marker_metric: str = ACCURACY_SCORE,
    datasets: Any = None,
    algorithm_axis: str = "columns",
    legend_position: str = "bottom",
    include_confidence_bands: bool = False,
    confidence_band_mode: str = CONFIDENCE_BAND_DEFAULT_MODE,
) -> None:
    """Plot representational stability and overlay best/near-best downstream dimensions."""
    results = results or crawl_results()
    raw_results = crawl_representational_raw_results() if include_confidence_bands else None
    algorithm_axis = _validate_axis_mode(algorithm_axis)
    confidence_band_mode = _validate_confidence_band_mode(confidence_band_mode)
    performance_marker_mode = _validate_performance_marker_mode(performance_marker_mode)
    legend_position = _validate_legend_position(legend_position)
    show_performance_markers = performance_marker_mode != "none"
    if show_performance_markers and perf_peak_plateau_map is None:
        if performance_marker_mode == "threshold":
            perf_peak_plateau_map = build_downstream_peak_plateau_map(
                train_seed=train_seed,
                classifier_name=performance_marker_classifier_name,
                relative_tolerance=relative_tolerance,
                absolute_tolerance=absolute_tolerance,
                min_plateau_size=min_plateau_size,
            )
        else:
            perf_peak_plateau_map = build_downstream_peak_plateau_map_statistical(
                train_seed=train_seed,
                classifier_name=performance_marker_classifier_name,
                metric=performance_marker_metric,
                alpha=performance_marker_alpha,
            )

    algorithms = sorted(results.keys())
    similarity_measures = sorted({sim for algo in results.values() for sim in algo.keys()})
    available_by_algorithm = {
        algo: sorted(
            {dataset for sim in results.get(algo, {}).values() for dataset in sim.keys()},
            key=_dataset_display_sort_key,
        )
        for algo in algorithms
    }
    algorithm_datasets, selected_datasets = _resolve_algorithm_datasets(
        algorithms,
        available_by_algorithm,
        datasets,
    )

    if not algorithms or not similarity_measures or len(selected_datasets) == 0:
        print("No representational stability results found.")
        return

    dataset_colors = _dataset_color_map(selected_datasets)

    if algorithm_axis == "columns":
        n_rows, n_cols = len(similarity_measures), len(algorithms)
    else:
        n_rows, n_cols = len(algorithms), len(similarity_measures)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharex=True, sharey=True)

    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    if n_cols == 1:
        axes = np.expand_dims(axes, axis=1)

    for algo_i, algo in enumerate(algorithms):
        for sim_i, sim in enumerate(similarity_measures):
            r, c = (sim_i, algo_i) if algorithm_axis == "columns" else (algo_i, sim_i)
            ax = axes[r][c]
            dims_seen = set()

            for dataset in algorithm_datasets.get(algo, []):
                dim_map = results[algo].get(sim, {}).get(dataset, {})
                if not dim_map:
                    continue

                color = dataset_colors[dataset]
                if include_confidence_bands and raw_results is not None:
                    _plot_confidence_band(
                        ax,
                        raw_results.get(algo, {}).get(sim, {}).get(dataset, {}),
                        color=color,
                        confidence_band_mode=confidence_band_mode,
                    )
                clean_dim_map = _plot_dimension_line(
                    ax,
                    dim_map,
                    marker="o",
                    linewidth=2,
                    alpha=0.9,
                    label=dataset,
                    color=color,
                    clip_on=False,
                )
                if not clean_dim_map:
                    continue

                dims = sorted(clean_dim_map.keys())
                dims_seen.update(dims)

                if show_performance_markers:
                    _overlay_performance_markers(
                        ax,
                        clean_dim_map,
                        perf_peak_plateau_map.get(algo, {}).get(dataset, {}) if perf_peak_plateau_map else None,
                        color=color,
                        line_marker_size=6.0,
                        marker_mode=performance_marker_mode,
                    )

            sim_display = _display_repsim_measure_name(sim)
            ax.set_title(f"{algo} — {sim_display}", pad=6)
            ax.set_xlabel("Embedding Dimension")
            ax.set_ylabel(sim_display)
            _apply_bounded_yaxis(ax)
            ax.set_xscale("log", base=2)
            ax.grid(True, which="major", linestyle="--", alpha=0.4)
            if dims_seen:
                xticks = sorted(dims_seen)
                ax.set_xticks(xticks)
                ax.set_xticklabels([str(x) for x in xticks])

    legend_handles = [
        plt.Line2D([0], [0], color=dataset_colors[d], label=DATASET_RENAME_DICT.get(d, d)) for d in selected_datasets
    ]
    marker_handles: List[Any] = []
    if show_performance_markers:
        marker_handles = _performance_marker_legend_handles(
            performance_marker_mode, marker_color=_legend_text_color()
        )

    _add_overview_legend(fig, legend_handles, marker_handles=marker_handles, legend_position=legend_position)
    plt.show()


# ---------------------------------------------------------------------------
# Public Data And Diagnostics API
# ---------------------------------------------------------------------------


def crawl_stability_performance_bootstrap_results(
    results_dir: Any = None,
) -> List[Dict[str, Any]]:
    """Load stability-performance bootstrap summary JSON files."""
    root = Path(results_dir) if results_dir is not None else STABILITY_PERFORMANCE_BOOTSTRAP_RESULTS_DIR
    if not root.exists():
        warnings.warn(f"Bootstrap results directory does not exist: {root}", RuntimeWarning)
        return []

    rows: List[Dict[str, Any]] = []
    for summary_path in sorted(root.glob("*_summary.json")):
        try:
            with open(summary_path, encoding="utf-8") as f:
                raw = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            warnings.warn(f"Skipping unreadable bootstrap summary {summary_path}: {exc}", RuntimeWarning)
            continue

        if not isinstance(raw, dict):
            continue
        try:
            hit_rate = float(raw["hit_rate"])
        except (KeyError, TypeError, ValueError):
            warnings.warn(f"Skipping bootstrap summary without numeric hit_rate: {summary_path}", RuntimeWarning)
            continue
        if not np.isfinite(hit_rate):
            continue

        row = dict(raw)
        row["hit_rate"] = hit_rate
        row["source_path"] = str(summary_path)
        rows.append(row)

    return rows


def filter_stability_performance_bootstrap_results(
    rows: List[Dict[str, Any]],
    *,
    excluded_algorithm_datasets: Optional[Dict[str, Set[str]]] = None,
) -> List[Dict[str, Any]]:
    """Return bootstrap rows after removing explicitly excluded algorithm/dataset combinations."""
    exclusions = excluded_algorithm_datasets if excluded_algorithm_datasets is not None else {ASNE: {DDI, COAUTHOR}}
    normalized_exclusions = {
        str(algorithm): {str(dataset) for dataset in datasets}
        for algorithm, datasets in exclusions.items()
    }

    filtered: List[Dict[str, Any]] = []
    for row in rows:
        algorithm = str(row.get("algorithm"))
        dataset = str(row.get("dataset"))
        if dataset in normalized_exclusions.get(algorithm, set()):
            continue
        filtered.append(row)
    return filtered


def _best_dims_from_dim_map(
    dim_map: Dict[int, float],
    *,
    objective: str = "max",
    rtol: float = 1e-12,
    atol: float = 1e-15,
) -> List[int]:
    clean = {}
    for dim, value in dim_map.items():
        try:
            dim_i = int(dim)
            value_f = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value_f):
            clean[dim_i] = value_f
    if not clean:
        return []
    if objective == "min":
        best_score = min(clean.values())
    else:
        best_score = max(clean.values())
    return sorted(
        dim
        for dim, value in clean.items()
        if np.isclose(value, best_score, rtol=rtol, atol=atol)
    )


def _bootstrap_original_optimum_relation(row: Dict[str, Any], caches: Dict[str, Any]) -> Optional[str]:
    algorithm = row.get("algorithm")
    dataset = row.get("dataset")
    classifier = row.get("classifier")
    metric = row.get("metric", ACCURACY_SCORE)
    stability_type = row.get("stability_type")
    stability_measure = row.get("stability_measure")
    stability_objective = row.get("stability_objective", "max")
    train_seed = row.get("train_seed", EXPERIMENTS_DEFAULT_SEED)

    if algorithm is None or dataset is None or classifier is None or stability_measure is None:
        return None

    perf_cache_key = (int(train_seed), str(metric))
    if perf_cache_key not in caches["performance"]:
        caches["performance"][perf_cache_key] = crawl_downstream_accuracy_results(
            train_seed=int(train_seed),
            metric=str(metric),
        )
    perf_dim_map = (
        caches["performance"][perf_cache_key]
        .get(str(algorithm), {})
        .get(str(classifier), {})
        .get(str(dataset), {})
    )
    performance_dims = _best_dims_from_dim_map(perf_dim_map, objective="max")

    if stability_type == FUNCTIONAL:
        if "functional_stability" not in caches:
            caches["functional_stability"] = crawl_functional_results()
        stability_dim_map = (
            caches["functional_stability"]
            .get(str(classifier), {})
            .get(str(algorithm), {})
            .get(str(stability_measure), {})
            .get(str(dataset), {})
        )
    else:
        if "representational_stability" not in caches:
            caches["representational_stability"] = crawl_results()
        stability_dim_map = (
            caches["representational_stability"]
            .get(str(algorithm), {})
            .get(str(stability_measure), {})
            .get(str(dataset), {})
        )
    stability_dims = _best_dims_from_dim_map(stability_dim_map, objective=str(stability_objective))
    if not stability_dims or not performance_dims:
        return None
    stability_set = set(int(dim) for dim in stability_dims)
    performance_set = set(int(dim) for dim in performance_dims)
    if stability_set.intersection(performance_set):
        return "same"
    if max(stability_set) < min(performance_set):
        return "lower"
    if min(stability_set) > max(performance_set):
        return "higher"
    return "mixed"


def _bootstrap_hit_rate_plot_rows(
    stability_measure: str,
    *,
    raw_rows: List[Dict[str, Any]],
    algorithms: Any = None,
    datasets: Any = None,
    classifier: Optional[str] = LOGISTIC_REGRESSION,
    metric: Optional[str] = ACCURACY_SCORE,
    stability_type: Optional[str] = None,
    performance_criterion: Optional[str] = "threshold",
    show_optimum_relation_markers: bool = True,
    optimum_relation_caches: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    algorithm_filter = _optional_filter_set(algorithms)
    dataset_filter = _optional_filter_set(datasets)
    caches = optimum_relation_caches if optimum_relation_caches is not None else {}
    caches.setdefault("performance", {})

    grouped_hit_rates: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    grouped_counts: Dict[Tuple[str, str], int] = defaultdict(int)
    grouped_relations: Dict[Tuple[str, str], List[str]] = defaultdict(list)

    for row in raw_rows:
        if str(row.get("stability_measure")) != str(stability_measure):
            continue
        if classifier is not None and str(row.get("classifier")) != str(classifier):
            continue
        if metric is not None and str(row.get("metric")) != str(metric):
            continue
        if stability_type is not None and str(row.get("stability_type")) != str(stability_type):
            continue
        if performance_criterion is not None and str(row.get("performance_criterion")) != str(performance_criterion):
            continue

        algorithm = row.get("algorithm")
        dataset = row.get("dataset")
        if algorithm is None or dataset is None:
            continue
        algorithm = str(algorithm)
        dataset = str(dataset)
        if algorithm_filter is not None and algorithm not in algorithm_filter:
            continue
        if dataset_filter is not None and dataset not in dataset_filter:
            continue

        grouped_hit_rates[(algorithm, dataset)].append(float(row["hit_rate"]))
        grouped_counts[(algorithm, dataset)] += 1
        if show_optimum_relation_markers:
            relation = _bootstrap_original_optimum_relation(row, caches)
            if relation is not None:
                grouped_relations[(algorithm, dataset)].append(relation)

    plot_rows = [
        {
            "algorithm": algorithm,
            "dataset": dataset,
            "hit_rate": float(np.mean(values)),
            "num_summaries": grouped_counts[(algorithm, dataset)],
            "optimum_relation": (
                Counter(grouped_relations.get((algorithm, dataset), [])).most_common(1)[0][0]
                if grouped_relations.get((algorithm, dataset), [])
                else None
            ),
        }
        for (algorithm, dataset), values in grouped_hit_rates.items()
        if values
    ]
    return plot_rows


def _draw_bootstrap_hit_rate_panel(
    ax: Any,
    plot_rows: List[Dict[str, Any]],
    *,
    stability_measure: str,
    selected_algorithms: List[str],
    selected_datasets: List[str],
    dataset_colors: Dict[str, Any],
    jitter: float = 0.12,
    marker_size: float = 34.0,
    show_ylabel: bool = True,
    square_panel: bool = True,
    show_xticklabels: bool = True,
) -> None:
    axis_label_size = float(LINEPLOT_PAPER_STYLE_DEFAULT["axis_label_size"])
    tick_label_size = float(LINEPLOT_PAPER_STYLE_DEFAULT["tick_label_size"])
    dataset_offsets = {
        dataset: float(offset)
        for dataset, offset in zip(
            selected_datasets,
            np.linspace(-float(jitter), float(jitter), num=len(selected_datasets)),
        )
    }
    if len(selected_datasets) == 1:
        dataset_offsets[selected_datasets[0]] = 0.0

    ax.set_facecolor("white")

    x_positions = {algorithm: idx for idx, algorithm in enumerate(selected_algorithms)}
    for row in sorted(
        plot_rows,
        key=lambda r: (_algorithm_alpha_sort_key(r["algorithm"]), _dataset_alpha_sort_key(r["dataset"])),
    ):
        x = x_positions[row["algorithm"]] + dataset_offsets[row["dataset"]]
        relation = row.get("optimum_relation")
        marker = BOOTSTRAP_OPTIMUM_RELATION_MARKERS.get(str(relation), {}).get("marker", "o")
        ax.scatter(
            x,
            row["hit_rate"],
            s=marker_size,
            marker=marker,
            color=dataset_colors[row["dataset"]],
            edgecolors="none",
            linewidths=0.0,
            alpha=1.0,
            clip_on=False,
            zorder=5,
        )

    ax.set_xticks([x_positions[algorithm] for algorithm in selected_algorithms])
    ax.set_xticklabels(
        [
            EMBEDDING_ALGORITHM_RENAME_DICT.get(algorithm, algorithm) if show_xticklabels else ""
            for algorithm in selected_algorithms
        ],
        rotation=30,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(np.arange(0.0, 1.01, 0.1))
    ax.set_yticklabels([f"{tick:.1f}" for tick in np.arange(0.0, 1.01, 0.1)])
    ax.set_xlabel("")
    ax.set_ylabel("Bootstrap Alignment Rate" if show_ylabel else "", color="black", fontsize=axis_label_size)
    ax.grid(axis="y", color="#9a9a9a", linewidth=0.7, alpha=0.28, linestyle="--", zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(
        axis="both",
        labelsize=tick_label_size,
        colors="black",
        direction="out",
        length=5.2,
        width=0.9,
        pad=1.5,
        bottom=True,
        left=True,
        top=False,
        right=False,
    )
    ax.tick_params(axis="x", labelsize=axis_label_size)
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    if square_panel:
        ax.set_box_aspect(1)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.9)

    title = _display_funcsim_measure_name(stability_measure)
    if title == stability_measure:
        title = _display_repsim_measure_name(stability_measure)
    ax.set_title(title, color="black", fontsize=axis_label_size + 1.0, pad=7.0)


def plot_stability_performance_bootstrap_hit_rate_grid(
    stability_measures: Any,
    *,
    results: Optional[List[Dict[str, Any]]] = None,
    algorithms: Any = None,
    datasets: Any = None,
    classifier: Optional[str] = LOGISTIC_REGRESSION,
    metric: Optional[str] = ACCURACY_SCORE,
    stability_type: Optional[str] = None,
    performance_criterion: Optional[str] = "threshold",
    results_dir: Any = None,
    output_dir: Any = BOOTSTRAP_HIT_RATE_DEFAULT_OUTPUT_DIR,
    save: bool = True,
    show: bool = True,
    figure_width_per_panel: float = 3.25,
    figure_height_per_panel: float = 2.25,
    jitter: float = 0.12,
    marker_size: float = 34.0,
    max_cols: int = 2,
    show_optimum_relation_markers: bool = True,
) -> Any:
    """Plot bootstrap hit rates for multiple stability measures in a two-column grid by default."""
    if isinstance(stability_measures, str):
        selected_measures = [stability_measures]
    else:
        selected_measures = [str(measure) for measure in stability_measures]
    if not selected_measures:
        warnings.warn("No stability measures supplied for bootstrap hit-rate grid.", RuntimeWarning)
        return None

    raw_rows = results if results is not None else crawl_stability_performance_bootstrap_results(results_dir)
    rows_by_measure: Dict[str, List[Dict[str, Any]]] = {}
    all_algorithms: Set[str] = set()
    all_datasets: Set[str] = set()
    optimum_relation_caches: Dict[str, Any] = {"performance": {}}
    for measure in selected_measures:
        measure_rows = _bootstrap_hit_rate_plot_rows(
            measure,
            raw_rows=raw_rows,
            algorithms=algorithms,
            datasets=datasets,
            classifier=classifier,
            metric=metric,
            stability_type=stability_type,
            performance_criterion=performance_criterion,
            show_optimum_relation_markers=show_optimum_relation_markers,
            optimum_relation_caches=optimum_relation_caches,
        )
        rows_by_measure[measure] = measure_rows
        all_algorithms.update(row["algorithm"] for row in measure_rows)
        all_datasets.update(row["dataset"] for row in measure_rows)

    if not any(rows_by_measure.values()):
        warnings.warn("No bootstrap hit-rate summaries found for the selected measures.", RuntimeWarning)
        return None

    selected_algorithms = sorted(all_algorithms, key=_algorithm_alpha_sort_key)
    selected_datasets = sorted(all_datasets, key=_dataset_alpha_sort_key)
    dataset_colors = _dataset_color_map(selected_datasets)

    n_cols = max(1, min(int(max_cols), len(selected_measures)))
    n_rows = int(np.ceil(len(selected_measures) / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figure_width_per_panel * n_cols, figure_height_per_panel * n_rows),
        squeeze=False,
    )
    fig.patch.set_facecolor("white")

    for panel_idx, measure in enumerate(selected_measures):
        row_idx, col_idx = divmod(panel_idx, n_cols)
        ax = axes[row_idx][col_idx]
        measure_rows = rows_by_measure.get(measure, [])
        if measure_rows:
            _draw_bootstrap_hit_rate_panel(
                ax,
                measure_rows,
                stability_measure=measure,
                selected_algorithms=selected_algorithms,
                selected_datasets=selected_datasets,
                dataset_colors=dataset_colors,
                jitter=jitter,
                marker_size=marker_size,
                show_ylabel=col_idx == 0,
                square_panel=False,
                show_xticklabels=True,
            )
        else:
            axis_label_size = float(LINEPLOT_PAPER_STYLE_DEFAULT["axis_label_size"])
            tick_label_size = float(LINEPLOT_PAPER_STYLE_DEFAULT["tick_label_size"])
            ax.set_facecolor("white")
            ax.set_ylim(0.0, 1.0)
            ax.set_xticks([idx for idx, _ in enumerate(selected_algorithms)])
            ax.set_xticklabels(
                [
                    EMBEDDING_ALGORITHM_RENAME_DICT.get(algorithm, algorithm)
                    for algorithm in selected_algorithms
                ],
                rotation=30,
                ha="right",
                rotation_mode="anchor",
            )
            ax.set_yticks(np.arange(0.0, 1.01, 0.1))
            ax.set_yticklabels([f"{tick:.1f}" for tick in np.arange(0.0, 1.01, 0.1)])
            ax.grid(axis="y", color="#9a9a9a", linewidth=0.7, alpha=0.28, linestyle="--", zorder=0)
            ax.tick_params(
                axis="both",
                labelsize=tick_label_size,
                colors="black",
                direction="out",
                length=5.2,
                width=0.9,
                pad=1.5,
                bottom=True,
                left=True,
                top=False,
                right=False,
            )
            ax.tick_params(axis="x", labelsize=axis_label_size)
            ax.xaxis.set_ticks_position("bottom")
            ax.yaxis.set_ticks_position("left")
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color("black")
                spine.set_linewidth(0.9)
            title = _display_funcsim_measure_name(measure)
            if title == measure:
                title = _display_repsim_measure_name(measure)
            ax.set_title(title, color="black", fontsize=axis_label_size + 1.0, pad=7.0)
            ax.set_xlabel("")
            ax.set_ylabel(
                "Bootstrap Alignment Rate" if col_idx == 0 else "",
                color="black",
                fontsize=axis_label_size,
            )
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center", color="black")

    for empty_idx in range(len(selected_measures), n_rows * n_cols):
        row_idx, col_idx = divmod(empty_idx, n_cols)
        axes[row_idx][col_idx].set_visible(False)

    dataset_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=dataset_colors[dataset],
            markeredgecolor="none",
            markeredgewidth=0.0,
            markersize=4.2,
            label=DATASET_RENAME_DICT.get(dataset, dataset),
        )
        for dataset in selected_datasets
    ]
    relation_set = {
        str(row["optimum_relation"])
        for measure_rows in rows_by_measure.values()
        for row in measure_rows
        if row.get("optimum_relation") is not None
    }
    relation_handles: List[Any] = []
    if show_optimum_relation_markers:
        ordered_relations = ["higher", "lower", "same"]
        if "mixed" in relation_set:
            ordered_relations.append("mixed")
        for relation in ordered_relations:
            if relation not in relation_set:
                continue
            style = BOOTSTRAP_OPTIMUM_RELATION_MARKERS[relation]
            relation_handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker=str(style["marker"]),
                    linestyle="",
                    markerfacecolor="black",
                    markeredgecolor="none",
                    markeredgewidth=0.0,
                    markersize=4.2,
                    label=str(style["label"]),
                )
            )
    if dataset_handles:
        legends: List[Any] = []
        if relation_handles:
            legend_datasets = fig.legend(
                handles=dataset_handles,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.135),
                ncol=min(len(dataset_handles), 7),
                frameon=False,
                fontsize=6.5,
                handletextpad=0.35,
            )
            legends.append(legend_datasets)
            legend_markers = fig.legend(
                handles=relation_handles,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.095),
                ncol=len(relation_handles),
                frameon=False,
                fontsize=6.5,
                handletextpad=0.45,
                columnspacing=1.2,
            )
            legends.append(legend_markers)
            for legend in legends:
                _set_legend_text_color(legend, "black")
            _draw_combined_legend_frame(fig, legends)
        else:
            legend = fig.legend(
                handles=dataset_handles,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.09),
                ncol=min(len(dataset_handles), 7),
                frameon=True,
                fontsize=6.5,
                handletextpad=0.35,
            )
            _apply_legend_frame(legend)
            _set_legend_text_color(legend, "black")

    fig.tight_layout(rect=[0, 0.19 if relation_handles else 0.16, 1, 1], w_pad=0.5, h_pad=1.8)
    fig.subplots_adjust(wspace=0.24, hspace=0.82)

    out_path = None
    if save:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        measure_token = "_".join(_sanitize_filename_token(measure) for measure in selected_measures)
        if len(measure_token) > 90:
            measure_token = f"{len(selected_measures)}_measures"
        stem_parts = [BOOTSTRAP_ALIGNMENT_FILE_STEM, measure_token]
        if classifier:
            stem_parts.append(classifier)
        if performance_criterion:
            stem_parts.append(performance_criterion)
        stem = "_".join(_sanitize_filename_token(part) for part in stem_parts)
        out_path = out_dir / f"{stem}.pdf"
        fig.savefig(out_path, format="pdf", bbox_inches="tight", dpi=300, facecolor="white")
        print(f"Saved {out_path}")

    if show:
        plt.show()
    if save:
        plt.close(fig)

    return out_path if save else (fig, axes, rows_by_measure)


def _embedding_cost_summary_paths(run_dir: Path) -> Tuple[Path, Path]:
    reports_dir = run_dir / REPORTS_DIR_NAME
    return reports_dir / EMBEDDING_COSTS_SUMMARY_FILE_NAME, reports_dir / DOWNSTREAM_COSTS_SUMMARY_FILE_NAME


def _resolve_embedding_cost_run_dirs(
    results_dirs: Any = None,
    run_ids: Any = "latest",
) -> List[Path]:
    roots = results_dirs if results_dirs is not None else EMBEDDING_COST_RESULTS_DIRS
    if isinstance(roots, (str, Path)):
        roots = [roots]
    root_paths = [Path(root) for root in roots]

    available: Dict[str, Path] = {}
    for root in root_paths:
        if not root.exists():
            continue
        for run_dir in root.iterdir():
            if not run_dir.is_dir():
                continue
            embedding_path, downstream_path = _embedding_cost_summary_paths(run_dir)
            if embedding_path.exists() or downstream_path.exists():
                available.setdefault(run_dir.name, run_dir)

    if not available:
        warnings.warn("No embedding-cost summary runs found.", RuntimeWarning)
        return []

    if run_ids is None or run_ids == "latest":
        return [available[sorted(available)[-1]]]
    if run_ids == "all":
        return [available[name] for name in sorted(available)]

    requested = [run_ids] if isinstance(run_ids, str) else list(run_ids)
    selected: List[Path] = []
    missing: List[str] = []
    for run_id in requested:
        run_id_s = str(run_id)
        if run_id_s in available:
            selected.append(available[run_id_s])
        else:
            missing.append(run_id_s)
    if missing:
        warnings.warn(f"Embedding-cost run ids not found: {missing}", RuntimeWarning)
    return selected


def _load_embedding_cost_summary(path: Path, *, run_id: str, stage: str) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        with path.open(newline="", encoding="utf-8") as f:
            raw = [dict(row) for row in csv.DictReader(f)]
    except OSError as exc:
        warnings.warn(f"Skipping unreadable embedding-cost summary {path}: {exc}", RuntimeWarning)
        return []
    rows: List[Dict[str, Any]] = []
    for row in raw:
        out = dict(row)
        out["run_id"] = run_id
        out["cost_stage"] = stage
        out["source_path"] = str(path)
        rows.append(out)
    return rows


def crawl_embedding_cost_results(
    results_dirs: Any = None,
    run_ids: Any = "latest",
) -> Dict[str, List[Dict[str, Any]]]:
    """Load embedding and downstream cost summary rows from case-study output runs."""
    embedding_rows: List[Dict[str, Any]] = []
    downstream_rows: List[Dict[str, Any]] = []
    for run_dir in _resolve_embedding_cost_run_dirs(results_dirs=results_dirs, run_ids=run_ids):
        embedding_path, downstream_path = _embedding_cost_summary_paths(run_dir)
        embedding_rows.extend(_load_embedding_cost_summary(embedding_path, run_id=run_dir.name, stage="embedding"))
        downstream_rows.extend(_load_embedding_cost_summary(downstream_path, run_id=run_dir.name, stage="downstream"))
    return {"embedding": embedding_rows, "downstream": downstream_rows}


def _filter_embedding_cost_rows(
    rows: List[Dict[str, Any]],
    *,
    datasets: Any,
    algorithms: Any,
    dimensions: Any,
    classifier: Optional[str] = None,
) -> List[Dict[str, Any]]:
    dataset_filter = _optional_filter_set(datasets)
    algorithm_filter = _optional_filter_set(algorithms)
    dimension_filter = None
    if dimensions is not None:
        if isinstance(dimensions, (str, int)):
            dimension_filter = {int(dimensions)}
        else:
            dimension_filter = {int(dim) for dim in dimensions}

    filtered: List[Dict[str, Any]] = []
    for row in rows:
        dataset = row.get("dataset")
        algorithm = row.get("algorithm")
        dimension = row.get("dimension")
        if dataset is None or algorithm is None or dimension is None:
            continue
        if dataset_filter is not None and str(dataset) not in dataset_filter:
            continue
        if algorithm_filter is not None and str(algorithm) not in algorithm_filter:
            continue
        if dimension_filter is not None and int(dimension) not in dimension_filter:
            continue
        if classifier is not None and str(row.get("classifier")) != str(classifier):
            continue
        filtered.append(row)
    return filtered


def _summarize_embedding_cost_stage(
    rows: List[Dict[str, Any]],
    *,
    time_unit: str,
    memory_unit: str,
    memory_field: str,
) -> Dict[Tuple[str, str, int], Dict[str, float]]:
    grouped: Dict[Tuple[str, str, int], Dict[str, List[float]]] = defaultdict(lambda: {"time": [], "memory": []})
    for row in rows:
        try:
            time_value = float(row["elapsed_seconds_mean"])
            if time_unit == "min":
                time_value /= 60.0
            elif time_unit == "h":
                time_value /= 3600.0
            elif time_unit != "s":
                raise ValueError("time_unit must be one of {'s', 'min', 'h'}")

            memory_value = float(row[memory_field])
            if memory_unit == "MB":
                memory_value /= 1024.0**2
            elif memory_unit == "GB":
                memory_value /= 1024.0**3
            elif memory_unit != "B":
                raise ValueError("memory_unit must be one of {'B', 'MB', 'GB'}")

            dimension = int(row["dimension"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (np.isfinite(time_value) and np.isfinite(memory_value)):
            continue
        key = (str(row["dataset"]), str(row["algorithm"]), dimension)
        grouped[key]["time"].append(time_value)
        grouped[key]["memory"].append(memory_value)

    summary: Dict[Tuple[str, str, int], Dict[str, float]] = {}
    for key, values in grouped.items():
        summary[key] = {
            "time": float(np.mean(values["time"])) if values["time"] else np.nan,
            "memory": float(np.mean(values["memory"])) if values["memory"] else np.nan,
        }
    return summary


def build_embedding_cost_tables_by_dataset(
    cost_results: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    *,
    results_dirs: Any = None,
    run_ids: Any = "latest",
    datasets: Any = None,
    algorithms: Any = None,
    dimensions: Any = None,
    classifier: Optional[str] = LOGISTIC_REGRESSION,
    time_unit: str = "s",
    memory_unit: str = "GB",
    memory_field: str = "peak_rss_delta_bytes_mean",
    round_digits: Optional[int] = 2,
) -> Dict[str, Any]:
    """Build one cost table per dataset with method rows and stage/metric columns."""
    import pandas as pd

    loaded = cost_results if cost_results is not None else crawl_embedding_cost_results(
        results_dirs=results_dirs,
        run_ids=run_ids,
    )
    embedding_rows = _filter_embedding_cost_rows(
        loaded.get("embedding", []),
        datasets=datasets,
        algorithms=algorithms,
        dimensions=dimensions,
    )
    downstream_rows = _filter_embedding_cost_rows(
        loaded.get("downstream", []),
        datasets=datasets,
        algorithms=algorithms,
        dimensions=dimensions,
        classifier=classifier,
    )

    embedding_summary = _summarize_embedding_cost_stage(
        embedding_rows,
        time_unit=time_unit,
        memory_unit=memory_unit,
        memory_field=memory_field,
    )
    downstream_summary = _summarize_embedding_cost_stage(
        downstream_rows,
        time_unit=time_unit,
        memory_unit=memory_unit,
        memory_field=memory_field,
    )

    keys = set(embedding_summary) | set(downstream_summary)
    selected_datasets = sorted({dataset for dataset, _, _ in keys}, key=_dataset_alpha_sort_key)
    columns = pd.MultiIndex.from_tuples(
        [
            ("Embedding Generation", f"Time ({time_unit})"),
            ("Embedding Generation", f"Memory ({memory_unit})"),
            ("Downstream Run", f"Time ({time_unit})"),
            ("Downstream Run", f"Memory ({memory_unit})"),
        ]
    )

    tables: Dict[str, Any] = {}
    for dataset in selected_datasets:
        dataset_algorithm_dimensions = sorted(
            [(algorithm, dimension) for ds, algorithm, dimension in keys if ds == dataset],
            key=lambda item: (_algorithm_alpha_sort_key(item[0]), int(item[1])),
        )
        table_rows: List[List[float]] = []
        index: List[Tuple[str, int]] = []
        for algorithm, dimension in dataset_algorithm_dimensions:
            emb = embedding_summary.get((dataset, algorithm, dimension), {})
            down = downstream_summary.get((dataset, algorithm, dimension), {})
            table_rows.append(
                [
                    emb.get("time", np.nan),
                    emb.get("memory", np.nan),
                    down.get("time", np.nan),
                    down.get("memory", np.nan),
                ]
            )
            index.append((str(EMBEDDING_ALGORITHM_RENAME_DICT.get(algorithm, algorithm)), int(dimension)))
        table = pd.DataFrame(table_rows, index=pd.MultiIndex.from_tuples(index, names=["Method", "Dimension"]), columns=columns)
        if round_digits is not None:
            table = table.round(int(round_digits))
        tables[dataset] = table

    return tables


def _bool_from_report_value(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def _float_from_report_value(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return np.nan
    return out if np.isfinite(out) else np.nan


def crawl_hyperparameter_sensitivity_results(
    results_dir: Any = None,
    *,
    datasets: Any = None,
    algorithms: Any = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Load hyperparameter-sensitivity report rows from all matching report directories."""
    dataset_filter = _optional_filter_set(datasets)
    algorithm_filter = _optional_filter_set(algorithms)
    loaded: Dict[str, List[Dict[str, Any]]] = {
        "stage1_comparison": [],
        "performance": [],
        "representational": [],
        "functional": [],
        "downstream_tuning": [],
    }

    root = Path(results_dir) if results_dir is not None else HYPERPARAMETER_SENSITIVITY_RESULTS_DIR
    if not root.exists():
        warnings.warn(f"Hyperparameter-sensitivity results directory does not exist: {root}", RuntimeWarning)
        return loaded

    for reports_dir in sorted(path for path in root.rglob(REPORTS_DIR_NAME) if path.is_dir()):
        try:
            path_algorithm = reports_dir.parents[2].name
            path_dataset = reports_dir.parents[1].name
        except IndexError:
            path_algorithm = path_dataset = None
        metadata_path = reports_dir / RUN_METADATA_FILE_NAME
        metadata: Dict[str, Any] = {}
        if metadata_path.exists():
            try:
                with metadata_path.open(encoding="utf-8") as f:
                    metadata = json.load(f)
            except (OSError, json.JSONDecodeError):
                metadata = {}
        algorithm = metadata.get("algorithm")
        dataset = metadata.get("dataset")
        if algorithm is not None and path_algorithm is not None and str(algorithm) != str(path_algorithm):
            warnings.warn(
                f"Skipping hyperparameter-sensitivity reports with inconsistent algorithm metadata: "
                f"{reports_dir} has metadata {algorithm!r} but path implies {path_algorithm!r}.",
                RuntimeWarning,
            )
            continue
        if dataset is not None and path_dataset is not None and str(dataset) != str(path_dataset):
            warnings.warn(
                f"Skipping hyperparameter-sensitivity reports with inconsistent dataset metadata: "
                f"{reports_dir} has metadata {dataset!r} but path implies {path_dataset!r}.",
                RuntimeWarning,
            )
            continue
        if algorithm is None or dataset is None:
            algorithm = path_algorithm
            dataset = path_dataset
        if algorithm_filter is not None and str(algorithm) not in algorithm_filter:
            continue
        if dataset_filter is not None and str(dataset) not in dataset_filter:
            continue

        stage1_path = reports_dir / HYPERPARAMETER_SENSITIVITY_STAGE1_COMPARISON_FILE_NAME
        if not stage1_path.exists():
            stage1_path = reports_dir / HYPERPARAMETER_SENSITIVITY_STAGE1_COMPARISON_LEGACY_JSON_FILE_NAME
        report_specs = {
            "stage1_comparison": stage1_path,
            "performance": reports_dir / HYPERPARAMETER_SENSITIVITY_STAGE2_DOWNSTREAM_PERFORMANCE_SUMMARY_FILE_NAME,
            "representational": (
                reports_dir / HYPERPARAMETER_SENSITIVITY_STAGE2_REPRESENTATIONAL_STABILITY_SUMMARY_FILE_NAME
            ),
            "functional": reports_dir / HYPERPARAMETER_SENSITIVITY_STAGE2_FUNCTIONAL_STABILITY_SUMMARY_FILE_NAME,
            "downstream_tuning": (
                reports_dir / HYPERPARAMETER_SENSITIVITY_STAGE2_DOWNSTREAM_TUNING_RESULTS_FILE_NAME
            ),
        }
        for report_kind, path in report_specs.items():
            if path.suffix == ".json" and path.exists():
                try:
                    with path.open(encoding="utf-8") as f:
                        rows = json.load(f)
                except (OSError, json.JSONDecodeError) as exc:
                    warnings.warn(f"Skipping unreadable report {path}: {exc}", RuntimeWarning)
                    rows = []
                if not isinstance(rows, list):
                    rows = []
            else:
                if not path.exists():
                    rows = []
                else:
                    try:
                        with path.open(newline="", encoding="utf-8") as f:
                            rows = [dict(row) for row in csv.DictReader(f)]
                    except OSError as exc:
                        warnings.warn(f"Skipping unreadable report {path}: {exc}", RuntimeWarning)
                        rows = []

            for row in rows:
                if not isinstance(row, dict):
                    continue
                out = dict(row)
                out.setdefault("algorithm", algorithm)
                out.setdefault("dataset", dataset)
                out["source_path"] = str(path)
                loaded[report_kind].append(out)

    return loaded


def build_hyperparameter_sensitivity_comparison_tables(
    sensitivity_results: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    *,
    results_dir: Any = None,
    datasets: Any = None,
    algorithms: Any = None,
    classifier: str = LOGISTIC_REGRESSION,
    performance_metric: str = ACCURACY_SCORE,
    stability_measures: Any = None,
    measure_labels: Optional[Dict[str, str]] = None,
    tuned_config_label: str = HYPERPARAMETER_SENSITIVITY_DIMENSION_SPECIFIC_LABEL,
    include_deltas: bool = True,
    only_tuned_when_params_changed: bool = False,
    only_changed_or_improved: bool = False,
    validation_gain_threshold: float = 0.0,
    column_layout: str = "metric_first",
    stage1_columns: str = "validation_scores",
    round_digits: Optional[int] = 3,
) -> Dict[str, Any]:
    """Build one table per algorithm comparing reference and dimension-specific Stage 2 results."""
    import pandas as pd

    loaded = sensitivity_results if sensitivity_results is not None else crawl_hyperparameter_sensitivity_results(
        results_dir=results_dir,
        datasets=datasets,
        algorithms=algorithms,
    )
    column_layout = str(column_layout)
    if column_layout not in {"metric_first", "source_first"}:
        raise ValueError("column_layout must be one of {'metric_first', 'source_first'}")
    stage1_columns = str(stage1_columns)
    if stage1_columns not in {"validation_scores", "change_summary"}:
        raise ValueError("stage1_columns must be one of {'validation_scores', 'change_summary'}")
    dataset_filter = _optional_filter_set(datasets)
    algorithm_filter = _optional_filter_set(algorithms)
    if stability_measures is None:
        selected_measure_order = None
        selected_measures = None
    elif isinstance(stability_measures, str):
        selected_measure_order = [stability_measures]
        selected_measures = {stability_measures}
    else:
        selected_measure_order = [str(measure) for measure in stability_measures]
        selected_measures = set(selected_measure_order)

    stage1_by_key: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    for row in loaded.get("stage1_comparison", []):
        try:
            dimension = int(row["dimension"])
        except (KeyError, TypeError, ValueError):
            continue
        dataset = str(row.get("dataset"))
        algorithm = str(row.get("algorithm"))
        if dataset_filter is not None and dataset not in dataset_filter:
            continue
        if algorithm_filter is not None and algorithm not in algorithm_filter:
            continue
        stage1_by_key[(dataset, algorithm, dimension)] = row

    perf_by_key: Dict[Tuple[str, str, int, str], float] = {}
    for row in loaded.get("performance", []):
        if str(row.get("classifier")) != str(classifier):
            continue
        try:
            dimension = int(row["dimension"])
        except (KeyError, TypeError, ValueError):
            continue
        value = _float_from_report_value(row.get(f"{performance_metric}_mean"))
        if not np.isfinite(value):
            continue
        dataset = str(row.get("dataset"))
        algorithm = str(row.get("algorithm"))
        config_label = str(row.get("config_label"))
        if dataset_filter is not None and dataset not in dataset_filter:
            continue
        if algorithm_filter is not None and algorithm not in algorithm_filter:
            continue
        perf_by_key[(dataset, algorithm, dimension, config_label)] = value

    stability_by_key: Dict[Tuple[str, str, int, str, str], float] = {}
    all_measures: Set[str] = set()
    for report_kind in ["representational", "functional"]:
        for row in loaded.get(report_kind, []):
            if report_kind == "functional" and str(row.get("classifier")) != str(classifier):
                continue
            measure = str(row.get("measure"))
            if selected_measures is not None and measure not in selected_measures:
                continue
            try:
                dimension = int(row["dimension"])
            except (KeyError, TypeError, ValueError):
                continue
            value = _float_from_report_value(row.get("value_mean"))
            if not np.isfinite(value):
                continue
            dataset = str(row.get("dataset"))
            algorithm = str(row.get("algorithm"))
            config_label = str(row.get("config_label"))
            if dataset_filter is not None and dataset not in dataset_filter:
                continue
            if algorithm_filter is not None and algorithm not in algorithm_filter:
                continue
            all_measures.add(measure)
            stability_by_key[(dataset, algorithm, dimension, config_label, measure)] = value

    if selected_measure_order is None:
        ordered_measures = sorted(all_measures)
    else:
        ordered_measures = [measure for measure in selected_measure_order if measure in all_measures]

    metrics = [performance_metric] + [str(measure) for measure in ordered_measures]
    metric_labels = [
        measure_labels[str(metric)]
        if measure_labels and str(metric) in measure_labels
        else str(metric)
        for metric in metrics
    ]

    main_reference_cache: Dict[Tuple[str, str], Dict[str, Dict[int, Dict[str, float]]]] = {}

    def reference_values(dataset: str, algorithm: str, dimension: int) -> List[float]:
        cache_key = (dataset, algorithm)
        if cache_key not in main_reference_cache:
            main_reference_cache[cache_key] = _main_regular_hyperparameter_metric_summary_maps(
                dataset=dataset,
                algorithm=algorithm,
                classifier=str(classifier),
                performance_metric=str(performance_metric),
                stability_measures=[str(measure) for measure in ordered_measures],
                dimensions=None,
                include_variance_bands=False,
            )
        metric_maps = main_reference_cache[cache_key]
        return [
            float(metric_maps.get(str(metric), {}).get(int(dimension), {}).get("mean", np.nan))
            for metric in metrics
        ]

    table_keys = set(stage1_by_key)
    table_keys.update((dataset, algorithm, dimension) for dataset, algorithm, dimension, _ in perf_by_key)
    table_keys.update((dataset, algorithm, dimension) for dataset, algorithm, dimension, _, _ in stability_by_key)
    selected_algorithms = sorted({algorithm for _, algorithm, _ in table_keys}, key=_algorithm_alpha_sort_key)

    tables: Dict[str, Any] = {}
    for algorithm in selected_algorithms:
        algorithm_keys = sorted(
            [(dataset, dimension) for dataset, alg, dimension in table_keys if alg == algorithm],
            key=lambda item: (_dataset_alpha_sort_key(item[0]), int(item[1])),
        )
        rows: List[List[Any]] = []
        index: List[Tuple[str, int]] = []
        for dataset, dimension in algorithm_keys:
            stage1_row = stage1_by_key.get((dataset, algorithm, dimension), {})
            params_changed = _bool_from_report_value(stage1_row.get("params_changed"))
            val_gain = _float_from_report_value(stage1_row.get("validation_score_improvement"))
            anchor_validation_score = _float_from_report_value(stage1_row.get("anchor_validation_score_at_dimension"))
            dimension_specific_validation_score = _float_from_report_value(
                stage1_row.get("dimension_specific_validation_score")
            )
            has_validation_gain = np.isfinite(val_gain) and val_gain > float(validation_gain_threshold)
            if only_changed_or_improved and not (params_changed is True or has_validation_gain):
                continue

            if stage1_columns == "change_summary":
                stage1_values: List[Any] = [
                    params_changed if params_changed is not None else np.nan,
                    val_gain,
                ]
            else:
                stage1_values = [anchor_validation_score, dimension_specific_validation_score]
            standard_values = reference_values(dataset, algorithm, int(dimension))
            tuned_values = [perf_by_key.get((dataset, algorithm, dimension, tuned_config_label), np.nan)]
            tuned_values.extend(
                stability_by_key.get((dataset, algorithm, dimension, tuned_config_label, str(measure)), np.nan)
                for measure in ordered_measures
            )
            if only_tuned_when_params_changed and params_changed is False:
                tuned_values = [np.nan for _ in tuned_values]
            delta_values = [
                tuned - standard
                if np.isfinite(tuned) and np.isfinite(standard)
                else np.nan
                for standard, tuned in zip(standard_values, tuned_values)
            ]

            if column_layout == "source_first":
                row_values = stage1_values + standard_values + tuned_values
                if include_deltas:
                    row_values.extend(delta_values)
            else:
                row_values = list(stage1_values)
                for metric_idx in range(len(metrics)):
                    row_values.extend([standard_values[metric_idx], tuned_values[metric_idx]])
                    if include_deltas:
                        row_values.append(delta_values[metric_idx])
            rows.append(row_values)
            index.append((str(DATASET_RENAME_DICT.get(dataset, dataset)), int(dimension)))

        if not rows:
            continue

        if stage1_columns == "change_summary":
            column_tuples: List[Tuple[str, str]] = [("Stage 1", "Params Changed"), ("Stage 1", "Val. Gain")]
        else:
            column_tuples = [("Stage 1", "Anchor Val. Acc."), ("Stage 1", "Dimension-Specific Val. Acc.")]
        if column_layout == "source_first":
            column_tuples.extend(("Anchor", label) for label in metric_labels)
            column_tuples.extend(("Dimension-Specific", label) for label in metric_labels)
            if include_deltas:
                column_tuples.extend(("Difference", label) for label in metric_labels)
        else:
            for label in metric_labels:
                column_tuples.extend([(label, "Anchor"), (label, "Dimension-Specific")])
                if include_deltas:
                    column_tuples.append((label, "Difference"))

        table = pd.DataFrame(
            rows,
            index=pd.MultiIndex.from_tuples(index, names=["Dataset", "Dimension"]),
            columns=pd.MultiIndex.from_tuples(column_tuples),
        )
        if round_digits is not None:
            numeric_columns = table.select_dtypes(include=[np.number]).columns
            table.loc[:, numeric_columns] = table.loc[:, numeric_columns].round(int(round_digits))
        tables[algorithm] = table

    return tables


def _hyperparameter_metric_summary_maps(
    sensitivity_results: Dict[str, List[Dict[str, Any]]],
    *,
    dataset: str,
    algorithm: str,
    classifier: str,
    performance_metric: str,
    stability_measures: List[str],
    tuned_config_label: str,
) -> Dict[str, Dict[str, Dict[int, Dict[str, float]]]]:
    metric_maps: Dict[str, Dict[str, Dict[int, Dict[str, float]]]] = {
        performance_metric: {tuned_config_label: {}}
    }
    for measure in stability_measures:
        metric_maps[measure] = {tuned_config_label: {}}

    for row in sensitivity_results.get("performance", []):
        if str(row.get("dataset")) != str(dataset):
            continue
        if str(row.get("algorithm")) != str(algorithm):
            continue
        if str(row.get("classifier")) != str(classifier):
            continue
        config_label = str(row.get("config_label"))
        if config_label != str(tuned_config_label):
            continue
        try:
            dimension = int(row["dimension"])
        except (KeyError, TypeError, ValueError):
            continue
        mean = _float_from_report_value(row.get(f"{performance_metric}_mean"))
        if not np.isfinite(mean):
            continue
        metric_maps[performance_metric][config_label][dimension] = {
            "mean": mean,
            "std": _float_from_report_value(row.get(f"{performance_metric}_std")),
            "n": _float_from_report_value(row.get("num_embeddings")),
        }

    for report_kind in ["representational", "functional"]:
        for row in sensitivity_results.get(report_kind, []):
            if str(row.get("dataset")) != str(dataset):
                continue
            if str(row.get("algorithm")) != str(algorithm):
                continue
            if report_kind == "functional" and str(row.get("classifier")) != str(classifier):
                continue
            measure = str(row.get("measure"))
            if measure not in metric_maps:
                continue
            config_label = str(row.get("config_label"))
            if config_label != str(tuned_config_label):
                continue
            try:
                dimension = int(row["dimension"])
            except (KeyError, TypeError, ValueError):
                continue
            mean = _float_from_report_value(row.get("value_mean"))
            if not np.isfinite(mean):
                continue
            metric_maps[measure][config_label][dimension] = {
                "mean": mean,
                "std": _float_from_report_value(row.get("value_std")),
                "n": _float_from_report_value(row.get("num_pairs")),
            }

    return metric_maps


def _plot_summary_band_from_map(
    ax: Any,
    summary_map: Dict[int, Dict[str, float]],
    *,
    color: Any,
    alpha: float,
    variance_band_mode: str,
    lower_bound_min: Optional[float] = None,
    upper_bound_max: Optional[float] = None,
    zorder: float = 1.0,
) -> List[float]:
    points: List[Tuple[int, float, float]] = []
    for dim, summary in summary_map.items():
        mean = _float_from_report_value(summary.get("mean"))
        std = _float_from_report_value(summary.get("std"))
        if not np.isfinite(std) or std <= 0:
            spread = np.nan
        elif str(variance_band_mode) == "std":
            spread = std
        elif str(variance_band_mode) == "2std":
            spread = 2.0 * std
        elif str(variance_band_mode) == "sem":
            n = _float_from_report_value(summary.get("n"))
            spread = np.nan if not np.isfinite(n) or n <= 0 else std / np.sqrt(n)
        elif str(variance_band_mode) in {"ci95", "95ci"}:
            n = _float_from_report_value(summary.get("n"))
            spread = np.nan if not np.isfinite(n) or n <= 0 else CONFIDENCE_BAND_Z_VALUE * std / np.sqrt(n)
        else:
            raise ValueError("variance_band_mode must be one of {'std', '2std', 'sem', 'ci95'}")
        if not np.isfinite(mean) or not np.isfinite(spread):
            continue
        lower = mean - spread
        upper = mean + spread
        if lower_bound_min is not None:
            lower = max(float(lower_bound_min), lower)
        if upper_bound_max is not None:
            upper = min(float(upper_bound_max), upper)
        if np.isfinite(lower) and np.isfinite(upper):
            points.append((int(dim), float(lower), float(upper)))

    if not points:
        return []

    bound_values: List[float] = []
    for segment in _split_dim_points_at_missing_dimensions(sorted((dim, lower) for dim, lower, _ in points)):
        segment_dims = [dim for dim, _ in segment]
        point_by_dim = {dim: (lower, upper) for dim, lower, upper in points}
        lower = [point_by_dim[dim][0] for dim in segment_dims]
        upper = [point_by_dim[dim][1] for dim in segment_dims]
        ax.fill_between(segment_dims, lower, upper, color=color, alpha=alpha, linewidth=0.0, zorder=zorder)
        bound_values.extend(lower)
        bound_values.extend(upper)
    return [float(value) for value in bound_values if np.isfinite(value)]


def _plot_dimension_specific_connectors(
    ax: Any,
    *,
    reference_dim_map: Dict[int, float],
    tuned_dim_map: Dict[int, float],
    color: Any,
    linewidth: float,
    alpha: float = 0.7,
    zorder: float = 2.8,
    allowed_reference_dimensions: Optional[Set[int]] = None,
) -> None:
    """Connect dimension-specific line segments back to neighboring reference points."""
    reference_dims = sorted(int(dim) for dim in reference_dim_map)
    if allowed_reference_dimensions is not None:
        allowed_reference_dimensions = {int(dim) for dim in allowed_reference_dimensions}
        reference_dims = [dim for dim in reference_dims if dim in allowed_reference_dimensions]
    if not reference_dims or not tuned_dim_map:
        return

    tuned_points = _sorted_finite_dim_points(tuned_dim_map)
    for segment in _split_dim_points_at_missing_dimensions(tuned_points):
        if not segment:
            continue
        first_dim, first_value = segment[0]
        last_dim, last_value = segment[-1]
        previous_dims = [dim for dim in reference_dims if dim < first_dim and np.isfinite(reference_dim_map[dim])]
        next_dims = [dim for dim in reference_dims if dim > last_dim and np.isfinite(reference_dim_map[dim])]
        if previous_dims:
            prev_dim = previous_dims[-1]
            ax.plot(
                [prev_dim, first_dim],
                [reference_dim_map[prev_dim], first_value],
                color=color,
                linestyle="--",
                linewidth=linewidth,
                alpha=alpha,
                zorder=zorder,
                clip_on=False,
            )
        if next_dims:
            next_dim = next_dims[0]
            ax.plot(
                [last_dim, next_dim],
                [last_value, reference_dim_map[next_dim]],
                color=color,
                linestyle="--",
                linewidth=linewidth,
                alpha=alpha,
                zorder=zorder,
                clip_on=False,
            )


def _summarize_numeric_values(values: Any) -> Dict[str, float]:
    clean = _sanitize_numeric_list(values)
    if not clean:
        return {"mean": np.nan, "std": np.nan, "n": 0.0}
    arr = np.asarray(clean, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else np.nan,
        "n": float(arr.size),
    }


def _filter_summary_map_dimensions(
    summary_map: Dict[int, Dict[str, float]],
    dimensions: Optional[Set[int]],
) -> Dict[int, Dict[str, float]]:
    if dimensions is None:
        return dict(summary_map)
    return {int(dim): summary for dim, summary in summary_map.items() if int(dim) in dimensions}


def _main_regular_hyperparameter_metric_summary_maps(
    *,
    dataset: str,
    algorithm: str,
    classifier: str,
    performance_metric: str,
    stability_measures: List[str],
    dimensions: Optional[Set[int]],
    include_variance_bands: bool,
) -> Dict[str, Dict[int, Dict[str, float]]]:
    out: Dict[str, Dict[int, Dict[str, float]]] = {}

    perf_means = crawl_downstream_accuracy_results(metric=performance_metric)
    perf_mean_map = perf_means.get(algorithm, {}).get(classifier, {}).get(dataset, {})
    perf_summary: Dict[int, Dict[str, float]] = {}
    if include_variance_bands:
        perf_raw = crawl_downstream_accuracy_raw_results(metric=performance_metric)
        for dim, values in perf_raw.get(algorithm, {}).get(classifier, {}).get(dataset, {}).items():
            perf_summary[int(dim)] = _summarize_numeric_values(values)
    for dim, mean in perf_mean_map.items():
        dim_i = int(dim)
        perf_summary.setdefault(dim_i, {"mean": float(mean), "std": np.nan, "n": np.nan})
        perf_summary[dim_i]["mean"] = float(mean)
    out[performance_metric] = _filter_summary_map_dimensions(perf_summary, dimensions)

    rep_means = crawl_results()
    rep_raw = crawl_representational_raw_results() if include_variance_bands else {}
    func_means = crawl_functional_results()
    func_raw = crawl_functional_grouped_raw_results() if include_variance_bands else {}
    for measure in stability_measures:
        summary: Dict[int, Dict[str, float]] = {}
        rep_mean_map = rep_means.get(algorithm, {}).get(measure, {}).get(dataset, {})
        if rep_mean_map:
            if include_variance_bands:
                for dim, values in rep_raw.get(algorithm, {}).get(measure, {}).get(dataset, {}).items():
                    summary[int(dim)] = _summarize_numeric_values(values)
            for dim, mean in rep_mean_map.items():
                dim_i = int(dim)
                summary.setdefault(dim_i, {"mean": float(mean), "std": np.nan, "n": np.nan})
                summary[dim_i]["mean"] = float(mean)
        else:
            func_mean_map = func_means.get(classifier, {}).get(algorithm, {}).get(measure, {}).get(dataset, {})
            if include_variance_bands:
                raw_dim_map = _functional_embedding_raw_dim_map(func_raw, classifier, algorithm, measure, dataset)
                for dim, values in raw_dim_map.items():
                    summary[int(dim)] = _summarize_numeric_values(values)
            for dim, mean in func_mean_map.items():
                dim_i = int(dim)
                summary.setdefault(dim_i, {"mean": float(mean), "std": np.nan, "n": np.nan})
                summary[dim_i]["mean"] = float(mean)
        out[measure] = _filter_summary_map_dimensions(summary, dimensions)

    return out


def plot_hyperparameter_sensitivity_grid_for_algorithm(
    algorithm: str,
    *,
    sensitivity_results: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    results_dir: Any = None,
    dataset: str = WIKIPEDIA,
    classifier: str = LOGISTIC_REGRESSION,
    performance_metric: str = ACCURACY_SCORE,
    stability_measures: Any = None,
    measure_labels: Optional[Dict[str, str]] = None,
    tuned_config_label: str = HYPERPARAMETER_SENSITIVITY_DIMENSION_SPECIFIC_LABEL,
    anchor_dimension: Optional[int] = TUNING_DEFAULT_DIMENSION,
    dimensions: Any = None,
    max_dimension_by_algorithm: Optional[Dict[str, int]] = None,
    tuned_only_when_params_changed: bool = False,
    limit_reference_to_tuned_dimensions: bool = True,
    include_variance_bands: bool = True,
    variance_band_mode: str = "std",
    connect_tuned_segments_to_reference: bool = True,
    connector_line_width: float = 0.75,
    connector_alpha: float = 0.72,
    jsd_symlog_linthresh: float = 0.0001,
    jsd_symlog_linscale: float = SYMLOG_DEFAULT_LINSCALE,
    style: Optional[Dict[str, Any]] = None,
    output_dir: Any = HYPERPARAMETER_SENSITIVITY_DEFAULT_OUTPUT_DIR,
    save: bool = True,
    show: bool = True,
    y_axis_mode: str = "zoom",
    y_axis_limits: Any = None,
    y_axis_padding: float = 0.05,
    standard_color: str = "#4d4d4d",
    tuned_color: str = "#0072B2",
) -> Any:
    """Plot main-reference-vs-dimension-specific sensitivity curves for one algorithm."""
    if stability_measures is None:
        selected_stability_measures = ["JaccardSimilarity", "AlignedCosineSimilarity", "StableCore", "JSD"]
    elif isinstance(stability_measures, str):
        selected_stability_measures = [stability_measures]
    else:
        selected_stability_measures = [str(measure) for measure in stability_measures]
    metrics = [str(performance_metric)] + selected_stability_measures

    loaded = sensitivity_results if sensitivity_results is not None else crawl_hyperparameter_sensitivity_results(
        results_dir=results_dir,
        datasets=[dataset],
        algorithms=[algorithm],
    )
    changed_by_dim: Dict[int, bool] = {}
    for row in loaded.get("stage1_comparison", []):
        if str(row.get("dataset")) != str(dataset) or str(row.get("algorithm")) != str(algorithm):
            continue
        try:
            dimension = int(row["dimension"])
        except (KeyError, TypeError, ValueError):
            continue
        changed = _bool_from_report_value(row.get("params_changed"))
        if changed is not None:
            changed_by_dim[dimension] = changed
    metric_maps = _hyperparameter_metric_summary_maps(
        loaded,
        dataset=str(dataset),
        algorithm=str(algorithm),
        classifier=str(classifier),
        performance_metric=str(performance_metric),
        stability_measures=selected_stability_measures,
        tuned_config_label=str(tuned_config_label),
    )
    if dimensions is None:
        selected_dimensions = {
            int(dim)
            for dim in globals().get("EXPERIMENTS_DIMENSIONS_LIST", [])
            if str(dim).strip()
        }
    elif isinstance(dimensions, (str, int)):
        selected_dimensions = {int(dimensions)}
    else:
        selected_dimensions = {int(dim) for dim in dimensions}

    dimension_caps = {str(VERSE): 512, "verse": 512}
    if max_dimension_by_algorithm:
        dimension_caps.update({str(key): int(value) for key, value in max_dimension_by_algorithm.items()})
    max_dimension = dimension_caps.get(str(algorithm))
    if max_dimension is not None:
        selected_dimensions = {dim for dim in selected_dimensions if dim <= int(max_dimension)}
    reference_metric_maps = _main_regular_hyperparameter_metric_summary_maps(
        dataset=str(dataset),
        algorithm=str(algorithm),
        classifier=str(classifier),
        performance_metric=str(performance_metric),
        stability_measures=selected_stability_measures,
        dimensions=selected_dimensions,
        include_variance_bands=include_variance_bands,
    )
    for metric in metrics:
        metric_maps.setdefault(metric, {})
        metric_maps[metric][str(tuned_config_label)] = _filter_summary_map_dimensions(
            metric_maps.get(metric, {}).get(str(tuned_config_label), {}),
            selected_dimensions,
        )

    available_metrics = [
        metric
        for metric in metrics
        if reference_metric_maps.get(metric)
        or metric_maps.get(metric, {}).get(str(tuned_config_label))
    ]
    if not available_metrics:
        warnings.warn(
            f"No hyperparameter-sensitivity rows found for {algorithm}/{dataset}/{classifier}.",
            RuntimeWarning,
        )
        return None

    cfg = _resolve_lineplot_paper_style(
        {
            "max_cols": 2,
            "figure_width": 6.5,
            "subplot_width": 3.0,
            "subplot_height": 1.8,
            "row_height_pad": 0.52,
            "tight_layout_bottom": 0.06,
            "tight_layout_top": 0.91,
            "hyperparameter_suptitle_y": 0.905,
            "hyperparameter_legend_position": "side",
            "hyperparameter_legend_bbox": (0.71, 0.24),
            "hyperparameter_show_y_label": False,
            **(style or {}),
        }
    )
    fig, row_axes = _create_centered_algorithm_axes(len(available_metrics), cfg)
    fig.patch.set_facecolor(str(cfg["figure_facecolor"]))
    manual_y_limits_by_panel = (
        _resolve_manual_y_limits(y_axis_limits, len(available_metrics)) if y_axis_mode == "manual" else None
    )
    if y_axis_mode not in {"fixed", "zoom", "manual"}:
        raise ValueError("y_axis_mode must be one of {'fixed', 'zoom', 'manual'}")

    axis_idx = 0
    for axes_in_row in row_axes:
        for col_idx, ax in enumerate(axes_in_row):
            metric = available_metrics[axis_idx]
            axis_idx += 1
            ax.set_facecolor(str(cfg["axes_facecolor"]))
            standard_map = reference_metric_maps.get(metric, {})
            tuned_map_all = metric_maps[metric].get(str(tuned_config_label), {})
            if tuned_only_when_params_changed:
                tuned_map = {
                    dim: summary
                    for dim, summary in tuned_map_all.items()
                    if changed_by_dim.get(int(dim), False)
                }
            else:
                tuned_map = dict(tuned_map_all)
            if anchor_dimension is not None:
                anchor_dim = int(anchor_dimension)
                if (
                    (selected_dimensions is None or anchor_dim in selected_dimensions)
                    and anchor_dim in standard_map
                ):
                    tuned_map[anchor_dim] = dict(standard_map[anchor_dim])
            if limit_reference_to_tuned_dimensions:
                standard_map = _filter_summary_map_dimensions(standard_map, set(tuned_map.keys()))

            standard_dim_map = {dim: summary["mean"] for dim, summary in standard_map.items()}
            tuned_dim_map = {dim: summary["mean"] for dim, summary in tuned_map.items()}
            plot_values = [
                float(value)
                for value in list(standard_dim_map.values()) + list(tuned_dim_map.values())
                if np.isfinite(float(value))
            ]
            if include_variance_bands:
                plot_values.extend(
                    _plot_summary_band_from_map(
                        ax,
                        standard_map,
                        color=standard_color,
                        alpha=CONFIDENCE_BAND_ALPHA,
                        variance_band_mode=variance_band_mode,
                        lower_bound_min=0.0,
                        upper_bound_max=1.0 if metric != "JSD" else None,
                        zorder=1.0,
                    )
                )
                plot_values.extend(
                    _plot_summary_band_from_map(
                        ax,
                        tuned_map,
                        color=tuned_color,
                        alpha=CONFIDENCE_BAND_ALPHA,
                        variance_band_mode=variance_band_mode,
                        lower_bound_min=0.0,
                        upper_bound_max=1.0 if metric != "JSD" else None,
                        zorder=1.2,
                    )
                )

            _plot_dimension_line(
                ax,
                standard_dim_map,
                marker="o",
                linewidth=float(cfg["line_width"]),
                markersize=float(cfg["marker_size"]),
                alpha=float(cfg["line_alpha"]),
                color=standard_color,
                label="Main regular results",
                clip_on=False,
                zorder=3,
            )
            if connect_tuned_segments_to_reference:
                connector_reference_dims = None
                if anchor_dimension is not None:
                    connector_reference_dims = {int(anchor_dimension)}
                _plot_dimension_specific_connectors(
                    ax,
                    reference_dim_map=standard_dim_map,
                    tuned_dim_map=tuned_dim_map,
                    color=tuned_color,
                    linewidth=float(connector_line_width),
                    alpha=float(connector_alpha),
                    zorder=3.5,
                    allowed_reference_dimensions=connector_reference_dims,
                )
            _plot_dimension_line(
                ax,
                tuned_dim_map,
                marker="o",
                linewidth=float(cfg["line_width"]) + 0.15,
                markersize=float(cfg["marker_size"]),
                alpha=1.0,
                color=tuned_color,
                label="Dimension-specific params",
                clip_on=False,
                zorder=4,
            )

            if measure_labels and metric in measure_labels:
                display_metric = str(measure_labels[metric])
            elif metric == ACCURACY_SCORE:
                display_metric = "Accuracy"
            elif metric in globals().get("REPSIM_MEASURE_RENAME_DICT", {}):
                display_metric = str(globals()["REPSIM_MEASURE_RENAME_DICT"][metric])
            elif metric in globals().get("FUNCSIM_MEASURE_RENAME_DICT", {}):
                display_metric = str(globals()["FUNCSIM_MEASURE_RENAME_DICT"][metric])
            else:
                main_rep = getattr(__main__, "REPSIM_MEASURE_RENAME_DICT", None)
                main_func = getattr(__main__, "FUNCSIM_MEASURE_RENAME_DICT", None)
                if isinstance(main_rep, dict) and metric in main_rep:
                    display_metric = str(main_rep[metric])
                elif isinstance(main_func, dict) and metric in main_func:
                    display_metric = str(main_func[metric])
                else:
                    display_metric = str(metric)
            title_text = str(display_metric)
            if bool(cfg.get("title_enumerate", True)):
                enum = _subplot_enum_label(axis_idx - 1, start=str(cfg.get("title_enum_start", "a")))
                title_text = f"({enum}) {title_text}"
            title_kwargs = {
                "fontsize": float(cfg["title_size"]),
                "color": str(cfg["text_color"]),
                "pad": float(cfg["title_pad"]),
            }
            title_y = cfg.get("title_y")
            if title_y is not None:
                title_kwargs["y"] = float(title_y)
            ax.set_title(title_text, **title_kwargs)
            ax.set_xscale("log", base=2)

            if metric == "JSD":
                default_y_axis_max = 1.0
            else:
                default_y_axis_max = 1.0
            ymin, ymax = _resolve_panel_y_limits(
                y_axis_mode=y_axis_mode,
                default_y_axis_max=default_y_axis_max,
                y_axis_padding=y_axis_padding,
                all_plot_values=plot_values,
                manual_y_limits_by_panel=manual_y_limits_by_panel,
                panel_idx=axis_idx - 1,
            )
            if metric == "JSD":
                symlog_linthresh = _validate_y_axis_symlog_linthresh(float(jsd_symlog_linthresh))
                symlog_linscale = _validate_y_axis_symlog_linscale(float(jsd_symlog_linscale))
                symlog_ymin, symlog_ymax = _resolve_symlog_y_limits(
                    0.0 if y_axis_mode != "manual" else ymin,
                    1.0 if y_axis_mode != "manual" else ymax,
                    default_y_axis_max=1.0,
                    y_axis_mode=y_axis_mode,
                )
                _apply_symlog_yaxis(
                    ax,
                    symlog_ymin,
                    symlog_ymax,
                    linthresh=symlog_linthresh,
                    linscale=symlog_linscale,
                )
            else:
                yticks = _resolve_y_ticks(
                    ymin=ymin,
                    ymax=ymax,
                    y_step=_nice_tick_step(ymax - ymin) if y_axis_mode == "zoom" else float(cfg["y_tick_step"]),
                    y_axis_mode=y_axis_mode,
                )
                ax.set_yscale("linear")
                ax.set_ylim(ymin, ymax)
                ax.set_yticks(yticks)
                ax.set_yticklabels(_format_axis_ticklabels(list(yticks)))
            ax.grid(
                True,
                which="major",
                linestyle="--",
                alpha=float(cfg["grid_alpha"]),
                color=str(cfg["grid_color"]),
            )
            ax.tick_params(
                axis="both",
                labelsize=float(cfg["tick_label_size"]),
                colors=str(cfg["text_color"]),
                direction=str(cfg["tick_direction"]),
                length=float(cfg["tick_length"]),
                width=float(cfg["tick_width"]),
                pad=float(cfg["tick_label_pad"]),
                bottom=True,
                left=True,
                top=False,
                right=False,
            )
            ax.xaxis.set_ticks_position("bottom")
            ax.yaxis.set_ticks_position("left")
            for spine in ax.spines.values():
                spine.set_color(str(cfg["spine_color"]))

            if col_idx == 0 and bool(cfg.get("hyperparameter_show_y_label", False)):
                ax.set_ylabel(
                    "Value",
                    fontsize=float(cfg["axis_label_size"]),
                    color=str(cfg["text_color"]),
                    labelpad=float(cfg["y_label_pad"]),
                )
            else:
                ax.set_ylabel("")
            ax.set_xlabel(
                "Embedding Dimension",
                fontsize=float(cfg["axis_label_size"]),
                color=str(cfg["text_color"]),
                labelpad=float(cfg["x_label_pad"]),
            )
            dims_seen = sorted(set(standard_dim_map) | set(tuned_dim_map))
            if selected_dimensions is not None and not limit_reference_to_tuned_dimensions:
                dims_seen = sorted(set(dims_seen).union(selected_dimensions))
            if dims_seen:
                ax.set_xticks(dims_seen)
                ax.set_xticklabels(
                    [str(dim) for dim in dims_seen],
                    rotation=float(cfg["x_tick_rotation"]),
                    ha=str(cfg["x_tick_ha"]),
                    rotation_mode=str(cfg["x_tick_rotation_mode"]),
                )

    display_algorithm = EMBEDDING_ALGORITHM_RENAME_DICT.get(str(algorithm), str(algorithm))
    display_dataset = DATASET_RENAME_DICT.get(str(dataset), str(dataset))
    fig.suptitle(
        f"{display_algorithm} on {display_dataset}",
        fontsize=float(cfg["title_size"]) + 1.0,
        color=str(cfg["text_color"]),
        y=float(cfg.get("hyperparameter_suptitle_y", 0.945)),
    )
    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            color=standard_color,
            marker="o",
            linewidth=float(cfg["line_width"]),
            label="Main regular results",
        ),
        plt.Line2D([0], [0], color=tuned_color, marker="o", linewidth=float(cfg["line_width"]) + 0.15, label="Dimension-specific params"),
    ]
    legend_position = str(cfg.get("hyperparameter_legend_position", "side"))
    if legend_position == "side":
        legend_bbox = cfg.get("hyperparameter_legend_bbox", (0.77, 0.19))
        fig.legend(
            handles=legend_handles,
            loc="center left",
            bbox_to_anchor=(float(legend_bbox[0]), float(legend_bbox[1])),
            ncol=1,
            frameon=False,
            fontsize=float(cfg["legend_font_size"]),
            labelcolor=str(cfg["text_color"]),
            handlelength=1.9,
            columnspacing=1.0,
            handletextpad=0.45,
        )
    elif legend_position == "bottom":
        fig.legend(
            handles=legend_handles,
            loc=str(cfg["legend_loc"]),
            bbox_to_anchor=(0.5, float(cfg["legend_bbox_y"])),
            ncol=2,
            frameon=False,
            fontsize=float(cfg["legend_font_size"]),
            labelcolor=str(cfg["text_color"]),
            handlelength=1.9,
            columnspacing=1.4,
            handletextpad=0.45,
        )
    fig.tight_layout(rect=[0, float(cfg["tight_layout_bottom"]), 1, float(cfg["tight_layout_top"])])

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "_".join(
        _sanitize_filename_token(part)
        for part in [HYPERPARAMETER_SENSITIVITY_GRID_FILE_STEM, str(dataset), str(algorithm), str(classifier)]
    )
    out_path = out_dir / f"{stem}.pdf"
    if save:
        fig.savefig(
            out_path,
            format="pdf",
            bbox_inches="tight",
            dpi=int(cfg["dpi"]),
            facecolor=str(cfg["figure_facecolor"]),
            edgecolor="none",
            transparent=False,
        )
        print(f"Saved {out_path}")
    if show:
        plt.show()
    if save:
        plt.close(fig)
        return out_path
    return fig, row_axes


def plot_hyperparameter_sensitivity_grids(
    sensitivity_results: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    *,
    results_dir: Any = None,
    dataset: str = WIKIPEDIA,
    algorithms: Any = None,
    classifier: str = LOGISTIC_REGRESSION,
    performance_metric: str = ACCURACY_SCORE,
    stability_measures: Any = None,
    measure_labels: Optional[Dict[str, str]] = None,
    output_dir: Any = HYPERPARAMETER_SENSITIVITY_DEFAULT_OUTPUT_DIR,
    save: bool = True,
    show: bool = True,
    **plot_kwargs: Any,
) -> Dict[str, Any]:
    """Create one sensitivity grid per algorithm."""
    loaded = sensitivity_results if sensitivity_results is not None else crawl_hyperparameter_sensitivity_results(
        results_dir=results_dir,
        datasets=[dataset],
        algorithms=algorithms,
    )
    algorithm_filter = _optional_filter_set(algorithms)
    available_algorithms = sorted(
        {
            str(row.get("algorithm"))
            for kind in ["performance", "representational", "functional", "stage1_comparison"]
            for row in loaded.get(kind, [])
            if str(row.get("dataset")) == str(dataset)
            and (algorithm_filter is None or str(row.get("algorithm")) in algorithm_filter)
        },
        key=_algorithm_alpha_sort_key,
    )
    outputs: Dict[str, Any] = {}
    for algorithm in available_algorithms:
        result = plot_hyperparameter_sensitivity_grid_for_algorithm(
            algorithm,
            sensitivity_results=loaded,
            dataset=str(dataset),
            classifier=str(classifier),
            performance_metric=str(performance_metric),
            stability_measures=stability_measures,
            measure_labels=measure_labels,
            output_dir=output_dir,
            save=save,
            show=show,
            **plot_kwargs,
        )
        if result is not None:
            outputs[algorithm] = result
    return outputs


def crawl_results() -> Dict[str, Dict[str, Dict[str, Dict[int, float]]]]:
    """Load representational stability results and aggregate means per dimension."""
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for algo_dir in Path(STABILITY_RESULTS_DIR).iterdir():
        if not algo_dir.is_dir():
            continue

        algo = algo_dir.name

        for dataset_dir in algo_dir.iterdir():
            if not dataset_dir.is_dir():
                continue

            dataset = dataset_dir.name
            json_path = dataset_dir / "regular" / "stability_results_representational.json"
            if not json_path.exists():
                continue

            with open(json_path) as f:
                raw = json.load(f)

            for sim_measure, dim_dict in raw.items():
                for dim, values in dim_dict.items():
                    vals = _sanitize_numeric_list(values)
                    if not vals:
                        continue

                    dim = int(dim)
                    mean_sim = float(np.mean(vals))
                    results[algo][sim_measure][dataset][dim] = mean_sim

    return results


def crawl_synth_results() -> Dict[str, Dict[str, Dict[str, Dict[int, Dict[float, Dict[int, float]]]]]]:
    """Load synthetic representational stability and aggregate mean similarity per dim/config."""
    synth_results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))

    for algo_dir in Path(STABILITY_RESULTS_DIR).iterdir():
        if not algo_dir.is_dir():
            continue
        algo = algo_dir.name

        for dataset_dir in algo_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            dataset = dataset_dir.name
            if dataset not in SYNTHETIC_DATASET_LIST:
                continue

            for config_dir in dataset_dir.iterdir():
                if not config_dir.is_dir():
                    continue

                parsed = _parse_synth_parent_dir_name(config_dir.name)
                if parsed is None:
                    continue
                num_nodes, density = parsed

                measure_dim_values = defaultdict(lambda: defaultdict(list))

                for graph_dir in config_dir.iterdir():
                    if not graph_dir.is_dir():
                        continue

                    json_path = graph_dir / STABILITY_RESULTS_JSON_FILE_NAME(REPRESENTATIONAL)
                    if not json_path.exists():
                        continue

                    with open(json_path) as f:
                        raw = json.load(f)

                    for sim_measure, dim_dict in raw.items():
                        if not isinstance(dim_dict, dict):
                            continue
                        for dim, values in dim_dict.items():
                            try:
                                dim_i = int(dim)
                            except (TypeError, ValueError):
                                continue
                            vals = _sanitize_numeric_list(values)
                            if vals:
                                measure_dim_values[sim_measure][dim_i].extend(vals)

                for sim_measure, dim_map in measure_dim_values.items():
                    for dim, vals in dim_map.items():
                        if not vals:
                            continue
                        mean_sim = float(np.nanmean(vals)) if not np.all(np.isnan(vals)) else np.nan
                        if np.isnan(mean_sim):
                            continue
                        synth_results[algo][sim_measure][dataset].setdefault(int(num_nodes), {})
                        synth_results[algo][sim_measure][dataset][int(num_nodes)].setdefault(float(density), {})
                        synth_results[algo][sim_measure][dataset][int(num_nodes)][float(density)][int(dim)] = mean_sim

    return synth_results


def crawl_synth_raw_results() -> Dict[str, Dict[str, Dict[str, Dict[int, Dict[float, Dict[int, List[float]]]]]]]:
    """Load synthetic representational stability as raw value lists per dimension/config."""
    synth_raw = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))

    for algo_dir in Path(STABILITY_RESULTS_DIR).iterdir():
        if not algo_dir.is_dir():
            continue
        algo = algo_dir.name

        for dataset_dir in algo_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            dataset = dataset_dir.name
            if dataset not in SYNTHETIC_DATASET_LIST:
                continue

            for config_dir in dataset_dir.iterdir():
                if not config_dir.is_dir():
                    continue

                parsed = _parse_synth_parent_dir_name(config_dir.name)
                if parsed is None:
                    continue
                num_nodes, density = parsed

                measure_dim_values = defaultdict(lambda: defaultdict(list))
                for graph_dir in config_dir.iterdir():
                    if not graph_dir.is_dir():
                        continue

                    json_path = graph_dir / STABILITY_RESULTS_JSON_FILE_NAME(REPRESENTATIONAL)
                    if not json_path.exists():
                        continue

                    with open(json_path) as f:
                        raw = json.load(f)

                    for sim_measure, dim_dict in raw.items():
                        if not isinstance(dim_dict, dict):
                            continue
                        for dim, values in dim_dict.items():
                            try:
                                dim_i = int(dim)
                            except (TypeError, ValueError):
                                continue
                            vals = _sanitize_numeric_list(values)
                            if vals:
                                measure_dim_values[sim_measure][dim_i].extend(vals)

                for sim_measure, dim_map in measure_dim_values.items():
                    for dim, vals in dim_map.items():
                        if not vals:
                            continue
                        synth_raw[algo][sim_measure][dataset].setdefault(int(num_nodes), {})
                        synth_raw[algo][sim_measure][dataset][int(num_nodes)].setdefault(float(density), {})
                        synth_raw[algo][sim_measure][dataset][int(num_nodes)][float(density)][int(dim)] = vals

    return synth_raw


def crawl_downstream_accuracy_results(
    train_seed: int = EXPERIMENTS_DEFAULT_SEED,
    metric: str = ACCURACY_SCORE,
) -> Dict[str, Dict[str, Dict[str, Dict[int, float]]]]:
    """Load downstream results and aggregate mean metric values per dimension."""
    perf_results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for json_path in Path(DOWNSTREAM_RESULTS_DIR).rglob(DOWNSTREAM_PERFORMANCE_JSON_FILE_NAME):
        dim_dir = json_path.parent.name
        clf_name = json_path.parent.parent.name
        split_name = json_path.parent.parent.parent.name
        dataset = json_path.parent.parent.parent.parent.name
        algo = json_path.parent.parent.parent.parent.parent.name

        if split_name != "regular" or not dim_dir.startswith("dim_"):
            continue

        try:
            dim = int(dim_dir.split("_", 1)[1])
        except ValueError:
            continue

        with open(json_path) as f:
            raw = json.load(f)

        metric_dict = raw.get(metric, {})
        values = []
        for _, train_seed_map in metric_dict.items():
            if not isinstance(train_seed_map, dict):
                continue
            val = train_seed_map.get(str(train_seed), train_seed_map.get(train_seed))
            if val is None:
                continue
            values.append(float(val))

        if values:
            perf_results[algo][clf_name][dataset][dim] = float(np.mean(values))

    return perf_results


def crawl_downstream_accuracy_samples_results(
    train_seed: int = EXPERIMENTS_DEFAULT_SEED,
    metric: str = ACCURACY_SCORE,
) -> Dict[str, Dict[str, Dict[str, Dict[int, List[float]]]]]:
    """Load downstream results and keep all metric samples per dimension."""
    perf_results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for json_path in Path(DOWNSTREAM_RESULTS_DIR).rglob(DOWNSTREAM_PERFORMANCE_JSON_FILE_NAME):
        dim_dir = json_path.parent.name
        clf_name = json_path.parent.parent.name
        split_name = json_path.parent.parent.parent.name
        dataset = json_path.parent.parent.parent.parent.name
        algo = json_path.parent.parent.parent.parent.parent.name

        if split_name != "regular" or not dim_dir.startswith("dim_"):
            continue

        try:
            dim = int(dim_dir.split("_", 1)[1])
        except ValueError:
            continue

        with open(json_path) as f:
            raw = json.load(f)

        metric_dict = raw.get(metric, {})
        values: List[float] = []
        for _, train_seed_map in metric_dict.items():
            if not isinstance(train_seed_map, dict):
                continue
            val = train_seed_map.get(str(train_seed), train_seed_map.get(train_seed))
            if val is None:
                continue
            try:
                fval = float(val)
            except (TypeError, ValueError):
                continue
            if np.isnan(fval):
                continue
            values.append(fval)

        if values:
            perf_results[algo][clf_name][dataset][dim] = values

    return perf_results


def crawl_functional_results() -> Dict[str, Dict[str, Dict[str, Dict[str, Dict[int, float]]]]]:
    """Load functional stability results and aggregate means per dimension."""
    func_results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))

    for algo_dir in Path(STABILITY_RESULTS_DIR).iterdir():
        if not algo_dir.is_dir():
            continue

        algo = algo_dir.name

        for dataset_dir in algo_dir.iterdir():
            if not dataset_dir.is_dir():
                continue

            dataset = dataset_dir.name
            json_path = dataset_dir / "regular" / STABILITY_RESULTS_JSON_FILE_NAME(FUNCTIONAL)
            if not json_path.exists():
                continue

            with open(json_path) as f:
                raw = json.load(f)

            for clf_name, measure_dict in raw.items():
                for measure, dim_dict in measure_dict.items():
                    for dim, values in dim_dict.items():
                        dim = int(dim)

                        if isinstance(values, list):
                            arr = np.asarray(values, dtype=float)
                            if arr.size == 0 or np.all(np.isnan(arr)):
                                continue
                            mean_val = float(np.nanmean(arr))
                        else:
                            try:
                                mean_val = float(values)
                            except (TypeError, ValueError):
                                continue
                            if np.isnan(mean_val):
                                continue

                        func_results[clf_name][algo][measure][dataset][dim] = mean_val

    return func_results


def identify_peak_and_plateau_dims(
    dim_to_score: Dict[int, float],
    *,
    relative_tolerance: Optional[float] = None,
    absolute_tolerance: float = 0.01,
    min_plateau_size: int = 1,
) -> Dict[str, Any]:
    """Return best dimension and a near-best plateau based on score tolerance."""
    clean_items = []
    for dim, score in dim_to_score.items():
        try:
            d = int(dim)
            s = float(score)
        except (TypeError, ValueError):
            continue
        if np.isnan(s):
            continue
        clean_items.append((d, s))

    if not clean_items:
        return {"best_dim": None, "best_score": np.nan, "plateau_dims": []}

    best_dim, best_score = max(clean_items, key=lambda x: x[1])
    rel_margin = abs(best_score) * max(relative_tolerance, 0.0) if relative_tolerance is not None else 0.0
    abs_margin = max(absolute_tolerance, 0.0)
    threshold = best_score - max(rel_margin, abs_margin)

    plateau_dims = sorted({d for d, s in clean_items if s >= threshold})
    if len(plateau_dims) < max(1, min_plateau_size):
        ranked_dims = [d for d, _ in sorted(clean_items, key=lambda x: x[1], reverse=True)]
        plateau_dims = sorted(set(ranked_dims[: max(1, min_plateau_size)]))

    return {"best_dim": best_dim, "best_score": best_score, "plateau_dims": plateau_dims}


def build_downstream_peak_plateau_map(
    perf_results: Optional[dict] = None,
    *,
    train_seed: int = EXPERIMENTS_DEFAULT_SEED,
    classifier_name: Optional[str] = None,
    relative_tolerance: Optional[float] = None,
    absolute_tolerance: float = 0.01,
    min_plateau_size: int = 1,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Aggregate downstream best/near-best dimensions per algorithm and dataset.

    If ``classifier_name`` is provided, only that downstream classifier is used
    when selecting the best and near-best dimensions. By default, values are
    aggregated across all available downstream classifiers.
    """
    perf_results = perf_results or crawl_downstream_accuracy_results(train_seed=train_seed)
    output: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)

    if classifier_name is not None:
        available_classifiers = sorted({clf for algo_map in perf_results.values() for clf in algo_map.keys()})
        if classifier_name not in available_classifiers:
            raise ValueError(f"Unknown classifier_name {classifier_name!r}. Available: {available_classifiers}")

    for algo, clf_map in perf_results.items():
        selected_clf_map = clf_map
        if classifier_name is not None:
            if classifier_name not in clf_map:
                continue
            selected_clf_map = {classifier_name: clf_map[classifier_name]}

        datasets = sorted({ds for ds_map in selected_clf_map.values() for ds in ds_map.keys()})
        for dataset in datasets:
            by_dim_values: Dict[int, List[float]] = defaultdict(list)
            for _, ds_map in selected_clf_map.items():
                for dim, score in ds_map.get(dataset, {}).items():
                    try:
                        by_dim_values[int(dim)].append(float(score))
                    except (TypeError, ValueError):
                        continue

            if not by_dim_values:
                continue

            mean_by_dim = {dim: float(np.mean(vals)) for dim, vals in by_dim_values.items() if vals}
            summary = identify_peak_and_plateau_dims(
                mean_by_dim,
                relative_tolerance=relative_tolerance,
                absolute_tolerance=absolute_tolerance,
                min_plateau_size=min_plateau_size,
            )
            output[algo][dataset] = summary

    return output


def build_downstream_peak_plateau_map_statistical(
    perf_results: Optional[dict] = None,
    *,
    train_seed: int = EXPERIMENTS_DEFAULT_SEED,
    classifier_name: Optional[str] = None,
    metric: str = ACCURACY_SCORE,
    alpha: float = 0.05,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Aggregate downstream best/statistically-indistinguishable dimensions."""
    perf_results = perf_results or crawl_downstream_accuracy_samples_results(train_seed=train_seed, metric=metric)
    output: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)

    if classifier_name is not None:
        available_classifiers = sorted({clf for algo_map in perf_results.values() for clf in algo_map.keys()})
        if classifier_name not in available_classifiers:
            raise ValueError(f"Unknown classifier_name {classifier_name!r}. Available: {available_classifiers}")

    for algo, clf_map in perf_results.items():
        selected_clf_map = clf_map
        if classifier_name is not None:
            if classifier_name not in clf_map:
                continue
            selected_clf_map = {classifier_name: clf_map[classifier_name]}

        datasets = sorted({ds for ds_map in selected_clf_map.values() for ds in ds_map.keys()})
        for dataset in datasets:
            by_dim_values: Dict[int, List[float]] = defaultdict(list)
            for _, ds_map in selected_clf_map.items():
                for dim, values in ds_map.get(dataset, {}).items():
                    try:
                        int_dim = int(dim)
                    except (TypeError, ValueError):
                        continue
                    if isinstance(values, (list, tuple, np.ndarray)):
                        value_iter = values
                    else:
                        value_iter = [values]
                    for value in value_iter:
                        try:
                            fval = float(value)
                        except (TypeError, ValueError):
                            continue
                        if np.isnan(fval):
                            continue
                        by_dim_values[int_dim].append(fval)

            if not by_dim_values:
                continue

            output[algo][dataset] = _identify_statistical_peak_and_plateau_dims(by_dim_values, alpha=alpha)

    return output


def crawl_representational_raw_results() -> Dict[str, Dict[str, Dict[str, Dict[int, List[float]]]]]:
    """Load representational results as raw value lists per dimension."""
    rep_raw = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for algo_dir in Path(STABILITY_RESULTS_DIR).iterdir():
        if not algo_dir.is_dir():
            continue
        algo = algo_dir.name

        for dataset_dir in algo_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            dataset = dataset_dir.name
            json_path = dataset_dir / "regular" / STABILITY_RESULTS_JSON_FILE_NAME(REPRESENTATIONAL)
            if not json_path.exists():
                continue
            with open(json_path) as f:
                raw = json.load(f)

            for measure, dim_dict in raw.items():
                for dim, values in dim_dict.items():
                    vals = _sanitize_numeric_list(values)
                    if vals:
                        rep_raw[algo][measure][dataset][int(dim)] = vals
    return rep_raw


def crawl_downstream_accuracy_raw_results(
    train_seed: int = EXPERIMENTS_DEFAULT_SEED,
    metric: str = ACCURACY_SCORE,
) -> Dict[str, Dict[str, Dict[str, Dict[int, List[float]]]]]:
    """Load downstream metrics as raw value lists per dimension."""
    perf_raw = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for json_path in Path(DOWNSTREAM_RESULTS_DIR).rglob(DOWNSTREAM_PERFORMANCE_JSON_FILE_NAME):
        dim_dir = json_path.parent.name
        clf_name = json_path.parent.parent.name
        split_name = json_path.parent.parent.parent.name
        dataset = json_path.parent.parent.parent.parent.name
        algo = json_path.parent.parent.parent.parent.parent.name

        if split_name != "regular" or not dim_dir.startswith("dim_"):
            continue
        try:
            dim = int(dim_dir.split("_", 1)[1])
        except ValueError:
            continue

        with open(json_path) as f:
            raw = json.load(f)

        metric_dict = raw.get(metric, {})
        vals = []
        for _, train_seed_map in metric_dict.items():
            if not isinstance(train_seed_map, dict):
                continue
            val = train_seed_map.get(str(train_seed), train_seed_map.get(train_seed))
            if val is None:
                continue
            vals.append(float(val))

        vals = _sanitize_numeric_list(vals)
        if vals:
            perf_raw[algo][clf_name][dataset][dim] = vals

    return perf_raw


def crawl_functional_grouped_raw_results() -> (
    Dict[str, Dict[str, Dict[str, Dict[str, Dict[int, Dict[str, List[float]]]]]]]
):
    """Load functional raw results and group by source (embedding/control runs)."""
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))

    for algo_dir in Path(STABILITY_RESULTS_DIR).iterdir():
        if not algo_dir.is_dir():
            continue
        algo = algo_dir.name

        for dataset_dir in algo_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            dataset = dataset_dir.name
            regular_dir = dataset_dir / "regular"

            _ingest_functional_file(
                regular_dir / STABILITY_RESULTS_JSON_FILE_NAME(FUNCTIONAL), "embedding", grouped, algo, dataset
            )
            _ingest_functional_file(
                regular_dir / FUNCSIM_CLF_CONTROL_RESULTS_JSON_FILE_NAME, "clf_seed_control", grouped, algo, dataset
            )
            _ingest_functional_file(
                regular_dir / FUNCSIM_NEGATIVE_SAMPLING_CONTROL_RESULTS_JSON_FILE_NAME,
                "neg_sampling_control",
                grouped,
                algo,
                dataset,
            )

    return grouped


def report_leaf_value_status() -> List[tuple]:
    """Inspect representational result leaves for empty lists and NaNs."""
    issues = []
    total = 0

    for algo_dir in Path(STABILITY_RESULTS_DIR).iterdir():
        if not algo_dir.is_dir():
            continue
        algo = algo_dir.name

        for dataset_dir in algo_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            dataset = dataset_dir.name

            json_path = dataset_dir / "regular" / "stability_results_representational.json"
            if not json_path.exists():
                continue

            with open(json_path) as f:
                raw = json.load(f)

            for sim_measure, dim_dict in raw.items():
                for dim, values in dim_dict.items():
                    total += 1
                    arr = np.asarray(values, dtype=float) if len(values) > 0 else np.asarray([])

                    status = []
                    if len(values) == 0:
                        status.append("EMPTY_LIST")
                    if arr.size > 0 and np.isnan(arr).any():
                        status.append("CONTAINS_NAN")

                    if status:
                        issues.append((algo, dataset, sim_measure, int(dim), ",".join(status), len(values)))

    print(f"Checked {total} leaf lists.")
    print(f"Problematic leaves: {len(issues)}")

    if not issues:
        print("No empty lists and no NaNs found in leaf lists.")
    else:
        print("algo | dataset | similarity | dim | status | list_len")
        for row in sorted(issues):
            print(" | ".join(map(str, row)))

    return issues


def build_script_lines(issues: List[Any], n_jobs: int) -> List[str]:
    """Build SBATCH shell lines to rerun problematic representational jobs."""
    lines = [
        "#!/bin/bash",
        "#SBATCH --partition=cpu-single",
        "#SBATCH --nodes=1",
        "#SBATCH --cpus-per-task=32",
        "#SBATCH --time=120:00:00",
        "#SBATCH --mem=1536gb",
        "#SBATCH --output=jobs/repsim_leaf_issues_%j.out",
        "module load devel/miniforge/",
        "source $MINIFORGE_HOME/etc/profile.d/conda.sh",
        "conda activate dimpact",
    ]
    for issue in issues:
        if isinstance(issue, dict):
            algorithm = issue["algorithm"]
            dataset = issue["dataset"]
            dimension = issue["dimension"]
            measure = issue.get("similarity", "jaccard")
        else:
            algorithm, dataset, measure, dimension = issue[:4]

        lines.append(
            "srun python -m stability.representational "
            f"-a {algorithm} -d {dataset} -m {measure} -dim {dimension} --n_jobs {n_jobs} --overwrite"
        )
    return lines
