import os
import os.path as osp

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score  # , precision_score
from sklearn.neural_network import MLPClassifier
from typing import Dict, List, Literal, Final, Tuple, get_args, Any, Callable

# DATASET TYPES
DATASET = Literal[
    "wiki",
    "Cora",
    "blogcatalog",
    "coauthor",
    "PubMed",
    "facebook",
    "ogbl_ddi",
    "barabasi-albert",
    "watts-strogatz",
]

WIKIPEDIA: Final[DATASET] = "wiki"
CORA: Final[DATASET] = "Cora"
BLOGCATALOG: Final[DATASET] = "blogcatalog"
COAUTHOR: Final[DATASET] = "coauthor"
PUBMED: Final[DATASET] = "PubMed"
FACEBOOK: Final[DATASET] = "facebook"
DDI: Final[DATASET] = "ogbl_ddi"
BARABASI_ALBERT: Final[DATASET] = "barabasi-albert"
WATTS_STROGATZ: Final[DATASET] = "watts-strogatz"

DATASET_LIST: Final[List[DATASET]] = list(get_args(DATASET))
EMPIRICAL_DATASET_LIST: Final[List[DATASET]] = [
    CORA,
    PUBMED,
    WIKIPEDIA,
    FACEBOOK,
    BLOGCATALOG,
    DDI,
    COAUTHOR,
]
DATASET_RENAME_DICT: Final[Dict[DATASET, str]] = {
    CORA: "Cora",
    PUBMED: "PubMed",
    WIKIPEDIA: "Wikipedia",
    BLOGCATALOG: "BlogCatalog",
    FACEBOOK: "Facebook",
    DDI: "OGBL-DDI",
    COAUTHOR: "CoAuthor",
}


SYNTHETIC_DATASET_LIST: Final[List[DATASET]] = [
    WATTS_STROGATZ,
    BARABASI_ALBERT,
]

PLANETOID_DATASETS: Final[List[DATASET]] = [CORA, PUBMED]

SUBSAMPLED_DATA_DIR_NAME: Final[str] = "sampled"

DATA_EDGE_LIST_DEFAULT_FILE_NAME: Final[str] = "edges.txt"
DATA_FEATURE_MATRIX_DEFAULT_FILE_NAME: Final[str] = "features.npy"

# EMBEDDINGS ALGORITHMS
EMBEDDING_ALGORITHM = Literal["graphsage", "node2vec", "dgi", "verse", "asne"]

GRAPHSAGE: Final[EMBEDDING_ALGORITHM] = "graphsage"
NODE2VEC: Final[EMBEDDING_ALGORITHM] = "node2vec"
DGI: Final[EMBEDDING_ALGORITHM] = "dgi"
VERSE: Final[EMBEDDING_ALGORITHM] = "verse"
ASNE: Final[EMBEDDING_ALGORITHM] = "asne"

EMBEDDING_ALGORITHM_RENAME_DICT: Final[Dict[EMBEDDING_ALGORITHM, str]] = {
    GRAPHSAGE: "GraphSAGE",
    NODE2VEC: NODE2VEC,
    DGI: "DGI",
    VERSE: "VERSE",
    ASNE: "ASNE",
}

EMBEDDING_ALGORITHM_LIST: Final[List[EMBEDDING_ALGORITHM]] = list(get_args(EMBEDDING_ALGORITHM))

DEFAULT_ENVIRONMENT_NAME: Final[str] = "dimpact"
GRAPE_ENVIRONMENT_NAME: Final[str] = "grape"
KARATECLUB_ENVIRONMENT_NAME: Final[str] = "karateclub"

ENVIRONMENTS_DICT: Final[Dict[EMBEDDING_ALGORITHM, str]] = {
    NODE2VEC: GRAPE_ENVIRONMENT_NAME,
    GRAPHSAGE: DEFAULT_ENVIRONMENT_NAME,
    DGI: DEFAULT_ENVIRONMENT_NAME,
    VERSE: DEFAULT_ENVIRONMENT_NAME,
    ASNE: KARATECLUB_ENVIRONMENT_NAME,
}

MODULE_NAME_DICT: Final[Dict[EMBEDDING_ALGORITHM, str]] = {
    NODE2VEC: "models.grape.node2vec",
    DGI: "models.pyg.dgi_inductive",
    GRAPHSAGE: "models.pyg.graphsage",
    VERSE: "models.verse.verse",
    ASNE: "models.karateclub.asne",
}

STABILITY_TYPE = Literal["representational", "functional"]

REPRESENTATIONAL: Final[STABILITY_TYPE] = "representational"
FUNCTIONAL: Final[STABILITY_TYPE] = "functional"

# ----------------------------------------------------------------------------------------------------------------------
# EXPERIMENT-RELATED VARIABLES
# ----------------------------------------------------------------------------------------------------------------------

# GENERAL EXPERIMENT VARIABLES
EXPERIMENTS_DIMENSIONS_LIST: Final[List[int]] = [2**i for i in range(2, 13)]
EXPERIMENTS_NUM_ITERATIONS: Final[int] = 30
EXPERIMENTS_NUM_DOWNSTREAM_CLF_RUNS: Final[int] = 5
EXPERIMENTS_NUM_CLF_CONTROL_EMBEDDINGS: Final[int] = 5
EXPERIMENTS_NUM_SYNTH_ITERATIONS: Final[int] = 10
EXPERIMENTS_DEFAULT_SEED: Final[int] = 2025

# KEYS FOR CONFIG FILES:
CONFIG_EMBEDDING_NAME_KEY: Final[str] = "embedding"
CONFIG_DATASET_NAME_KEY: Final[str] = "dataset_name"
CONFIG_DATA_SUBSAMPLING_INDICATOR_KEY: Final[str] = "snowball_sample"
CONFIG_DATA_SUBSAMPLING_RATIO_KEY: Final[str] = "sampling_ratio"
CONFIG_SYNTH_DATA_NUM_NODES_KEY: Final[str] = "num_nodes"
CONFIG_SYNTH_DATA_DENSITY_KEY: Final[str] = "density"
CONFIG_DATA_SAMPLING_SEED_KEY: Final[str] = "sampling_seed"
CONFIG_DIMENSION_KEY: Final[str] = "dimension"
CONFIG_ITERATIONS_KEY: Final[str] = "iterations"
CONFIG_TRAINING_SEEDS_KEY: Final[str] = "training_seeds"

# KEYS FOR DOWNSTREAM TASK METADATA FILES:
DATASET_METADATA_TASK_KEY: Final[str] = "task"
DATASET_METADATA_NUM_CLASSES_KEY: Final[str] = "num_classes"
DATASET_METADATA_CLASS_LABELS_KEY: Final[str] = "class_labels"

# SYNTHETIC DATA EXPERIMENT CONFIGURATIONS
SYNTH_DATA_EXPERIMENTS_NUM_SEEDS: Final[int] = 10
WATTS_STROGATZ_DEFAULT_REWIRING_PROBABILITY: Final[float] = 0.1
SYNTH_DATA_EXPERIMENTS_DEFAULT_NUM_NODES: Final[int] = 1600
SYNTH_DATA_EXPERIMENTS_DEFAULT_DENSITY: Final[float] = 0.01
SYNTH_DATA_EXPERIMENTS_NUM_NODES_LIST: Final[List[int]] = [
    200,
    400,
    800,
    1600,
    3200,
    6400,
    12800,
]
SYNTH_DATA_EXPERIMENTS_DENSITIES_LIST: Final[List[float]] = [
    0.001,
    0.002,
    0.005,
    0.01,
    0.02,
    0.05,
    0.1,
]


# PARAMETER GRIDS FOR TUNING OF EMBEDDINGS
TUNING_PARAM_GRID_DICT: Final[Dict[EMBEDDING_ALGORITHM, Dict[str, List[Any]]]] = {
    NODE2VEC: {
        "walk_length": [50, 80, 100],
        "context_size": [5, 10, 20],
        "p": [0.5, 1, 2],
        "q": [0.5, 1, 2],
    },
    DGI: {"num_layers": [2, 3, 4, 5]},
    GRAPHSAGE: {"num_layers": [2, 3, 4, 5]},
    VERSE: {
        "alpha": [0.7, 0.8, 0.85, 0.9],
        "n_neg_samples": [1, 2, 3, 4, 5],
    },
    ASNE: {"down_sampling": [0, 1e-5, 1e-4, 1e-3]},
}

DATASET_SPECIFIC_PARAM_DICT: Dict[EMBEDDING_ALGORITHM, Dict[DATASET, Dict[str, Any]]] = {
    method: {dataset: dict() for dataset in EMPIRICAL_DATASET_LIST} for method in EMBEDDING_ALGORITHM_LIST
}
DATASET_SPECIFIC_PARAM_DICT[DGI][COAUTHOR]["max_sampled_nodes_per_batch"] = 100_000


SYNTH_DATASET_SPECIFIC_PARAM_DICT: Dict[EMBEDDING_ALGORITHM, Dict[DATASET, Dict[int, Dict[str, Any]]]] = {
    method: {
        dataset: {num_nodes: dict() for num_nodes in SYNTH_DATA_EXPERIMENTS_NUM_NODES_LIST}
        for dataset in SYNTHETIC_DATASET_LIST
    }
    for method in EMBEDDING_ALGORITHM_LIST
}
SYNTH_DATASET_SPECIFIC_PARAM_DICT[DGI][WATTS_STROGATZ][12800]["max_sampled_nodes_per_batch"] = 100_000

# TUNING SUMMARY CAN BE DONE BOTH FOR EMBEDDINGS AND FOR DOWNSTREAM CLASSIFIERS
TUNING_DEFAULT_DIMENSION: Final[int] = 128
TUNING_DEFAULT_ITERATIONS: Final[int] = 5
TUNING_SUMMARY_FILE_NAME: Final[str] = "tuning_results.json"
TUNING_SUMMARY_PARAMS_KEY: Final[str] = "params"
TUNING_SUMMARY_RESULTS_KEY: Final[str] = "results"
TUNING_SUMMARY_SCORE_KEY: Final[str] = "avg_performance"

# FUNCTIONAL PERFORMANCE AND STABILITY
DOWNSTREAM_TASK = Literal["node_classification", "graph_reconstruction", "link_prediction"]
NODE_CLASSIFICATION: Final[DOWNSTREAM_TASK] = "node_classification"
LINK_PREDICTION: Final[DOWNSTREAM_TASK] = "link_prediction"
GRAPH_RECONSTRUCTION: Final[DOWNSTREAM_TASK] = "graph_reconstruction"
DOWNSTREAM_TASKS: Final[List[DOWNSTREAM_TASK]] = list(get_args(DOWNSTREAM_TASK))

DATASET_TASK_DICT: Final[Dict[DATASET, DOWNSTREAM_TASK]] = {
    CORA: NODE_CLASSIFICATION,
    PUBMED: NODE_CLASSIFICATION,
    WIKIPEDIA: NODE_CLASSIFICATION,
    BLOGCATALOG: NODE_CLASSIFICATION,
    FACEBOOK: NODE_CLASSIFICATION,
    COAUTHOR: LINK_PREDICTION,
    DDI: LINK_PREDICTION,
    BARABASI_ALBERT: GRAPH_RECONSTRUCTION,
    WATTS_STROGATZ: GRAPH_RECONSTRUCTION,
}

DOWNSTREAM_CLASSIFIER = Literal["LogisticRegression", "MLP"]

LOGISTIC_REGRESSION: Final[DOWNSTREAM_CLASSIFIER] = "LogisticRegression"
MULTILAYER_PERCEPTRON: Final[DOWNSTREAM_CLASSIFIER] = "MLP"

DOWNSTREAM_CLASSIFIERS: Final[List[DOWNSTREAM_CLASSIFIER]] = list(get_args(DOWNSTREAM_CLASSIFIER))

DOWNSTREAM_PERFORMANCE_MEASURE = Literal["accuracy", "micro_f1", "macro_f1"]  # , "precision"]
ACCURACY_SCORE: Final[DOWNSTREAM_PERFORMANCE_MEASURE] = "accuracy"
MICRO_F1_SCORE: Final[DOWNSTREAM_PERFORMANCE_MEASURE] = "micro_f1"
MACRO_F1_SCORE: Final[DOWNSTREAM_PERFORMANCE_MEASURE] = "macro_f1"

DOWNSTREAM_PERFORMANCE_MEASURES: Final[List[DOWNSTREAM_PERFORMANCE_MEASURE]] = list(
    get_args(DOWNSTREAM_PERFORMANCE_MEASURE)
)
DOWNSTREAM_MEASURE_DICT: Final[Dict[DOWNSTREAM_PERFORMANCE_MEASURE, Callable[[np.ndarray, np.ndarray], float]]] = {
    ACCURACY_SCORE: accuracy_score,
    MICRO_F1_SCORE: lambda yt, yp: f1_score(yt, yp, average="micro"),
    MACRO_F1_SCORE: lambda yt, yp: f1_score(yt, yp, average="macro"),
}

# downstream runtime defaults
DOWNSTREAM_PREDICTION_BATCH_SIZE_DEFAULT: Final[int] = 200_000
DOWNSTREAM_LP_TRAIN_BATCH_SIZE_DEFAULT: Final[int] = 100_000
DOWNSTREAM_LP_CACHE_DISABLE_DIMENSION_THRESHOLD_DEFAULT: Final[int] = 1000


NODE_CLASSIFICATION_DEFAULT_TRAINING_RATIO: Final[float] = 0.7
NODE_CLASSIFICATION_DEFAULT_VALIDATION_RATIO: Final[float] = 0.1

GRAPH_RECONSTRUCTION_DEFAULT_TRAINING_RATIO: Final[float] = 0.4
GRAPH_RECONSTRUCTION_DEFAULT_VALIDATION_RATIO: Final[float] = 0.3


DOWNSTREAM_TASK_DATA_TRAIN_CATEGORY: Final[str] = "train"
DOWNSTREAM_TASK_DATA_VAL_CATEGORY: Final[str] = "val"
DOWNSTREAM_TASK_DATA_TEST_CATEGORY: Final[str] = "test"
DOWNSTREAM_TASK_DATA_SPLIT_COL_KEY: Final[str] = "split"
DOWNSTREAM_TASK_DATA_LABEL_COL_KEY: Final[str] = "label"
DOWNSTREAM_TASK_DATA_SRC_EDGE_COL_KEY: Final[str] = "u"
DOWNSTREAM_TASK_DATA_DEST_EDGE_COL_KEY: Final[str] = "v"
DOWNSTREAM_TASK_DATA_LP_COLUMN_NAMES: Final[List[str]] = [
    DOWNSTREAM_TASK_DATA_SRC_EDGE_COL_KEY,
    DOWNSTREAM_TASK_DATA_DEST_EDGE_COL_KEY,
    DOWNSTREAM_TASK_DATA_SPLIT_COL_KEY,
    DOWNSTREAM_TASK_DATA_LABEL_COL_KEY,
]
DOWNSTREAM_TASK_DATA_NC_COLUMN_NAMES: Final[List[str]] = [
    DOWNSTREAM_TASK_DATA_SPLIT_COL_KEY,
    DOWNSTREAM_TASK_DATA_LABEL_COL_KEY,
]

# Parameters for classifiers
LR_BASE_PARAMS: Final[Dict[str, Any]] = {"max_iter": 1000}
LR_TUNING_PARAMS: Final[Dict[str, List[float]]] = {"C": [10.0**i for i in range(-8, 6, 1)]}

MLP_BASE_PARAMS: Final[Dict[str, Any]] = {"max_iter": 500}
MLP_TUNING_PARAMS: Final[Dict[str, List[float]]] = {"alpha": [10.0**i for i in range(-8, 4, 1)]}

# Parameters for classifiers to tune
DOWNSTREAM_CLASSIFIER_DICT_CLF_KEY: Final[str] = "clf"
DOWNSTREAM_CLASSIFIER_DICT_BASE_PARAMS_KEY: Final[str] = "base_params"
DOWNSTREAM_CLASSIFIER_DICT_TUNING_PARAMS_KEY: Final[str] = "tuning_params"

DOWNSTREAM_CLASSIFIER_DICT: Final[Dict[DOWNSTREAM_CLASSIFIER, Dict[str, Any]]] = {
    LOGISTIC_REGRESSION: {
        DOWNSTREAM_CLASSIFIER_DICT_CLF_KEY: LogisticRegression,
        DOWNSTREAM_CLASSIFIER_DICT_BASE_PARAMS_KEY: LR_BASE_PARAMS,
        DOWNSTREAM_CLASSIFIER_DICT_TUNING_PARAMS_KEY: LR_TUNING_PARAMS,
    },
    MULTILAYER_PERCEPTRON: {
        DOWNSTREAM_CLASSIFIER_DICT_CLF_KEY: MLPClassifier,
        DOWNSTREAM_CLASSIFIER_DICT_BASE_PARAMS_KEY: MLP_BASE_PARAMS,
        DOWNSTREAM_CLASSIFIER_DICT_TUNING_PARAMS_KEY: MLP_TUNING_PARAMS,
    },
}


MLP_LAYER_DICT: Final[Dict[int, Tuple[int, ...]]] = {
    4: (4,),
    8: (8,),
    16: (12,),
    32: (24,),
    64: (48,),
    128: (96,),
    256: (128,),
    512: (256,),
    1024: (256,),
    2048: (256,),
    4096: (512,),
    8192: (1024,),
}

# ADDITIONAL STATISTICS FOR TUNING SUMMARY OF DOWNSTREAM CLASSIFIERS
# TUNING_SUMMARY_ACCURACY_SCORE_KEY: Final[str] = "accuracy"
# TUNING_SUMMARY_MACRO_F1_SCORE_KEY: Final[str] = "macro_f1"
# TUNING_SUMMARY_MICRO_F1_SCORE_KEY: Final[str] = "micro_f1"
# TUNING_SUMMARY_PRECISION_SCORE_KEY: Final[str] = "precision"


MAX_DIMENSION_DICT: Dict[EMBEDDING_ALGORITHM, Dict[DATASET, int]] = {
    em: {dt: 4096 for dt in DATASET_LIST} for em in EMBEDDING_ALGORITHM_LIST
}
MAX_DIMENSION_DICT[VERSE][CORA] = 2048
MAX_DIMENSION_DICT[GRAPHSAGE][COAUTHOR] = 1024


# ----------------------------------------------------------------------------------------------------------------------
# CASE STUDY SETTINGS
# ----------------------------------------------------------------------------------------------------------------------

EMBEDDING_COSTS_CASE_STUDY_NAME: Final[str] = "embedding_costs"
HYPERPARAMETER_SENSITIVITY_CASE_STUDY_NAME: Final[str] = "hyperparameter_sensitivity"
STABILITY_PERFORMANCE_BOOTSTRAP_CASE_STUDY_NAME: Final[str] = "stability_performance_bootstrap"


LP_EDGE_FEATURE_OP = Literal["hadamard", "concat"]

EMBEDDING_COSTS_SUPPORTED_ALGORITHMS: Final[List[EMBEDDING_ALGORITHM]] = [
    GRAPHSAGE,
    NODE2VEC,
    DGI,
    VERSE,
    ASNE,
]
EMBEDDING_COSTS_MEASUREMENT_SCOPE: Final[str] = (
    "Existing repository embedding training call, including the validation scoring step performed by each model "
    "implementation after the embedding is computed."
)
EMBEDDING_COSTS_DOWNSTREAM_MEASUREMENT_SCOPE: Final[str] = (
    "Single downstream evaluation pass on one produced case-study embedding, using tuned downstream classifier "
    "hyperparameters from the main downstream-results tree."
)

HYPERPARAMETER_SENSITIVITY_EMBEDDING_TRAINING_SCOPE: Final[str] = (
    "Case-study embedding generation via the same model training entry points used by train.py, "
    "without embedding-cost memory sampling."
)

STABILITY_PERFORMANCE_BOOTSTRAP_MAX_OUTPUT_PATH_LENGTH: Final[int] = 259
STABILITY_PERFORMANCE_BOOTSTRAP_PERFORMANCE_CRITERION = Literal["strict_best", "threshold", "statistical"]
STABILITY_PERFORMANCE_BOOTSTRAP_TIE_RTOL: Final[float] = 1e-12
STABILITY_PERFORMANCE_BOOTSTRAP_TIE_ATOL: Final[float] = 1e-15


# ----------------------------------------------------------------------------------------------------------------------
# PATHS AND FILE NAMES
# ----------------------------------------------------------------------------------------------------------------------

# MAIN DIRECTORIES
MAIN_DIR: Final[str] = osp.dirname(os.path.abspath(__file__))

CONFIGS_DIR: Final[str] = osp.join(MAIN_DIR, "configs")
os.makedirs(CONFIGS_DIR, exist_ok=True)

DATA_DIR: Final[str] = osp.join(MAIN_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

OUTPUT_DIR: Final[str] = os.path.join(MAIN_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

EMBEDDINGS_DIR: Final[str] = os.path.join(OUTPUT_DIR, "embeddings")
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

DOWNSTREAM_RESULTS_DIR: Final[str] = os.path.join(OUTPUT_DIR, "downstream_results")
os.makedirs(DOWNSTREAM_RESULTS_DIR, exist_ok=True)

STABILITY_RESULTS_DIR: Final[str] = os.path.join(OUTPUT_DIR, "stability_results")
os.makedirs(STABILITY_RESULTS_DIR, exist_ok=True)

PLOTS_DIR: Final[str] = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

TABLES_DIR: Final[str] = os.path.join(OUTPUT_DIR, "tables")
os.makedirs(TABLES_DIR, exist_ok=True)

# CASE-STUDY OUTPUT DIRECTORIES
EMBEDDING_COSTS_CASE_STUDY_OUTPUT_DIR: Final[str] = osp.join(OUTPUT_DIR, EMBEDDING_COSTS_CASE_STUDY_NAME)
HYPERPARAMETER_SENSITIVITY_CASE_STUDY_OUTPUT_DIR: Final[str] = osp.join(
    OUTPUT_DIR,
    HYPERPARAMETER_SENSITIVITY_CASE_STUDY_NAME,
)
STABILITY_PERFORMANCE_BOOTSTRAP_CASE_STUDY_OUTPUT_DIR: Final[str] = osp.join(
    OUTPUT_DIR,
    STABILITY_PERFORMANCE_BOOTSTRAP_CASE_STUDY_NAME,
)

# SHARED OUTPUT SUBDIRECTORIES
REPORTS_DIR_NAME: Final[str] = "reports"
LOGS_DIR_NAME: Final[str] = "logs"
EMBEDDINGS_DIR_NAME: Final[str] = "embeddings"
CONFIGS_DIR_NAME: Final[str] = "configs"
PREDICTIONS_DIR_NAME: Final[str] = "predictions"
SUMMARIES_DIR_NAME: Final[str] = "summaries"
DOWNSTREAM_TUNING_DIR_NAME: Final[str] = "downstream_tuning"
HYPERPARAMETER_SENSITIVITY_STAGE1_RETUNING_DIR_NAME: Final[str] = "stage1_retuning"
HYPERPARAMETER_SENSITIVITY_STAGE2_STABILITY_DIR_NAME: Final[str] = "stage2_stability"

TUNE_DIR_NAME: Final[str] = "tune"
STABILITY_ANALYSIS_DIR_NAME: Final[str] = "stability_analysis"
HYPERPARAMETER_SENSITIVITY_DIMENSION_SPECIFIC_LABEL: Final[str] = "dimension_specific"

VERSE_SRC_PATH: Final[str] = osp.join(MAIN_DIR, "models", "verse", "src")
BCSR_GRAPH_FILE_NAME: Final[str] = "graph.bcsr"


def DIMENSION_SUBDIR_NAME(dimension: int) -> str:
    return f"dim_{dimension}"


def TUNE_RUN_SUBDIR_NAME(tune_id: int) -> str:
    return f"{TUNE_DIR_NAME}_{tune_id}"


# FILENAMES
CONFIG_DEFAULTS_FILE_NAME: Final[str] = "defaults.json"
CONFIG_DEFAULTS_FILE_PATH: Final[str] = osp.join(CONFIGS_DIR, CONFIG_DEFAULTS_FILE_NAME)
DOWNSTREAM_TASK_DATA_FILE_NAME: Final[str] = "downstream_task_data.csv"
DOWNSTREAM_METADATA_JSON_FILE_NAME: Final[str] = "downstream_task_metadata.json"
RUN_METADATA_FILE_NAME: Final[str] = "run_metadata.json"
RUN_METADATA_HISTORY_FILE_NAME: Final[str] = "run_metadata_history.jsonl"

EMBEDDING_RUN_LOG_FILE_TEMPLATE: Final[str] = "{algorithm}_{dataset}_dim{dimension}_s{seed}.log"

EMBEDDING_COSTS_CONFIG_FILE_TEMPLATE: Final[str] = "{algorithm}_{dataset}_dim{dimension}_s{seed}.json"
EMBEDDING_COSTS_DOWNSTREAM_LOG_FILE_TEMPLATE: Final[str] = (
    "{algorithm}_{dataset}_dim{dimension}_s{seed}_{classifier}_downstream.log"
)
EMBEDDING_COSTS_FILE_NAME: Final[str] = "embedding_costs.csv"
EMBEDDING_COSTS_SUMMARY_FILE_NAME: Final[str] = "embedding_costs_summary.csv"
DOWNSTREAM_COSTS_FILE_NAME: Final[str] = "downstream_costs.csv"
DOWNSTREAM_COSTS_SUMMARY_FILE_NAME: Final[str] = "downstream_costs_summary.csv"

HYPERPARAMETER_SENSITIVITY_BEST_PARAMS_FILE_NAME: Final[str] = "best_params.json"
HYPERPARAMETER_SENSITIVITY_OUTPUTS_FILE_TEMPLATE: Final[str] = "outputs_s{seed}.npy"
HYPERPARAMETER_SENSITIVITY_PREDICTIONS_FILE_TEMPLATE: Final[str] = "predictions_s{seed}.npy"
HYPERPARAMETER_SENSITIVITY_SCORES_FILE_TEMPLATE: Final[str] = "scores_s{seed}.json"
HYPERPARAMETER_SENSITIVITY_CLASSIFIER_PARAMS_FILE_TEMPLATE: Final[str] = "classifier_params_s{seed}.json"
HYPERPARAMETER_SENSITIVITY_STAGE1_TUNING_SUMMARY_BY_DIMENSION_FILE_NAME: Final[str] = (
    "stage1_tuning_summary_by_dimension.json"
)
HYPERPARAMETER_SENSITIVITY_STAGE1_TUNING_RESULTS_FILE_NAME: Final[str] = "stage1_tuning_results.csv"
HYPERPARAMETER_SENSITIVITY_STAGE1_COMPARISON_FILE_NAME: Final[str] = (
    "stage1_anchor_vs_dimension_specific.csv"
)
HYPERPARAMETER_SENSITIVITY_STAGE1_COMPARISON_LEGACY_JSON_FILE_NAME: Final[str] = (
    "stage1_anchor_vs_dimension_specific.json"
)
HYPERPARAMETER_SENSITIVITY_STAGE2_SELECTED_DIMENSIONS_FILE_NAME: Final[str] = "stage2_selected_dimensions.json"
HYPERPARAMETER_SENSITIVITY_STAGE2_EMBEDDING_RUNS_FILE_NAME: Final[str] = "stage2_embedding_runs.csv"
HYPERPARAMETER_SENSITIVITY_STAGE2_DOWNSTREAM_TUNING_RESULTS_FILE_NAME: Final[str] = (
    "stage2_downstream_tuning_results.csv"
)
HYPERPARAMETER_SENSITIVITY_STAGE2_DOWNSTREAM_PERFORMANCE_FILE_NAME: Final[str] = (
    "stage2_downstream_performance.csv"
)
HYPERPARAMETER_SENSITIVITY_STAGE2_REPRESENTATIONAL_STABILITY_FILE_NAME: Final[str] = (
    "stage2_representational_stability.csv"
)
HYPERPARAMETER_SENSITIVITY_STAGE2_FUNCTIONAL_STABILITY_FILE_NAME: Final[str] = (
    "stage2_functional_stability.csv"
)
HYPERPARAMETER_SENSITIVITY_STAGE2_REPRESENTATIONAL_STABILITY_SUMMARY_FILE_NAME: Final[str] = (
    "stage2_representational_stability_summary.csv"
)
HYPERPARAMETER_SENSITIVITY_STAGE2_FUNCTIONAL_STABILITY_SUMMARY_FILE_NAME: Final[str] = (
    "stage2_functional_stability_summary.csv"
)
HYPERPARAMETER_SENSITIVITY_STAGE2_DOWNSTREAM_PERFORMANCE_SUMMARY_FILE_NAME: Final[str] = (
    "stage2_downstream_performance_summary.csv"
)

STABILITY_PERFORMANCE_BOOTSTRAP_RUN_SUMMARY_FILE_TEMPLATE: Final[str] = "{stem}_summary.json"
STABILITY_PERFORMANCE_BOOTSTRAP_RUN_BOOTSTRAPS_FILE_TEMPLATE: Final[str] = "{stem}_bootstraps.csv"
STABILITY_PERFORMANCE_BOOTSTRAP_RUN_STEM_ALL_CLASSIFIERS_TOKEN: Final[str] = "all"


def TMP_TUNING_RESULTS_FILE_NAME(tune_id: int) -> str:
    if tune_id is None:
        return "results.json"
    return f"results_tid{tune_id}.json"


def SAMPLED_DATA_FILE_NAME(sampling_ratio: float, seed: int, is_edge_list: bool = False) -> str:
    file_ending = "txt" if is_edge_list else "pkl"
    return f"graph_sampled_r{int(round(100 * sampling_ratio))}_s{seed}.{file_ending}"


def MODEL_FILE_NAME(model_seed: int, tune_id: int = None) -> str:
    if tune_id is None:
        return f"model_s{model_seed}.pt"
    return f"model_tid{tune_id}_s{model_seed}.pt"


def DOWNSTREAM_PREDICTIONS_FILENAME(emb_id: int, model_seed: int, negative_sampling_seed: int = None) -> str:
    if negative_sampling_seed is None:
        return f"predictions_e{emb_id}_s{model_seed}.npy"
    else:
        return f"predictions_e{emb_id}_s{model_seed}_ns{negative_sampling_seed}.npy"


def EMBEDDING_FILE_NAME(model_seed: int, tune_id: int = None) -> str:
    if tune_id is None:
        return f"embedding_s{model_seed}.npy"
    return f"embedding_tid{tune_id}_s{model_seed}.npy"


def STABILITY_RESULTS_JSON_FILE_NAME(stability_type: STABILITY_TYPE) -> str:
    return f"stability_results_{stability_type}.json"


DOWNSTREAM_PERFORMANCE_JSON_FILE_NAME: Final[str] = "downstream_performance.json"
DOWNSTREAM_PERFORMANCE_NS_JSON_FILE_NAME: Final[str] = "downstream_performance_neg_sampling.json"
FUNCSIM_CLF_CONTROL_RESULTS_JSON_FILE_NAME: Final[str] = "stability_results_functional_clf.json"
FUNCSIM_NEGATIVE_SAMPLING_CONTROL_RESULTS_JSON_FILE_NAME: Final[str] = "stability_results_functional_ns.json"


# ADVANCED PATHS
def SYNTHETIC_DATASET_SUBDIRS(dataset_params: Dict[str, Any], parent_only: bool = False) -> str:
    d_string = str(dataset_params[CONFIG_SYNTH_DATA_DENSITY_KEY]).split(".")[1]
    if parent_only:
        return f"graphs_n{dataset_params[CONFIG_SYNTH_DATA_NUM_NODES_KEY]}_d{d_string}"

    return osp.join(
        f"graphs_n{dataset_params[CONFIG_SYNTH_DATA_NUM_NODES_KEY]}_d{d_string}",
        f"graph_s{dataset_params[CONFIG_DATA_SAMPLING_SEED_KEY]}",
    )


def SUBSAMPLING_DATASET_SUBDIR(dataset_params: Dict[str, Any]) -> str:
    return (
        f"sampled_r{100 * dataset_params[CONFIG_DATA_SUBSAMPLING_RATIO_KEY]}"
        f"_s{dataset_params[CONFIG_DATA_SAMPLING_SEED_KEY]}"
    )


def BUILD_DATASET_SRC_DIR(dataset_params: Dict[str, Any]) -> str:
    dataset_name = dataset_params[CONFIG_DATASET_NAME_KEY]
    if dataset_name in SYNTHETIC_DATASET_LIST:
        synth_model_dir = osp.join(
            DATA_DIR,
            dataset_name,
            SYNTHETIC_DATASET_SUBDIRS(dataset_params),
        )
        os.makedirs(synth_model_dir, exist_ok=True)
        return synth_model_dir
    else:
        if dataset_params[CONFIG_DATA_SUBSAMPLING_INDICATOR_KEY]:
            return osp.join(
                DATA_DIR,
                dataset_name,
                SUBSAMPLED_DATA_DIR_NAME,
                SUBSAMPLING_DATASET_SUBDIR(dataset_params),
            )
        else:
            return osp.join(DATA_DIR, dataset_name)


def BUILD_DATASET_DIR_NAME(dataset_params: Dict[str, Any]) -> str:
    dataset_name = dataset_params[CONFIG_DATASET_NAME_KEY]
    if dataset_name in SYNTHETIC_DATASET_LIST:
        return SYNTHETIC_DATASET_SUBDIRS(dataset_params)
    else:
        if dataset_params[CONFIG_DATA_SUBSAMPLING_INDICATOR_KEY]:
            return SUBSAMPLING_DATASET_SUBDIR(dataset_params)
        else:
            return "regular"


def CREATE_MODELS_PATH(
    dataset_params: Dict[str, Any],
    embedding_name: EMBEDDING_ALGORITHM,
    embedding_dim: int,
    b_tune: bool = False,
) -> str:
    dataset_name = dataset_params[CONFIG_DATASET_NAME_KEY]

    algorithm_dir: str = str(osp.join(EMBEDDINGS_DIR, embedding_name))
    dataset_dir: str = str(osp.join(algorithm_dir, dataset_name))

    dataset_dir = osp.join(dataset_dir, BUILD_DATASET_DIR_NAME(dataset_params))

    task_dir: str = TUNE_DIR_NAME if b_tune else STABILITY_ANALYSIS_DIR_NAME

    save_dir = osp.join(dataset_dir, task_dir, DIMENSION_SUBDIR_NAME(embedding_dim))
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def CREATE_SYNTH_TUNING_RESULTS_PATH(
    dataset_params: Dict[str, Any],
    embedding_name: EMBEDDING_ALGORITHM,
) -> str:
    dataset_name = dataset_params[CONFIG_DATASET_NAME_KEY]

    algorithm_dir: str = str(osp.join(EMBEDDINGS_DIR, embedding_name))
    dataset_dir: str = str(osp.join(algorithm_dir, dataset_name))

    dataset_dir = osp.join(dataset_dir, SYNTHETIC_DATASET_SUBDIRS(dataset_params, parent_only=True))

    os.makedirs(dataset_dir, exist_ok=True)
    return dataset_dir


def CREATE_DOWNSTREAM_RESULTS_PATH(
    dataset_params: Dict[str, Any],
    embedding_name: EMBEDDING_ALGORITHM,
    clf_name: DOWNSTREAM_CLASSIFIER,
    embedding_dim: int,
) -> str:
    dataset_name = dataset_params[CONFIG_DATASET_NAME_KEY]

    algorithm_dir: str = str(osp.join(DOWNSTREAM_RESULTS_DIR, embedding_name))
    dataset_dir: str = str(osp.join(algorithm_dir, dataset_name))

    dataset_dir = osp.join(dataset_dir, BUILD_DATASET_DIR_NAME(dataset_params))

    save_dir = osp.join(dataset_dir, clf_name, DIMENSION_SUBDIR_NAME(embedding_dim))
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def CREATE_STABILITY_RESULTS_PATH(dataset_params: Dict[str, Any], embedding_name: EMBEDDING_ALGORITHM) -> str:
    dataset_name = dataset_params[CONFIG_DATASET_NAME_KEY]
    save_dir = osp.join(
        STABILITY_RESULTS_DIR,
        embedding_name,
        dataset_name,
        BUILD_DATASET_DIR_NAME(dataset_params),
    )
    os.makedirs(save_dir, exist_ok=True)
    return save_dir
