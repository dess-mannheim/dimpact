# The Impact of Dimensionality on Node Embedding Stability and Performance

This repository is the companion repository for the paper
["The Impact of Dimensionality on the Stability of Node Embeddings"](https://arxiv.org/abs/2604.08492).
It contains the reference code for reproducing experiments on how node embedding dimensionality affects downstream 
performance, representational stability across random seeds, and functional stability across downstream predictions.

Below we describe the design of the repository and how to rerun our experiments.

## 1. Scope

This repository contains scripts to reproduce the experiment pipeline used in the paper:

- tuning embedding hyperparameters,
- training embeddings across dimensions and seeds,
- evaluating downstream tasks,
- computing representational and functional stability summaries.

The runnable embedding methods are:

- `graphsage`
- `dgi`
- `node2vec`
- `verse`
- `asne`

The empirical datasets used in the reproducibility scripts are:

- `Cora`
- `PubMed`
- `wiki`
- `facebook`
- `blogcatalog`
- `ogbl_ddi`
- `coauthor`

The synthetic graph families used in the reproducibility scripts are:

- `barabasi-albert`
- `watts-strogatz`

The downstream classifiers used in the reproducibility scripts are:

- `LogisticRegression`
- `MLP`

Experiment-wide defaults and path helpers are centralized in `paths_globals.py`.

## 2. Repository Structure

The main directories group model implementations, dataset handling, experiment utilities, and stability measures:

- `configs/`  
  Default per-model training hyperparameters.

- `data/`  
  Dataset code plus prepared dataset artifacts used by the pipeline.

- `envs/`  
  Conda environment specifications for the main environment and method-specific environments.

- `models/`  
  Embedding implementations grouped by framework/source (`pyg`, `grape`, `karateclub`, `verse`), plus downstream classifier utilities.

- `min_ge/`  
  Code for estimating optimal embedding dimensions according to the [MinGE method](https://doi.org/10.24963/ijcai.2021/381).

- `stability/measures/`  
  Representational and functional similarity measures.

- `tools/`  
  Shared utilities for data loading, configuration, tuning selection, embedding loading, and analysis helpers.

Experiment-wide constants for names, defaults, and path construction are centralized in `paths_globals.py`.
The main root-level scripts are the executable workflow entry points; they are introduced in the order they are used below.

## 3. Workflow

All workflow scripts expose their full argument list through `--help`. In the examples below, `-dim` is shown only when a restricted dimension subset is useful. If omitted, tuning uses the default tuning dimension, while training and stability scripts use the default experiment dimension grid from `paths_globals.py`. Seed defaults are also defined centrally and do not need to be passed explicitly to reproduce the default runs.

### Step A - Tune Embedding Hyperparameters

Run `tune_embeddings.py` before empirical training so `train.py` can load best-known parameters.

Example:

```bash
python tune_embeddings.py -a graphsage -d Cora
```

Tuning summaries are written under `output/embeddings/.../tune/.../tuning_results.json`.

### Step B - Train Empirical Embeddings

Run `train.py` with one or more algorithms, datasets, and dimensions.

Example:

```bash
python train.py -a graphsage -d Cora --n_jobs 4
```

Embeddings are written to `output/embeddings/<algorithm>/<dataset>/.../stability_analysis/dim_<d>/`.

### Step C - Train Synthetic Embeddings

Synthetic graph experiments use `train_synth_embeddings.py`.

Example:

```bash
python train_synth_embeddings.py -a graphsage -d watts-strogatz --n_jobs 4
```

### Step D - Run Downstream Tasks

Evaluate embeddings for downstream performance and generate prediction files used in functional stability.

Example:

```bash
python run_downstream_tasks.py -a graphsage -d Cora -c LogisticRegression MLP --n_jobs 4
```

Results are written under `output/downstream_results/...`.

### Step E - Compute Representational Stability

Example:

```bash
python stability/representational.py -a graphsage -d Cora --n_jobs 4
```

Results are written to `output/stability_results/<algorithm>/<dataset>/.../stability_results_representational.json`.

### Step F - Compute Functional Stability

Example:

```bash
python stability/functional.py -a graphsage -d Cora -c LogisticRegression MLP --n_jobs 4
```

Results are written to `output/stability_results/<algorithm>/<dataset>/.../stability_results_functional.json`.

## 4. Setup Notes

- `envs/dimpact.yml` records the reference environment used for the main code path. It should be treated as a reproducibility reference; on other machines or future package distributions, individual dependency versions may need adjustment.
- `node2vec` is launched through a separate `grape` environment, and `asne` through a separate `karateclub` environment. YAML files for both method-specific environments are provided in `envs/`.
- `verse` uses C++ sources under `models/verse/src/`, which need to be compiled before running VERSE experiments. The included code follows the reference implementation at <https://github.com/xgfs/verse>.
- First dataset load may trigger dataset preparation or download, depending on the dataset.
- Large sweeps can be CPU/RAM intensive; use `--n_jobs` conservatively.

## Citation

If you use this repository or build on the experiments, please cite:

```bibtex
@online{schumacher_impact_2026,
  title = {The Impact of Dimensionality on the Stability of Node Embeddings},
  author = {Schumacher, Tobias and Reichelt, Simon and Strohmaier, Markus},
  eprint = {2604.08492},
  eprinttype = {arXiv},
  doi = {10.48550/arXiv.2604.08492},
  url = {https://arxiv.org/abs/2604.08492}
}
```
