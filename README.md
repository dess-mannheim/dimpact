# The Impact of Dimensionality on Node Embedding Stability and Performance

This repository contains experiments for studying how embedding dimensionality affects:

- **downstream performance** (e.g., node classification, link prediction),
- **representational stability** (how similar embeddings are across random seeds),
- **functional stability** (how similar downstream predictions are across random seeds).

> Note: SDNE/HOPE training code lives in an external repository (`Implementation_pytorch_GEM`).
> This repo still contains analysis tooling and result handling for SDNE/HOPE outputs.

---

## 1. What this project does

The project runs a full pipeline over multiple embedding methods and datasets:

1. tune embedding hyperparameters,
2. train embeddings for many dimensions and seeds,
3. run downstream tasks,
4. compute representational and functional stability,
5. generate plots from stored results.

Supported embedding methods in the current code include:

- `graphsage`
- `dgi`
- `node2vec`
- `verse`
- `asne`

Core experiment defaults are centrally defined in `paths_globals.py` (dimensions, iteration counts, tasks, naming conventions, paths).

---

## 2. Repository structure (current)

- `train.py`  
  Main training entrypoint for embeddings across datasets/dimensions.

- `tune_embeddings.py`  
  Hyperparameter tuning for embedding methods (grid search + summary persistence).

- `run_downstream_tasks.py`  
  Tunes/evaluates downstream classifiers and stores predictions/performance.

- `stability/representational.py`  
  Computes representational similarity scores between embedding pairs.

- `stability/functional.py`  
  Computes functional similarity scores between downstream prediction pairs.

- `models/`  
  Embedding implementations grouped by framework/source (`pyg`, `grape`, `karateclub`, `verse`, `gem`).

- `tools/`  
  Shared utilities for data loading, configuration, tuning selection, helper scripts.

- `evaluation/`  
  Plotting/report scripts that consume saved JSON outputs.

- `configs/defaults.json`  
  Default per-model training hyperparameters.

- `data/`  
  Raw/downloaded datasets and derived per-dataset artifacts.

- `output/`  
  Generated embeddings, downstream results, and stability results.

---

## 3. High-level workflow

## Step A — Tune embedding hyperparameters

Run `tune_embeddings.py` first so the training stage can use best-known parameters.

Example:

```bash
python tune_embeddings.py -a graphsage -d Cora -dim 128
```

Tuning summaries are written under `output/embeddings/.../tune/.../tuning_results.json`.

## Step B — Train embeddings over dimensions

Run `train.py` with one or more algorithms/datasets/dimensions.

Example:

```bash
python train.py -a graphsage -d Cora -dim 4 8 16 32 64 128 --n_jobs 4
```

Embeddings are written to `output/embeddings/<algorithm>/<dataset>/.../stability_analysis/dim_<d>/`.

## Step C — Run downstream tasks

Evaluate embeddings for performance and generate prediction files used in functional stability.

Example:

```bash
python run_downstream_tasks.py -a graphsage -d Cora -dim 4 8 16 32 64 128 --n_jobs 4
```

Results are written under `output/downstream_results/...`.

## Step D — Compute representational stability

Example:

```bash
python stability/representational.py -a graphsage -d Cora -dim 4 8 16 32 64 128 --n_jobs 4
```

Results are written to `output/stability_results/<algorithm>/<dataset>/.../stability_results_representational.json`.

## Step E — Compute functional stability

Example:

```bash
python stability/functional.py -a graphsage -d Cora -c LogisticRegression MLP -dim 4 8 16 32 64 128 --n_jobs 4
```

Results are written to `output/stability_results/<algorithm>/<dataset>/.../stability_results_functional.json`.

---

## 4. Important concepts to understand first

If you are new to this codebase, learn in this order:

1. **`paths_globals.py`**
   - dataset/model names,
   - experiment defaults (dimensions, iteration counts),
   - task mapping per dataset,
   - output path construction functions.

2. **`tools/data_utils.py`**
   - how datasets are loaded,
   - how downstream split files are generated,
   - empirical vs synthetic dataset flow.

3. **`train.py`**
   - orchestration over seeds/dimensions,
   - method-specific training branches,
   - model/embedding persistence.

4. **`run_downstream_tasks.py` + `stability/*.py`**
   - how performance and stability are actually measured,
   - where predictions and summary JSON files are consumed.

---

## 5. Environment notes

- The code expects Python packages from the project environment (PyTorch, PyG, scikit-learn, etc.).
- Some embedding methods are launched through separate environments via subprocess logic.
- First dataset load may trigger automatic download depending on the dataset.

---

## 6. Common output locations

- Embeddings: `output/embeddings/`
- Downstream metrics/predictions: `output/downstream_results/`
- Stability analysis JSON files: `output/stability_results/`
- Plots: usually under `plots/` (depending on plotting script)

---

## 7. Reproducibility tips

- Keep algorithm, dataset, dimension list, and seed-related defaults fixed when comparing runs.
- Run tuning before large training sweeps.
- Verify that downstream predictions exist before running functional stability.
- For large sweeps, use `--n_jobs` conservatively to avoid CPU/RAM oversubscription.

