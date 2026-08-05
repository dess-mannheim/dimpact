# Case Studies

This folder contains publication case-study scripts. Run them from the repository root with module syntax, for example:

```bash
python -m case_studies.embedding_costs ...
```

Case-study outputs are written under the regular output directory:

```text
output/<case_study_name>/
```

## Embedding Costs

`embedding_costs.py` measures wall-clock runtime and process RSS memory for embedding generation. By default, it also runs one downstream evaluation pass with tuned downstream classifier parameters for each produced embedding.

Example:

```bash
python -m case_studies.embedding_costs -a graphsage dgi -d Cora -dim 64 128 -n 3 -c LogisticRegression --n_jobs 4
```

Outputs are written to:

```text
output/embedding_costs/<run_id>/
```

Main report files:

- `reports/embedding_costs.csv`
- `reports/embedding_costs_summary.csv`
- `reports/downstream_costs.csv`
- `reports/downstream_costs_summary.csv`
- `reports/run_metadata.json`

Use `--skip_downstream` to measure embedding computation only.

## Stability-Performance Bootstrap

`stability_performance_bootstrap.py` bootstraps whether the most stable dimension overlaps with the best or near-best downstream-performance dimensions.

Example:

```bash
python -m case_studies.stability_performance_bootstrap -a graphsage node2vec -d Cora wiki -m JaccardSimilarity SecondOrderCosineSimilarity -c LogisticRegression
```

Outputs are written to:

```text
output/stability_performance_bootstrap/
```

For each algorithm/dataset/classifier/measure combination, the script writes a summary JSON and a bootstrap-row CSV. Existing outputs are skipped unless `--overwrite` is supplied.

## Hyperparameter Sensitivity

`hyperparameter_sensitivity.py` tests whether dimension-specific embedding hyperparameters change performance or stability relative to the main regular results.

Example:

```bash
python -m case_studies.hyperparameter_sensitivity -a graphsage -d Cora -dim 4 8 16 32 64 128 --stage2_tune_logreg
```

Outputs are written to:

```text
output/hyperparameter_sensitivity/<algorithm>/<dataset>/.../
```

Use `--collect_existing_stage2_results` to rebuild merged reports from already computed Stage 2 artifacts.

## MinGE

`min_ge.py` computes MinGE and related feature-entropy quantities used in the publication tables.

Example:

```bash
python -m case_studies.min_ge --table
```
