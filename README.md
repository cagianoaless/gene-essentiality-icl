# Tabular In-Context Learning for Gene Essentiality Prediction Across Cancer Cell Lines

Alessandro Cagiano, Manuel Dileo, Andrea Sottoriva, Matteo Zignani

This repository contains the code and released results for predicting CRISPR gene-dependency scores in unseen cancer cell lines with tabular in-context learning. We compare TabPFN-3.5, TabICL v2 and Kumo Large on the [Tahoe-x1 DepMap benchmark](https://github.com/tahoebio/tahoe-x1/tree/167cc19fce5888c7738b0f73f61cb4a8d3d1d457/scripts/depmap), and study smaller labelled cell-line panels and a Lung lineage holdout.

## Pipeline

![Pipeline (light mode)](figures/pipeline_esann_light.svg#gh-light-mode-only)
![Pipeline (dark mode)](figures/pipeline_esann_dark.svg#gh-dark-mode-only)

**Our representation** uses 31 features per cell-line–gene pair: 15 cell-line PCA scores, 15 gene loadings and one normalized/log1p expression value. Normalization and PCA are fitted on training-cell expression, then applied unchanged to held-out expression. CRISPR dependency scores are training-context labels, not input features.

## Reproduce

Use Python 3.11 from the repository root.

**Verify included results (CPU; no biological downloads or model checkpoints):**

```sh
python -m pip install -r requirements.txt
python -m pytest -q
python evaluate.py --expected --out runs/summary
python plot.py --out runs/plots
```

The six asset-free tests check training-only PCA, the 31-feature interface, context selection and leakage guards, and evaluation. The last two commands **reaggregate archived per-unit metrics and regenerate plots**; they do not rerun model inference.

**Run a model on the benchmark (requires the registered external data):**

```sh
python download_data.py --out data/tahoe
python prepare.py --config configs/depmap.yaml --data-root data/tahoe \
  --fold 0 --stratum 60to70 --budget 8 --seed 0 --pca-mode refit --out runs/inputs
python predict.py --prepared runs/inputs/manifest.json \
  --context depmap/fold0/60to70/selection0/C8 --model rf --out runs/rf
python evaluate.py --prepared runs/inputs/manifest.json \
  --prediction runs/rf/result.json \
  --truth data/tahoe/raw/depmap-gene-dependencies.csv --out runs/metrics.json
```

The example uses Random Forest and **refits PCA on training expression**. It is not an exact replay of the archived fitted PCA. If the Tahoe download URLs are inaccessible, obtain the registered files through an authorized source, place them in `data/tahoe` and run `python download_data.py --out data/tahoe --skip-download`. Raw-file hashes and exact source URLs are in [data/sources.json](data/sources.json); historical PCA assets and their hashes are in [data/private_assets.json](data/private_assets.json). Reproducing exact historical inference requires those assets and the relevant model checkpoint.

For TabPFN/TabICL use `requirements-inference.txt`, and for Kumo use `requirements-kumo.txt`; checkpoint identities are listed in [results/main/checkpoints.json](results/main/checkpoints.json). The main benchmark configuration is [configs/main_benchmark.yaml](configs/main_benchmark.yaml).

## Results

[Main benchmark](results/main/endpoints.csv) · [DepMap cell-line sweep](figures/depmap_seed_auroc.png) · [Lung lineage holdout](figures/lung_ood_seed_auroc.png)

The headline TabPFN result uses **all eligible labelled training pairs and 8 estimators**; TabICL and Kumo use **32,768 labelled pairs and 1 estimator**. A TabPFN C32768 comparator is also reported. These pipeline-level results do not isolate architecture or pretraining effects, and published external baselines may use different query supports.

## Citation and license

See [CITATION.cff](CITATION.cff) for the manuscript citation. Original project code is [Apache-2.0](LICENSE); external data and model weights retain their providers' terms ([NOTICE.md](NOTICE.md)).
