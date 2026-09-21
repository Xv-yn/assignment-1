# COMP5328 Assignment 1 - Robust NMF Experiments

This project implements and compares Non-negative Matrix Factorization (NMF)
algorithms on the ORL and Extended YaleB face datasets under occlusion noise.

## Implemented Algorithms

- `F-norm MU`: standard Frobenius-norm NMF with multiplicative updates.
- `L1-NMF`: robust L1-norm NMF using reweighted multiplicative updates.
- `L2,1-NMF`: robust column-wise L2,1-norm NMF using reweighted multiplicative
  updates.

The algorithm implementations are in `code/algorithm/`.

## Environment Setup

Use Python 3, then install the required packages:

```bash
pip install -r code/requirements.txt
```

The NMF algorithms are implemented with NumPy. `scikit-learn` is only used for
optional clustering evaluation metrics.

## Data Layout

The experiment scripts expect the data root to contain both datasets:

```text
data/
  ORL/
  CroppedYaleB/
```

This repository also supports the nested layout:

```text
data/data/
  ORL/
  CroppedYaleB/
```

You can pass either layout with `--data-root`; the loader will resolve it
automatically.

## Run Experiments

Run commands from the `code/` directory.

### ORL: Vary Occlusion Block Size

```bash
python run_orl_blocksize_experiment.py --data-root ../data/data
```

### ORL: Vary Number of Occlusion Blocks

```bash
python run_orl_blockcount_experiment.py --data-root ../data/data
```

### Extended YaleB: Vary Occlusion Block Size

```bash
python run_yaleb_blocksize_experiment.py --data-root ../data/data
```

### Extended YaleB: Vary Number of Occlusion Blocks

```bash
python run_yaleb_blockcount_experiment_fast.py --data-root ../data/data
```

Each experiment writes raw CSV files, summary CSV files, and RRE plots to
`code/results/` by default.

## Optional Clustering Metrics

By default, scripts compute Relative Reconstruction Error (RRE). To also compute
accuracy and normalized mutual information (NMI), add:

```bash
--include-clustering
```

Example:

```bash
python run_orl_blocksize_experiment.py --data-root ../data/data --include-clustering
```

## Generate Noise Demonstrations

To create original-vs-occluded example images for ORL and Extended YaleB:

```bash
python make_noise_demo.py --data-root ../data/data
```

The outputs are saved to:

```text
code/results/orl_noise_demo.png
code/results/yaleb_noise_demo.png
```

## Existing Results

The `code/results/` folder contains the current CSV summaries and plots for:

- ORL block-size experiment.
- ORL block-count experiment.
- Extended YaleB block-size experiment.
- Extended YaleB block-count experiment.
- Noise demonstration images.
