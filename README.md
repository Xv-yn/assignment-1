# COMP4328/5328/8328 — Assignment 1: Robustness of NMF under occlusion noise

Five Non-negative Matrix Factorization algorithms implemented from scratch,
and an experimental study of how each one holds up when face images are
corrupted by contiguous white occlusion blocks.

> **Marking note.** The brief applies a **−20 penalty** for not including
> instructions on how to run the code. The instructions are below, and are
> repeated in Appendix B of the report.

---

## 1. Repository layout

```
code/
  algorithm/              the library — NumPy only for the factorisations
    common.py             shared majorise–minimise / multiplicative-update kernel
    nmf_mu.py             1. Frobenius-norm NMF            (taught in the course)
    nmf_l1.py             2. L1-norm NMF                   (not taught)
    nmf_hypersurface.py   3. Hypersurface / pseudo-Huber   (not taught)
    nmf_l21.py            4. L2,1-norm NMF                 (not taught)
    nmf_l1_reg.py         5. L1-regularised robust NMF     (not taught)
    noise.py              occlusion noise: block size, block count, contamination ratio
    data_io.py            ORL / Extended YaleB loaders
    evaluate.py           RRE, clustering accuracy, NMI  (evaluation only)
  data/                   EMPTY — put ORL/ and CroppedYaleB/ here
  results/                CSVs and figures written by the scripts
  run_experiment.py       the robustness grid (3 noise axes × 5 algorithms)
  run_convergence.py      objective vs iteration
  run_sensitivity.py      δ, c, λ and rank sweeps
  make_figures.py         noise demo, reconstructions, learned bases
  make_tables.py          turns the summary CSVs into LaTeX tables
  run_all.sh              reproduces every number and figure in the report
  test_algorithms.py      self-checks (descent, recovery, robustness, noise)
report/
  report.tex              the report (12 pt Times, 14 pt title, per the brief)
  references.bib
  Makefile                `make` → report.pdf, `make pages` → page count
```

## 2. Requirements

Python 3.9 or newer.

```bash
pip install -r code/requirements.txt
```

**Library policy** (the brief restricts what may be used to implement NMF):

| Package | Where it is used | Allowed because |
|---|---|---|
| numpy | everywhere | explicitly permitted |
| scipy | `evaluate.py` only (Hungarian matching) | evaluation only |
| scikit-learn | `evaluate.py` only (K-means, accuracy, NMI) | evaluation only, as the brief permits |
| pillow | `data_io.py` only (decode `.pgm`) | data loading, as in `assignment1.ipynb` |
| matplotlib | plotting scripts only | figures |

No NMF implementation from any external library is used anywhere. The five
algorithms in `algorithm/nmf_*.py` and `algorithm/common.py` import nothing
beyond NumPy and the standard library.

## 3. Data

The datasets are **not** committed to this repository. Copy the two folders in
so that these paths exist:

```
code/data/ORL/s1/1.pgm ...
code/data/CroppedYaleB/yaleB01/... 
```

Expected after loading: ORL = 400 images / 40 subjects; Extended YaleB = 2414
images / 38 subjects (38 subject folders — `yaleB01`–`yaleB13` and
`yaleB15`–`yaleB39`, there is no `yaleB14` — each contributing 64 captures
plus one `*Ambient.pgm` reference frame that is skipped).

`--data-root` accepts either the folder containing `ORL/` and `CroppedYaleB/`
or its parent, so `--data-root data` and `--data-root ../data` both work.

## 4. Running

Reproduce everything:

```bash
cd code
bash run_all.sh                 # ORL ≈ 20 min, Extended YaleB ≈ 3 h
FAST=1 bash run_all.sh          # reduced grid, a few minutes — smoke test
DATA_ROOT=../data bash run_all.sh   # if the data lives elsewhere
```

Or run one piece at a time (every script takes `--help`):

```bash
cd code
python run_experiment.py  --dataset orl   --data-root data
python run_experiment.py  --dataset yaleb --data-root data --max-iter 250
python run_convergence.py --dataset orl   --data-root data
python run_sensitivity.py --study delta --dataset orl --data-root data
python run_sensitivity.py --study scale --dataset orl --data-root data
python run_sensitivity.py --study lam   --dataset orl --data-root data
python run_sensitivity.py --study rank  --dataset orl --data-root data
python make_figures.py    --dataset orl   --data-root data
python make_tables.py                       # results/*.csv → report/tables/*.tex
```

Everything is seeded (`--seed`, default `2026`), so reruns reproduce the
reported numbers exactly.

Check the implementations before trusting any of it (no pytest needed, runs in
under a minute, needs no dataset):

```bash
cd code && python test_algorithms.py
```

It asserts non-negativity, monotone descent of each objective (the property
the majorise-minimise derivations guarantee, and the check most likely to
catch a wrong weight formula), exact recovery of a synthetic rank-k matrix,
that the robust losses really do beat the baseline on corrupted data, that the
noise generator corrupts exactly the images and pixels it claims, and that
every algorithm is deterministic under a fixed seed.

Build the report:

```bash
cd report && make          # → report.pdf
make pages                 # check the 20-page limit
```

## 5. What each experiment varies

| Axis | Values | Held fixed | Question it answers |
|---|---|---|---|
| block size `b` | 0, 5, 10, 15, 20 px | 1 block, all images | how large an occlusion can each loss absorb? |
| block count | 0, 1, 2, 3, 4 | `b`=10, all images | does damage depend only on pixel count, or also on its layout? |
| contamination ratio | 0.1, 0.25, 0.5, 0.75, 1.0 | `b`=15, 2 blocks | what happens when only *some* images are corrupted? |

The third axis is not in the brief, and it is the one that makes the
comparison meaningful. Axes 1–2 keep every image equally damaged, so a
**per-image** robust loss such as L2,1 sees near-identical column residuals,
its weights `d_j = 1/‖r_j‖₂` become near-uniform, and it degenerates
numerically into the Frobenius baseline. Only axis 3 creates the uneven
damage that a column-wise loss can actually exploit.

Metrics: RRE (required), clustering accuracy and NMI (optional, both
reported). Protocol: independent uniform 90 % subsample, 5 repeats, mean ± sd.
Within a repeat every algorithm sees the identical subsample and the identical
corrupted matrix.

## 6. Submission checklist

- [ ] `report.pdf` built from LaTeX (Word is not accepted), 10–15 pages, ≤ 20
- [ ] Group ID and every member's name / SID / unikey on the title page
- [ ] Contribution of each member stated (Appendix A)
- [ ] Instructions on how to run the code in the report appendix (Appendix B)
- [ ] `code/data/` present but **empty** in the zip
- [ ] All `\todo{}` markers removed from the report
- [ ] Zip named `GroupIDXX_StudentIDxxxxxxxx_StudentIDxxxxxxxx_....zip`
- [ ] Submitted before **8 October 2026** (−20 % per day late, 5 days maximum)
