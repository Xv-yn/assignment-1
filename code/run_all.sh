#!/usr/bin/env bash
# Reproduce every number and every figure in the report, in order.
#
# Run from the `code/` directory after copying the two dataset folders
# (ORL/ and CroppedYaleB/) into code/data/:
#
#     bash run_all.sh                 # uses code/data
#     DATA_ROOT=../data bash run_all.sh
#
# Wall clock on a 2-core cloud VM: ~20 min for ORL, ~3 h for YaleB.
# Pass FAST=1 to run a reduced grid (fewer iterations) for a quick check.
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-data}"
OUT="${OUT:-results}"
PY="${PY:-python3}"

if [[ "${FAST:-0}" == "1" ]]; then
  ITER_ORL=60;  ITER_YALE=60;  RUNS=2
else
  ITER_ORL=400; ITER_YALE=250; RUNS=5
fi

echo "=== 0/8  self-checks ==="
$PY test_algorithms.py

echo "=== 1/8  ORL qualitative figures ==="
$PY make_figures.py     --dataset orl   --data-root "$DATA_ROOT" --output "$OUT" --max-iter "$ITER_ORL"

echo "=== 2/8  ORL robustness grid (3 noise axes x 5 algorithms) ==="
$PY run_experiment.py   --dataset orl   --data-root "$DATA_ROOT" --output "$OUT" --max-iter "$ITER_ORL" --runs "$RUNS"

echo "=== 3/8  ORL convergence ==="
$PY run_convergence.py  --dataset orl   --data-root "$DATA_ROOT" --output "$OUT" --max-iter "$ITER_ORL"

echo "=== 4/8  ORL hyper-parameter sensitivity ==="
$PY run_sensitivity.py  --study delta --dataset orl --data-root "$DATA_ROOT" --output "$OUT" --max-iter "$ITER_ORL"
$PY run_sensitivity.py  --study lam   --dataset orl --data-root "$DATA_ROOT" --output "$OUT" --max-iter "$ITER_ORL"
$PY run_sensitivity.py  --study rank  --dataset orl --data-root "$DATA_ROOT" --output "$OUT" --max-iter "$ITER_ORL"

echo "=== 5/8  Extended YaleB qualitative figures ==="
$PY make_figures.py     --dataset yaleb --data-root "$DATA_ROOT" --output "$OUT" --max-iter "$ITER_YALE"

echo "=== 6/8  Extended YaleB robustness grid ==="
$PY run_experiment.py   --dataset yaleb --data-root "$DATA_ROOT" --output "$OUT" --max-iter "$ITER_YALE" --runs "$RUNS"

echo "=== 7/8  Extended YaleB convergence ==="
$PY run_convergence.py  --dataset yaleb --data-root "$DATA_ROOT" --output "$OUT" --max-iter "$ITER_YALE"

echo
echo "All done. Results in $OUT/"
