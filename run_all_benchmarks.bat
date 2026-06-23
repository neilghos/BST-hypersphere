@echo off
echo =======================================================
echo Starting Full 10-Seed Benchmark Suite for BST Model
echo =======================================================

echo.
echo Running benchmark for dataset: alpha
python benchmark.py --dataset alpha

echo.
echo Running benchmark for dataset: otc
python benchmark.py --dataset otc

echo.
echo Running benchmark for dataset: epinions
python benchmark.py --dataset epinions

echo.
echo Running benchmark for dataset: wiki-rfa
python benchmark.py --dataset wiki-rfa

echo.
echo Running benchmark for dataset: wiki-elec
python benchmark.py --dataset wiki-elec

echo.
echo =======================================================
echo All benchmarks completed successfully!
echo =======================================================
pause
