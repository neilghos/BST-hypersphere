

## How to Run

### 1. Single Seed Run (Fast Testing)
To run the full pipeline (Stages 1, 2, and 3) for a single random seed and see the evaluation metrics print to the console, use `trainer.py`. 

```bash
python trainer.py --dataset all
```

**Flags:**
- `--dataset`: Specifies which dataset to evaluate. 
  - Choices: `alpha`, `otc`, `epinions`, `slashdot`, `wiki-rfa`, `wiki-elec`.
  - Use `all` to run the pipeline sequentially across all 6 benchmark datasets.

### 2. Full Benchmark Run (10 Seeds)
To rigorously reproduce the state-of-the-art results reported in the paper, use `benchmark.py`. This script executes the entire pipeline across 10 independent random seeds to calculate the robust Mean and Standard Deviation, outputting the results to a CSV file in the `results/` folder.

```bash
python benchmark.py --dataset all
```

**Flags:**
- `--dataset`: Specifies which dataset to benchmark. Same choices as above (`alpha`, `otc`, `epinions`, `slashdot`, `wiki-rfa`, `wiki-elec`, `all`).

## Dependencies
All required packages are listed in `requirements.txt`. To install them, run:
```bash
pip install -r requirements.txt
```
