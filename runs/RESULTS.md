# Experimental results
Auto-generated from `runs/summary.json` by `experiments/write_results_md.py`.
All experiments use $p = 113$, train\_frac = 0.30, AdamW with weight decay = 1.0.

## Per-run summary

| run | ops | steps | final test acc | grok step (test_acc ≥ 0.95) |
|-----|-----|------:|---------------:|------------------:|
| `01_baseline_addition` | + | 30,000 | 1.0000 | 7,200 |
| `02_baseline_subtraction` | - | 15,000 | 1.0000 | 7,000 |
| `03_baseline_multiplication` | * | 25,000 | 0.9998 | 4,200 |
| `04_multitask_two` | +/- | 20,000 | 1.0000 | 6,000 |

## Per-task grok steps (multi-task runs)

| run | + | - |
|-----|------:|------:|
| `04_multitask_two` | 5,800 | 6,000 |
