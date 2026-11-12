# Experiment log

One Markdown file per experiment: `YYYY-MM-DD-<short-name>.md`.
Every entry must contain enough to rerun it exactly:

```markdown
# <short name>

- **Date:**
- **Commit:** `git rev-parse HEAD` (working tree clean? yes/no)
- **Config:** path in `configs/` (or inline)
- **Seed:**
- **Hardware:** CPU, GPU, driver / CUDA version, RAM
- **Software:** rustc, Burn version, Python/PyTorch version if relevant
- **Command:** exact command line
- **Data:** dataset, split, subset

## Results
Numbers (with p50/p95/p99 for latencies, mean ± std over N runs where applicable) and links to plots.

## Notes
What was expected, what happened, what it means.
```
