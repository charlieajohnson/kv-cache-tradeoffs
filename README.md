# kv-cache-tradeoffs

Reproducible benchmark suite for studying **KV cache memory**, **throughput**, and **quality** trade-offs in decoder-only transformer inference across MHA, GQA, and MQA in sub-100M parameter regimes.

## Objectives

- Quantify memory scaling behavior under controlled conditions.
- Compare inference quality impacts (perplexity and next-token accuracy).
- Expose latency decomposition across attention/MLP/memory stages.
- Provide a deterministic, config-first research pipeline.

## Reference

The project follows the design in:

- `/Users/charlie/Library/Mobile Documents/com~apple~CloudDocs/Charlie/05_MiniProject/RST/kv-cache-tradeoffs.md`
- `/Users/charlie/Library/Mobile Documents/com~apple~CloudDocs/Charlie/05_MiniProject/RST/Memory — GitHub Repo.md`

## Repo conventions

- `configs/` are the source of truth for experiments.
- `results/raw/` stores unmodified run artifacts.
- `results/processed/` stores analysis-ready aggregates.
- `results/figures/` stores publication figures.
- `docs/agent-memory/` holds collaborative working notes, feature queue, and project status.

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run smoke checks:

```bash
make install
make test
make bench-smoke
```

## CLI

```bash
python -m kvbench.cli bench.kv_scaling --config configs/bench/kv_scaling.yaml
python -m kvbench.cli bench.throughput --config configs/bench/throughput.yaml
python -m kvbench.cli bench.sweep --config configs/bench/compression_sweep.yaml
python -m kvbench.cli plot all --config configs/runs/paper_defaults.yaml
```

## Data and environment

The framework is designed for small, repeatable runs on a single GPU.
Set:

- CUDA 12+
- PyTorch 2.x
- Reproducibility flags via fixed seeds and deterministic kernels where available

## Project status

See:

- `/Users/charlie/Documents/GitHub/kv-cache-tradeoffs/docs/agent-memory/project-status.md`
- `/Users/charlie/Documents/GitHub/kv-cache-tradeoffs/docs/agent-memory/feature-queue.md`
- `/Users/charlie/Documents/GitHub/kv-cache-tradeoffs/docs/agent-memory/guidelines.md`

## Citation

If you use this framework, cite with `CITATION.cff`.
