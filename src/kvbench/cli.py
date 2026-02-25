from __future__ import annotations

import json
from pathlib import Path
import typer

from kvbench.config import ExperimentConfig
from kvbench.bench import run_kv_scaling, run_throughput, run_latency_breakdown, run_compression_sweep
from kvbench.plotting import (
    fig_kv_memory,
    fig_throughput,
    fig_compression_threshold,
    fig_latency_breakdown,
)
from kvbench.utils import setup_logger

app = typer.Typer()


def _load(name: str) -> ExperimentConfig:
    return ExperimentConfig.from_file(name)


def _bench_kv_scaling(config: str):
    cfg = _load(config).data
    results = [r.__dict__ for r in run_kv_scaling(cfg)]
    print(json.dumps(results, indent=2))


@app.command()
def bench_kv_scaling(config: str):
    _bench_kv_scaling(config)


@app.command(name="bench.kv_scaling")
def bench_kv_scaling_dot(config: str):
    _bench_kv_scaling(config)


def _bench_throughput(config: str):
    cfg = _load(config).data
    results = [r.__dict__ for r in run_throughput(cfg)]
    print(json.dumps(results, indent=2))


@app.command()
def bench_throughput(config: str):
    _bench_throughput(config)


@app.command(name="bench.throughput")
def bench_throughput_dot(config: str):
    _bench_throughput(config)


def _bench_sweep(config: str):
    cfg = _load(config).data
    results = [r.__dict__ for r in run_compression_sweep(cfg)]
    print(json.dumps(results, indent=2))


@app.command()
def bench_sweep(config: str):
    _bench_sweep(config)


@app.command(name="bench.sweep")
def bench_sweep_dot(config: str):
    _bench_sweep(config)


@app.command()
def plot(what: str, config: str):
    cfg = _load(config).data
    out_dir = Path(cfg.get("output_dir", "results/figures"))
    out_dir.mkdir(parents=True, exist_ok=True)
    if what == "kv":
        fig_kv_memory(run_kv_scaling(cfg), out_dir / "kv_memory.png")
    elif what == "throughput":
        fig_throughput(run_throughput(cfg), out_dir / "throughput.png")
    elif what == "compression":
        fig_compression_threshold(run_compression_sweep(cfg), out_dir / "compression.png")
    elif what == "latency":
        fig_latency_breakdown(run_latency_breakdown(cfg), out_dir / "latency.png")
    else:
        raise typer.BadParameter("plot target must be kv, throughput, compression, latency")


if __name__ == "__main__":
    setup_logger()
    app()
