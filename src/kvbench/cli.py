from __future__ import annotations

import json
from pathlib import Path

import typer

from kvbench.bench import (
    run_compression_sweep,
    run_kv_scaling,
    run_latency_breakdown,
    run_throughput,
)
from kvbench.config import ExperimentConfig
from kvbench.plotting import (
    fig_compression_threshold,
    fig_kv_memory,
    fig_latency_breakdown,
    fig_throughput,
)
from kvbench.logging import setup_logger

app = typer.Typer()


def _load(name: str) -> ExperimentConfig:
    return ExperimentConfig.from_file(name)


def _bench_kv_scaling(config: str):
    cfg = _load(config).data
    results = [r.__dict__ for r in run_kv_scaling(cfg)]
    print(json.dumps(results, indent=2))


@app.command()
def bench_kv_scaling(config: str = typer.Option(..., "--config")):
    _bench_kv_scaling(config)


@app.command(name="bench.kv_scaling")
def bench_kv_scaling_dot(config: str = typer.Option(..., "--config")):
    _bench_kv_scaling(config)


@app.command(name="bench_kv_scaling")
def bench_kv_scaling_underscore(config: str = typer.Option(..., "--config")):
    _bench_kv_scaling(config)


def _bench_throughput(config: str):
    cfg = _load(config).data
    results = [r.__dict__ for r in run_throughput(cfg)]
    print(json.dumps(results, indent=2))


@app.command()
def bench_throughput(config: str = typer.Option(..., "--config")):
    _bench_throughput(config)


@app.command(name="bench.throughput")
def bench_throughput_dot(config: str = typer.Option(..., "--config")):
    _bench_throughput(config)


@app.command(name="bench_throughput")
def bench_throughput_underscore(config: str = typer.Option(..., "--config")):
    _bench_throughput(config)


def _bench_sweep(config: str):
    cfg = _load(config).data
    results = [r.__dict__ for r in run_compression_sweep(cfg)]
    print(json.dumps(results, indent=2))


@app.command()
def bench_sweep(config: str = typer.Option(..., "--config")):
    _bench_sweep(config)


@app.command(name="bench.sweep")
def bench_sweep_dot(config: str = typer.Option(..., "--config")):
    _bench_sweep(config)


@app.command(name="bench_sweep")
def bench_sweep_underscore(config: str = typer.Option(..., "--config")):
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
