#!/usr/bin/env python3
"""
AIOps anomaly detection for CPU & memory
========================================

Learns each metric's normal behaviour and flags outliers with an
IsolationForest (unsupervised ML). Runs on live Prometheus / Grafana Cloud
metrics, or on a representative demo series.

The features per sample are the value plus its short-term rate of change, so
both sudden spikes and abnormal levels are caught.

Usage:
    python detect_anomalies.py                     # demo data
    python detect_anomalies.py --source prometheus --hours 24
      # needs env: GRAFANA_PROM_URL, GRAFANA_PROM_USER, GRAFANA_PROM_TOKEN
"""
from __future__ import annotations
import argparse, os
from datetime import datetime, timedelta

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.ensemble import IsolationForest

BG="#0b1220"; PANEL="#0f1a2b"; INK="#e6eef7"; MUTED="#8aa0b6"
CYAN="#22d3ee"; VIOLET="#a78bfa"; AMBER="#fbbf24"; GRID="#1e2d43"

PROMQL = {
    "CPU %":    '100 - (avg(rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)',
    "Memory %": '(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100',
}


def demo_data(hours=24, step_min=5, seed=11):
    rng = np.random.default_rng(seed)
    n = int(hours * 60 / step_min)
    start = datetime.now() - timedelta(hours=hours)
    dates = np.array([start + timedelta(minutes=step_min * i) for i in range(n)])
    cpu = np.clip(rng.normal(8, 2.5, n), 1, None)          # idle-ish CPU
    mem = np.clip(rng.normal(23, 1.2, n), 1, None)         # steady memory
    # inject realistic incidents (like a load test / batch job)
    for c in (int(n*0.35), int(n*0.7)):
        cpu[c:c+3] += rng.uniform(45, 70)                  # CPU spikes
    mem[int(n*0.55):int(n*0.55)+4] += 18                   # a memory blip
    return dates, {"CPU %": cpu, "Memory %": mem}


def query_prometheus(hours):
    import requests
    base = os.environ["GRAFANA_PROM_URL"].rstrip("/") + "/api/v1/query_range"
    user, token = os.environ["GRAFANA_PROM_USER"], os.environ["GRAFANA_PROM_TOKEN"]
    end = datetime.now(); start = end - timedelta(hours=hours)
    series, dates = {}, None
    for name, promql in PROMQL.items():
        r = requests.get(base, auth=(user, token), params={
            "query": promql, "start": start.timestamp(), "end": end.timestamp(), "step": "300"})
        r.raise_for_status()
        res = r.json()["data"]["result"]
        if not res:
            raise SystemExit(f"No data for '{name}'. Check the endpoint/token/time range.")
        vals = res[0]["values"]
        dates = np.array([datetime.fromtimestamp(float(ts)) for ts, _ in vals])
        series[name] = np.array([float(v) for _, v in vals])
    return dates, series


def detect(values):
    """IsolationForest on [value, delta]; returns boolean anomaly mask."""
    delta = np.diff(values, prepend=values[0])
    X = np.column_stack([values, delta])
    model = IsolationForest(contamination=0.03, random_state=0)
    return model.fit_predict(X) == -1


def render(dates, series, out_png):
    plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": PANEL, "text.color": INK,
                         "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.edgecolor": GRID, "font.family": "DejaVu Sans"})
    names = list(series.keys())
    fig, axes = plt.subplots(len(names), 1, figsize=(10, 6), dpi=130, sharex=True)
    colors = [CYAN, VIOLET]
    total = 0
    for ax, name, col in zip(axes, names, colors):
        vals = series[name]; mask = detect(vals); total += int(mask.sum())
        ax.plot(dates, vals, color=col, lw=1.8, label=name)
        if mask.any():
            ax.scatter(np.array(dates)[mask], vals[mask], color=AMBER, s=45,
                       zorder=5, edgecolor=BG, label="Anomaly")
        ax.set_ylabel(name); ax.grid(True, color=GRID, lw=0.6, alpha=0.6)
        leg = ax.legend(loc="upper left", framealpha=0.15, facecolor=PANEL, edgecolor=GRID, fontsize=9)
        for t in leg.get_texts(): t.set_color(INK)
    axes[0].set_title("Live anomaly detection — CPU & memory (IsolationForest)",
                      color=INK, fontsize=15, fontweight="bold", loc="left")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.tight_layout(); fig.savefig(out_png, facecolor=BG)
    print(f"Chart written to {out_png}  |  anomalies flagged: {total}")


def main():
    ap = argparse.ArgumentParser(description="AIOps CPU/memory anomaly detection")
    ap.add_argument("--source", choices=["demo", "prometheus"], default="demo")
    ap.add_argument("--hours", type=int, default=24)
    ap.add_argument("--out", default="anomaly-detection.png")
    args = ap.parse_args()
    dates, series = (query_prometheus(args.hours) if args.source == "prometheus"
                     else demo_data(args.hours))
    print("=" * 60)
    print(f"Anomaly detection  (source: {args.source})")
    for name, vals in series.items():
        m = detect(vals)
        print(f"  {name:9s}: {int(m.sum())} anomalies  (latest {vals[-1]:.1f})")
    print("=" * 60)
    render(dates, series, args.out)


if __name__ == "__main__":
    main()
