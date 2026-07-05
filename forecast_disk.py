#!/usr/bin/env python3
"""
AIOps disk-capacity forecaster
==============================

Predicts *when* a server's disk will cross a critical threshold (default 85%)
and flags anomalies in the usage history - the "predict, don't just react"
idea behind AIOps.

- Works on live Prometheus / Grafana Cloud metrics (see query_prometheus), or
  on a representative demo series so it runs out of the box.
- Fits a linear trend, projects it forward, and reports the estimated date the
  disk hits the threshold.
- Detects anomalies as points whose residual from the trend exceeds 3 sigma.
- Renders a dark-themed chart (history + forecast + confidence band + threshold
  + predicted crossing + anomalies) suitable for a portfolio site.

Usage:
    python forecast_disk.py                 # demo data
    python forecast_disk.py --threshold 85  # custom threshold
    # live data (requires a Grafana Cloud Prometheus endpoint + token):
    #   set GRAFANA_PROM_URL, GRAFANA_PROM_USER, GRAFANA_PROM_TOKEN
    python forecast_disk.py --source prometheus
"""
from __future__ import annotations
import argparse, os, sys
from datetime import datetime, timedelta

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

THRESHOLD_DEFAULT = 85.0
BG   = "#0b1220"; PANEL = "#0f1a2b"; INK = "#e6eef7"; MUTED = "#8aa0b6"
CYAN = "#22d3ee"; BLUE = "#3b82f6"; RED = "#f87171"; AMBER = "#fbbf24"; GRID = "#1e2d43"


# --------------------------------------------------------------------------- #
# Data sources
# --------------------------------------------------------------------------- #
def demo_series(days: int = 45, seed: int = 7):
    """A realistic disk-usage history: steady growth + noise + a couple of spikes."""
    rng = np.random.default_rng(seed)
    t = np.arange(days)
    base = 42.0 + 0.72 * t                       # ~0.72%/day growth
    weekly = 0.6 * np.sin(2 * np.pi * t / 7)     # mild weekly wobble
    noise = rng.normal(0, 0.5, days)
    usage = base + weekly + noise
    # inject anomalies (e.g. a log/backup blow-up that clears the next day)
    for idx, bump in [(18, 5.5), (31, 4.2)]:
        usage[idx] += bump
    start = datetime.now() - timedelta(days=days - 1)
    dates = np.array([start + timedelta(days=int(i)) for i in t])
    return dates, usage


def query_prometheus(threshold_metric: str, days: int):
    """Fetch a real disk-usage series from Grafana Cloud / Prometheus.

    Expects env vars GRAFANA_PROM_URL (…/api/prom), GRAFANA_PROM_USER, GRAFANA_PROM_TOKEN.
    Returns (dates, usage%) arrays. Kept simple; extend the PromQL as needed.
    """
    import requests
    url   = os.environ["GRAFANA_PROM_URL"].rstrip("/") + "/api/v1/query_range"
    user  = os.environ["GRAFANA_PROM_USER"]
    token = os.environ["GRAFANA_PROM_TOKEN"]
    end = datetime.now(); start = end - timedelta(days=days)
    promql = ('100 - (node_filesystem_avail_bytes{mountpoint="/",fstype!~"tmpfs|overlay"} '
              '/ node_filesystem_size_bytes{mountpoint="/",fstype!~"tmpfs|overlay"} * 100)')
    r = requests.get(url, auth=(user, token), params={
        "query": promql, "start": start.timestamp(), "end": end.timestamp(), "step": "3600"})
    r.raise_for_status()
    res = r.json()["data"]["result"]
    if not res:
        raise SystemExit("No data returned from Prometheus for that query/time range.")
    values = res[0]["values"]
    dates = np.array([datetime.fromtimestamp(float(ts)) for ts, _ in values])
    usage = np.array([float(v) for _, v in values])
    return dates, usage


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def fit_and_forecast(dates, usage, threshold, horizon_days=30):
    """Linear trend fit + forward projection + anomaly flags + threshold crossing."""
    t = np.array([(d - dates[0]).total_seconds() / 86400.0 for d in dates])  # days
    A = np.vstack([t, np.ones_like(t)]).T
    slope, intercept = np.linalg.lstsq(A, usage, rcond=None)[0]
    fit = slope * t + intercept

    # anomalies: residual z-score > 3
    resid = usage - fit
    sigma = resid.std()
    z = np.abs(resid) / (sigma if sigma else 1)
    anomalies = z > 3.0

    # forecast forward
    future_t = np.arange(t[-1] + 1, t[-1] + 1 + horizon_days)
    future_dates = np.array([dates[0] + timedelta(days=float(ft)) for ft in future_t])
    future_fit = slope * future_t + intercept
    # simple widening confidence band
    band = sigma * (1.0 + 0.06 * (future_t - t[-1]))

    # when does the trend cross the threshold?
    cross_day = (threshold - intercept) / slope if slope > 0 else None
    cross_date = dates[0] + timedelta(days=float(cross_day)) if cross_day and cross_day > t[-1] else None
    days_to = (cross_day - t[-1]) if cross_date else None

    return dict(t=t, fit=fit, slope=slope, sigma=sigma, anomalies=anomalies,
                future_dates=future_dates, future_fit=future_fit, band=band,
                cross_date=cross_date, days_to=days_to)


# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #
def render(dates, usage, m, threshold, out_png):
    plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": PANEL,
                         "text.color": INK, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.edgecolor": GRID, "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=130)

    ax.plot(dates, usage, color=CYAN, lw=2, label="Observed disk usage")
    ax.plot(m["future_dates"], m["future_fit"], color=BLUE, lw=2, ls="--", label="Forecast (trend)")
    ax.fill_between(m["future_dates"], m["future_fit"] - m["band"], m["future_fit"] + m["band"],
                    color=BLUE, alpha=0.15, label="Confidence band")

    ax.axhline(threshold, color=RED, lw=1.5, ls=":", label=f"Critical {threshold:.0f}%")
    if m["anomalies"].any():
        ax.scatter(np.array(dates)[m["anomalies"]], usage[m["anomalies"]],
                   color=AMBER, s=55, zorder=5, edgecolor=BG, label="Anomaly")

    if m["cross_date"] is not None:
        ax.axvline(m["cross_date"], color=RED, lw=1, alpha=0.6)
        ax.annotate(f"~{m['days_to']:.0f} days to {threshold:.0f}%\n{m['cross_date']:%d %b %Y}",
                    xy=(m["cross_date"], threshold), xytext=(-10, -55),
                    textcoords="offset points", ha="right", color=INK, fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.4", fc="#20304a", ec=RED, alpha=0.9),
                    arrowprops=dict(arrowstyle="->", color=RED))

    ax.set_title("Predictive disk-capacity monitoring (AIOps)", color=INK, fontsize=15, fontweight="bold", loc="left")
    ax.set_ylabel("Disk used %")
    ax.set_ylim(min(usage.min() - 5, 35), max(threshold + 8, usage.max() + 8))
    ax.grid(True, color=GRID, lw=0.6, alpha=0.6)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    leg = ax.legend(loc="upper left", framealpha=0.15, facecolor=PANEL, edgecolor=GRID, fontsize=9)
    for txt in leg.get_texts(): txt.set_color(INK)
    fig.tight_layout()
    fig.savefig(out_png, facecolor=BG)
    print(f"Chart written to {out_png}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="AIOps disk-capacity forecaster")
    ap.add_argument("--source", choices=["demo", "prometheus"], default="demo")
    ap.add_argument("--threshold", type=float, default=THRESHOLD_DEFAULT)
    ap.add_argument("--days", type=int, default=45, help="history window (demo) / lookback (prometheus)")
    ap.add_argument("--out", default="disk-forecast.png")
    args = ap.parse_args()

    if args.source == "prometheus":
        dates, usage = query_prometheus("disk", args.days)
    else:
        dates, usage = demo_series(args.days)

    m = fit_and_forecast(dates, usage, args.threshold)

    print("=" * 60)
    print(f"AIOps disk-capacity forecast  (source: {args.source})")
    print("=" * 60)
    print(f"Latest usage       : {usage[-1]:.1f}%")
    print(f"Growth rate (trend): {m['slope']:.2f}% per day")
    print(f"Anomalies detected : {int(m['anomalies'].sum())}")
    if m["cross_date"] is not None:
        print(f"Predicted {args.threshold:.0f}% breach: {m['cross_date']:%Y-%m-%d}  (~{m['days_to']:.0f} days)")
        print("Recommendation     : plan capacity / clean-up before the breach date.")
    else:
        print(f"Predicted {args.threshold:.0f}% breach: not within the forecast horizon (trend flat/negative).")
    print("=" * 60)

    render(dates, usage, m, args.threshold, args.out)


if __name__ == "__main__":
    main()
