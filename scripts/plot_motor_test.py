#!/usr/bin/env python3
"""Plot motor test IMU telemetry from runs/motor_test_*.csv.

Single file  — time-series per test (accel_x, accel_y, gyro_z).
Multiple files — surface comparison overlay for each test.

Usage:
    python scripts/plot_motor_test.py                          # auto-find all CSVs in runs/
    python scripts/plot_motor_test.py runs/motor_test_*.csv   # explicit files
    python scripts/plot_motor_test.py runs/                   # directory
"""

import sys
import csv
import glob
import os
from collections import defaultdict, OrderedDict

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({"font.size": 8})

# ── Data loading ──────────────────────────────────────────────────────────────

def load_csv(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({
                "surface": r["surface"],
                "test":    r["test"],
                "t_ms":    float(r["t_ms"]),
                "ax":      float(r["ax"]),
                "ay":      float(r["ay"]),
                "az":      float(r["az"]),
                "gx":      float(r["gx"]),
                "gy":      float(r["gy"]),
                "gz":      float(r["gz"]),
            })
    return rows

def by_test(rows):
    """Return OrderedDict preserving insertion order of tests."""
    d = OrderedDict()
    for r in rows:
        d.setdefault(r["test"], []).append(r)
    return d

# ── Metrics ───────────────────────────────────────────────────────────────────

def metrics(data):
    ax = [r["ax"] for r in data]
    ay = [r["ay"] for r in data]
    gz = [r["gz"] for r in data]
    lat  = abs(np.mean(ay))
    fwd  = abs(np.mean(ax))
    purity = lat / (lat + fwd) * 100 if lat + fwd > 0.02 else float("nan")
    return {
        "mean_ax": np.mean(ax), "peak_ax": np.max(np.abs(ax)),
        "mean_ay": np.mean(ay), "peak_ay": np.max(np.abs(ay)),
        "mean_gz": np.mean(gz), "peak_gz": np.max(np.abs(gz)),
        "purity":  purity,
        "n":       len(data),
        "dur_s":   data[-1]["t_ms"] / 1000 if data else 0,
    }

# ── Single-file plot ──────────────────────────────────────────────────────────

def plot_single(path):
    rows = load_csv(path)
    if not rows:
        print(f"  No data in {path}")
        return

    surface = rows[0]["surface"]
    tests   = by_test(rows)
    n       = len(tests)

    fig, axes = plt.subplots(n, 3, figsize=(13, 2.6 * n), squeeze=False)
    fig.suptitle(
        f"Motor Test IMU — {surface}\n{os.path.basename(path)}",
        fontsize=10, y=1.01,
    )

    COLS = [
        ("ax", "accel_x  (m/s²)",  "tab:blue"),
        ("ay", "accel_y  (m/s²)",  "tab:green"),
        ("gz", "gyro_z  (rad/s)",   "tab:red"),
    ]

    for i, (test_name, data) in enumerate(tests.items()):
        t = np.array([r["t_ms"] for r in data]) / 1000.0
        m = metrics(data)
        for j, (key, label, color) in enumerate(COLS):
            ax = axes[i][j]
            vals = np.array([r[key] for r in data])
            ax.plot(t, vals, color=color, linewidth=0.9)
            ax.axhline(0, color="k", linewidth=0.4, linestyle="--")
            if i == 0:
                ax.set_title(label, fontsize=9)
            if j == 0:
                # Row label + purity summary
                summary = (
                    f"{test_name}\n"
                    f"ax={m['mean_ax']:+.2f}  ay={m['mean_ay']:+.2f}  "
                    f"gz={m['mean_gz']:+.3f}\n"
                )
                if not np.isnan(m["purity"]):
                    summary += f"purity {m['purity']:.0f}%"
                ax.set_ylabel(summary, fontsize=7, rotation=0, ha="right",
                              va="center", labelpad=90)
            if i == n - 1:
                ax.set_xlabel("t (s)")
            ax.tick_params(labelsize=7)

    plt.tight_layout()
    out = path.replace(".csv", "_plot.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.show()

# ── Multi-file comparison ─────────────────────────────────────────────────────

def plot_compare(paths):
    all_data   = {}   # surface -> test -> rows
    test_order = []   # preserve order from first file

    for path in paths:
        rows = load_csv(path)
        if not rows:
            continue
        surface = rows[0]["surface"]
        td = by_test(rows)
        all_data[surface] = td
        if not test_order:
            test_order = list(td.keys())

    if not all_data:
        print("No data found.")
        return

    surfaces = list(all_data.keys())
    n = len(test_order)
    colors = plt.cm.tab10.colors

    fig, axes = plt.subplots(n, 3, figsize=(13, 2.6 * n), squeeze=False)
    fig.suptitle("Motor Test IMU — Surface Comparison", fontsize=10, y=1.01)

    COLS = [
        ("ax", "accel_x  (m/s²)"),
        ("ay", "accel_y  (m/s²)"),
        ("gz", "gyro_z  (rad/s)"),
    ]

    for i, test_name in enumerate(test_order):
        for j, (key, label) in enumerate(COLS):
            ax = axes[i][j]
            ax.axhline(0, color="k", linewidth=0.4, linestyle="--")
            if i == 0:
                ax.set_title(label, fontsize=9)
            if i == n - 1:
                ax.set_xlabel("t (s)")
            ax.tick_params(labelsize=7)

            for k, surface in enumerate(surfaces):
                data = all_data.get(surface, {}).get(test_name)
                if not data:
                    continue
                t    = np.array([r["t_ms"] for r in data]) / 1000.0
                vals = np.array([r[key]    for r in data])
                ax.plot(t, vals, color=colors[k % len(colors)],
                        linewidth=0.9, label=surface if j == 0 else None)

            if j == 0:
                m_list = []
                for surface in surfaces:
                    data = all_data.get(surface, {}).get(test_name)
                    if data:
                        m = metrics(data)
                        p = f"{m['purity']:.0f}%" if not np.isnan(m["purity"]) else "—"
                        m_list.append(f"{surface}: purity {p}")
                summary = test_name + "\n" + "\n".join(m_list)
                ax.set_ylabel(summary, fontsize=7, rotation=0, ha="right",
                              va="center", labelpad=100)
                ax.legend(fontsize=7, loc="upper right")

    plt.tight_layout()
    out = "runs/motor_test_comparison.png"
    os.makedirs("runs", exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.show()

# ── Entry point ───────────────────────────────────────────────────────────────

def resolve_paths(args):
    paths = []
    for a in args:
        if os.path.isdir(a):
            paths.extend(sorted(glob.glob(os.path.join(a, "motor_test_*.csv"))))
        else:
            paths.extend(sorted(glob.glob(a)))
    return paths

if __name__ == "__main__":
    args  = sys.argv[1:]
    paths = resolve_paths(args) if args else sorted(glob.glob("runs/motor_test_*.csv"))

    if not paths:
        print("No CSV files found. Run motor_test on the Pi first, then scp the runs/ dir here.")
        print("  scp -r raspbot:~/Projects/Robot/runs .")
        sys.exit(1)

    print(f"Found {len(paths)} file(s):")
    for p in paths:
        print(f"  {p}")
    print()

    if len(paths) == 1:
        plot_single(paths[0])
    else:
        print("Options:  [1] individual plots   [2] surface comparison overlay")
        choice = (input("Choice [2]: ").strip() or "2")
        if choice == "1":
            for p in paths:
                plot_single(p)
        else:
            plot_compare(paths)
