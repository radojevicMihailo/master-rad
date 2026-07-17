# -*- coding: utf-8 -*-
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

BASE = os.path.dirname(os.path.abspath(__file__)) + "/"
SLIKE = BASE + "rad/slike/"


def load_runs(csv_path):
    runs = defaultdict(lambda: ([], []))
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            x_values, y_values = runs[int(row["run"])]
            x_values.append(int(row["evaluations"]))
            y_values.append(int(row["best_fitness"]))
    return dict(sorted(runs.items()))


def mean_curve(runs):
    run_list = list(runs.values())
    length = min(len(x) for x, _ in run_list)
    xs = run_list[0][0][:length]
    ys = [sum(run[1][i] for run in run_list) / len(run_list) for i in range(length)]
    return xs, ys


def plot_with_mean(csv_path, out_prefix):
    runs = load_runs(csv_path)
    xs_mean, ys_mean = mean_curve(runs)
    for use_log, suffix in [(False, "linear"), (True, "loglog")]:
        plt.figure(figsize=(10, 5))
        first = True
        for run_id, (xs, ys) in runs.items():
            plt.plot(xs, ys, linewidth=0.9, alpha=0.55, color="#9bb4c8",
                     label=f"Pojedinačna pokretanja ({len(runs)})" if first else None)
            first = False
        plt.plot(xs_mean, ys_mean, linewidth=2.0, color="#1f3d5c", label="Srednji kumulativni maksimum")
        if use_log:
            plt.xscale("log")
            plt.yscale("log")
        plt.grid(True, which="both", alpha=0.3)
        plt.xlabel("Broj evaluacija")
        plt.ylabel("Zarada")
        plt.legend(loc="lower right")
        plt.tight_layout()
        out = f"{SLIKE}{out_prefix}-{suffix}.pdf"
        plt.savefig(out)
        plt.close()
        print("sacuvano:", out)


def plot_runs_png(csv_path, out_name):
    runs = load_runs(csv_path)
    plt.figure(figsize=(10, 5))
    for run_id, (xs, ys) in runs.items():
        plt.plot(xs, ys, linewidth=1.2, alpha=0.85, label=f"Pokretanje {run_id}")
    plt.grid(True, which="both", alpha=0.3)
    plt.xlabel("Broj evaluacija")
    plt.ylabel("Zarada")
    plt.legend(loc="lower right")
    plt.tight_layout()
    out = SLIKE + out_name
    plt.savefig(out, dpi=200)
    plt.close()
    print("sacuvano:", out)


plot_with_mean(BASE + "rezultati-10x10/svi-podaci/ga-cpp-convergence-1000-100-30.csv", "conv-10x10")
plot_with_mean(BASE + "rezultati-100x100/svi-podaci/ga-cpp-convergence-5000-100-50.csv", "conv-100x100")
plot_with_mean(BASE + "rezultati-1000x1000/ga-cpp-convergence-15000-100-100-6runs-2026-05-24_21-17-15.csv", "conv-1000x1000")
plot_runs_png(BASE + "rezultati-1000x1000/ga-cpp-convergence-15000-100-100-6runs-2026-05-24_21-17-15.csv", "conv-1000x1000-runs.png")
