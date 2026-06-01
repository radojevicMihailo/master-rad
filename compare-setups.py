"""
Compare multiple GA setups by plotting mean convergence + stdev band per setup.

Usage:
    python3 compare-setups.py <csv1> <csv2> ... [--labels lab1,lab2,...]

Each CSV must have columns: run,generation,evaluations,best_fitness
(matches output of genetic-algorithm.cpp and the Python GA's convergence dump).
"""

import sys
import csv
import os
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np


def load_csv(path):
    runs = defaultdict(list)  # run_id -> list of (evaluations, fitness)
    with open(path) as file:
        reader = csv.DictReader(file)
        for row in reader:
            runs[row["run"]].append((int(row["evaluations"]), int(row["best_fitness"])))
    # Align by generation index across runs
    histories = []
    evaluations_axis = None
    for run_id in sorted(runs.keys(), key=int):
        runs[run_id].sort(key=lambda pair: pair[0])
        xs = [pair[0] for pair in runs[run_id]]
        ys = [pair[1] for pair in runs[run_id]]
        if evaluations_axis is None:
            evaluations_axis = xs
        histories.append(ys)
    return np.array(evaluations_axis), np.array(histories)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_paths", nargs="+")
    parser.add_argument("--labels", default=None,
                        help="comma-separated labels matching csv order; defaults to filenames")
    parser.add_argument("--output", default="comparison.svg")
    args = parser.parse_args()

    if args.labels:
        labels = args.labels.split(",")
    else:
        labels = [os.path.basename(path).replace(".csv", "") for path in args.csv_paths]

    if len(labels) != len(args.csv_paths):
        print("Number of labels must match number of CSVs")
        sys.exit(1)

    for use_log, suffix in [(True, "-loglog.svg"), (False, "-linear.svg")]:
        plt.figure(figsize=(11, 6.5))
        for path, label in zip(args.csv_paths, labels):
            evaluations_axis, histories = load_csv(path)
            mean_curve = histories.mean(axis=0)
            stdev_curve = histories.std(axis=0)
            line, = plt.plot(evaluations_axis, mean_curve, linewidth=1.5, label=label)
            plt.fill_between(evaluations_axis,
                             mean_curve - stdev_curve,
                             mean_curve + stdev_curve,
                             alpha=0.15, color=line.get_color())
        if use_log:
            plt.xscale("log")
            plt.yscale("log")
        plt.xlabel("Broj evaluacija")
        plt.ylabel("Vrednost funkcije cilja (zarada)")
        plt.title("Poredjenje GA setup-ova (srednja vrednost +/- stdev)")
        plt.grid(True, which="both", alpha=0.3)
        plt.legend(loc="lower right", fontsize=9)
        plt.tight_layout()

        out_path = args.output.replace(".svg", suffix)
        plt.savefig(out_path, format="svg")
        print(f"Sacuvano: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
