"""
Plot one average best-so-far line per CSV file, averaged across all runs in
that file. Each CSV must have columns: run,generation,evaluations,best_fitness
(the schema produced by genetic-algorithm.cpp / .py and random-search.cpp).

One line per input file = mean over its runs. Useful for comparing algorithms
or population sizes on the same axes.

Edit CSV_FILES below to choose which files to plot, then run:
    python3 plot-pop-comparison.py
"""

import os
import csv as csv_module
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt


RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rezultati-10x10/grafici" \
"")

CSV_FILES = [
    "rezultati-10x10/svi-podaci/ga-cpp-convergence-1000-1000-30.csv",
    "rezultati-10x10/svi-podaci/ga-cpp-convergence-2000-500-50.csv",
]

OUTPUT_PATH = os.path.join(RESULTS_DIR, "compare-average-10x10-1.svg")


def label_from_path(path):
    """Human label from filename, e.g. 'ga-cpp-convergence-20000-...' -> 'ga-cpp 20000'."""
    base = os.path.splitext(os.path.basename(path))[0]
    parts = base.split("-")
    algo = []
    for token in parts:
        if token.isdigit():
            algo.append(token)  # first number = population size
            break
        if token != "convergence":
            algo.append(token)
    return " ".join(algo) if algo else base


def parse_convergence_csv(path):
    """Return OrderedDict[run_id] = (evaluations_list, best_fitness_list)."""
    runs = OrderedDict()
    with open(path) as file:
        reader = csv_module.DictReader(file)
        for row in reader:
            run_id = row["run"]
            if run_id not in runs:
                runs[run_id] = ([], [])
            runs[run_id][0].append(int(row["evaluations"]))
            runs[run_id][1].append(int(row["best_fitness"]))
    return runs


def mean_curve(path):
    """Return (evaluations, mean_best_so_far) averaged across all runs in file."""
    runs = parse_convergence_csv(path)
    if not runs:
        return None, None, 0
    min_length = min(len(best) for _, best in runs.values())
    best_so_far = np.array([
        np.maximum.accumulate(best[:min_length]) for _, best in runs.values()
    ])
    mean_best = best_so_far.mean(axis=0)
    first_run = next(iter(runs.values()))
    evaluations = np.array(first_run[0][:min_length])
    return evaluations, mean_best, len(runs)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve(path):
    return path if os.path.isabs(path) else os.path.join(BASE_DIR, path)


def main():
    plt.figure(figsize=(11, 6.5))
    plotted = 0
    for entry in CSV_FILES:
        path = resolve(entry)
        if not os.path.exists(path):
            print(f"WARN: not found {path}")
            continue
        evaluations, mean_best, num_runs = mean_curve(path)
        if evaluations is None:
            print(f"WARN: no rows in {path}")
            continue
        plt.plot(evaluations, mean_best, linewidth=1.8,
                 label=f"{label_from_path(path)} ({num_runs} pokretanja)")
        plotted += 1

    if plotted == 0:
        print("No data plotted. Check CSV_FILES.")
        return

    plt.xscale("log")
    plt.xlabel("Broj evaluacija")
    plt.ylabel("Prosecna vrednost najboljeg resenja (zarada)")
    plt.title("Poredjenje algoritama — prosek po pokretanjima")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    plt.savefig(OUTPUT_PATH, format="svg")
    print(f"Sacuvano: {OUTPUT_PATH}")
    plt.show()


if __name__ == "__main__":
    main()
