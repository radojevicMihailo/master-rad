import sys
import csv
import os
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    print("Usage: python3 plot-convergence.py <path-to-convergence.csv>")
    sys.exit(1)

csv_path = sys.argv[1]

runs = {}
with open(csv_path, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        run_id = row["run"]
        if run_id not in runs:
            runs[run_id] = ([], [])
        runs[run_id][0].append(int(row["evaluations"]))
        runs[run_id][1].append(int(row["best_fitness"]))

def make_plot(scale_label, use_log):
    plt.figure(figsize=(10, 6))
    for run_id, (x_values, y_values) in runs.items():
        plt.plot(x_values, y_values, linewidth=1, alpha=0.7, label=f"Pokretanje {run_id}")
    if use_log:
        plt.xscale("log")
        plt.yscale("log")
    plt.xlabel("Broj evaluacija")
    plt.ylabel("Vrednost funkcije cilja (zarada)")
    plt.title(f"Konvergencija GA ({scale_label}) — {os.path.basename(csv_path)}")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()

    suffix = "-loglog.svg" if use_log else "-linear.svg"
    svg_path = csv_path.replace(".csv", suffix)
    plt.savefig(svg_path, format="svg")
    print(f"Grafik sacuvan: {svg_path}")

make_plot("log-log", True)
make_plot("linearno", False)
plt.show()
