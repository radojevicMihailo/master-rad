import csv
import random
import os
import statistics
import time
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

# ── Load Data from Files ──────────────────────────────────────────────────────

PROBLEM_SIZE = "10x10"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"podaci-{PROBLEM_SIZE}")


def load_matrix_csv(filepath):
    matrix = []
    with open(filepath, "r") as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            matrix.append([int(value) for value in row[1:]])
    return matrix


def load_demand_csv(filepath):
    demand = []
    with open(filepath, "r") as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) >= 2:
                try:
                    demand.append(int(row[1]))
                except ValueError:
                    continue
    return demand


def load_matrix_tsv(filepath):
    matrix = []
    with open(filepath, "r") as file:
        lines = file.readlines()
        for line in lines[1:]:
            parts = line.strip().split("\t")
            if len(parts) > 1:
                row = [int(value) for value in parts[1:]]
                matrix.append(row)
    return matrix


def load_demand_tsv(filepath):
    demand = []
    with open(filepath, "r") as file:
        for line in file.readlines():
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                try:
                    demand.append(int(parts[1]))
                except ValueError:
                    continue
    return demand


def load_data(problem_size, data_directory):
    size_number = problem_size.split("x")[0]

    csv_time = os.path.join(data_directory, f"vremena_izvrsavanja_{size_number}.csv")
    csv_profit = os.path.join(data_directory, f"zarada_po_servisu_{size_number}.csv")
    csv_demand = os.path.join(data_directory, f"zahtevi_za_servisima_{size_number}.csv")

    if os.path.exists(csv_time):
        time_matrix = load_matrix_csv(csv_time)
        profit_matrix = load_matrix_csv(csv_profit)
        demand_vector = load_demand_csv(csv_demand)
    else:
        time_matrix = load_matrix_tsv(os.path.join(data_directory, "vremena_izvrsavanja.txt"))
        profit_matrix = load_matrix_tsv(os.path.join(data_directory, "zarada_po_servisu.txt"))
        demand_vector = load_demand_tsv(os.path.join(data_directory, "zahtevi_za_servisima.txt"))

    return time_matrix, profit_matrix, demand_vector


print(f"Ucitavanje podataka za {PROBLEM_SIZE}...")
_time_list, _profit_list, _demand_list = load_data(PROBLEM_SIZE, DATA_DIR)

TIME = np.array(_time_list, dtype=np.int32)
PROFIT = np.array(_profit_list, dtype=np.int32)
DEMAND = np.array(_demand_list, dtype=np.int32)

NUM_SERVICES, NUM_COMPUTERS = TIME.shape
MAX_TIME = 2880

print(f"Dimenzije problema: {NUM_SERVICES} servisa x {NUM_COMPUTERS} racunara")

# ── Random Search Hyperparameters (mirrors GA budget) ─────────────────────────

POPULATION_SIZE = 100
GENERATIONS = 500

print("\nRandom Search Parametri:")
print(f"  Velicina populacije: {POPULATION_SIZE}")
print(f"  Broj generacija: {GENERATIONS}")
print(f"  Ukupno evaluacija po pokretanju: {POPULATION_SIZE * GENERATIONS}")

# ── Precomputed Helpers ───────────────────────────────────────────────────────

PROFIT_PER_MINUTE = PROFIT.astype(np.float64) / TIME.astype(np.float64)
MAX_UNITS = (MAX_TIME // TIME).astype(np.int32)

SERVICES_BY_RATIO_PER_COMPUTER = [
    np.argsort(PROFIT_PER_MINUTE[:, computer]).tolist()
    for computer in range(NUM_COMPUTERS)
]

SERVICES_BEST_FIRST_PER_COMPUTER = [
    np.argsort(PROFIT_PER_MINUTE[:, computer])[::-1].tolist()
    for computer in range(NUM_COMPUTERS)
]


# ── Repair ────────────────────────────────────────────────────────────────────

def repair(chromosome):
    np.clip(chromosome, 0, None, out=chromosome)

    row_totals = chromosome.sum(axis=1)
    over_demand = row_totals > DEMAND
    if over_demand.any():
        ratios = DEMAND[over_demand].astype(np.float64) / row_totals[over_demand]
        chromosome[over_demand] = (chromosome[over_demand] * ratios[:, np.newaxis]).astype(np.int32)

    for computer in range(NUM_COMPUTERS):
        used = int(np.sum(TIME[:, computer] * chromosome[:, computer]))
        if used > MAX_TIME:
            for service in SERVICES_BY_RATIO_PER_COMPUTER[computer]:
                if used <= MAX_TIME:
                    break
                if chromosome[service, computer] > 0:
                    excess = used - MAX_TIME
                    units_to_remove = min(int(chromosome[service, computer]), (excess + int(TIME[service, computer]) - 1) // int(TIME[service, computer]))
                    chromosome[service, computer] -= units_to_remove
                    used -= units_to_remove * int(TIME[service, computer])

    for computer in range(NUM_COMPUTERS):
        used = int(np.sum(TIME[:, computer] * chromosome[:, computer]))
        remaining = MAX_TIME - used
        for service in SERVICES_BEST_FIRST_PER_COMPUTER[computer]:
            if remaining < TIME[service, computer]:
                continue
            demand_left = int(DEMAND[service]) - int(chromosome[service].sum())
            if demand_left <= 0:
                continue
            add = min(demand_left, remaining // int(TIME[service, computer]))
            if add > 0:
                chromosome[service, computer] += add
                remaining -= add * int(TIME[service, computer])
            if remaining < 5:
                break

    return chromosome


# ── Fitness ───────────────────────────────────────────────────────────────────

def fitness(chromosome):
    return int(np.sum(PROFIT * chromosome))


# ── Initialization ────────────────────────────────────────────────────────────

def init_random():
    upper_bounds = np.minimum(DEMAND[:, np.newaxis], MAX_UNITS)
    chromosome = np.zeros((NUM_SERVICES, NUM_COMPUTERS), dtype=np.int32)
    for service in range(NUM_SERVICES):
        for computer in range(NUM_COMPUTERS):
            chromosome[service, computer] = random.randint(0, int(upper_bounds[service, computer]))
    return repair(chromosome)


def init_population(size):
    return [init_random() for _ in range(size)]


# ── Main Random Search Loop ───────────────────────────────────────────────────

def run_random_search():
    best_ever = None
    best_fitness_ever = 0
    generation_best_history = []

    print("Pokretanje random pretrage...\n")
    for generation in range(GENERATIONS):
        population = init_population(POPULATION_SIZE)
        fitnesses = [fitness(chromosome) for chromosome in population]

        generation_best_index = max(range(len(fitnesses)), key=lambda idx: fitnesses[idx])
        generation_best_fitness = fitnesses[generation_best_index]
        generation_best_history.append(generation_best_fitness)

        if generation_best_fitness > best_fitness_ever:
            best_fitness_ever = generation_best_fitness
            best_ever = population[generation_best_index].copy()

        if generation % 10 == 0:
            print(f"Gen {generation:4d} | Gen: {generation_best_fitness:,} | Best ever: {best_fitness_ever:,}")

    return best_ever, best_fitness_ever, generation_best_history


# ── Output ────────────────────────────────────────────────────────────────────

def print_solution(chromosome, total_profit):
    print("\n" + "=" * 80)
    print(f"  MAKSIMALNA ZARADA: {total_profit:,} dinara")
    print("=" * 80)

    print("\n── Iskoriscenje racunara ────────────────────────────────────────────────")
    time_per_computer = np.sum(TIME * chromosome, axis=0)
    total_used = int(time_per_computer.sum())
    total_capacity = NUM_COMPUTERS * MAX_TIME
    for computer in range(NUM_COMPUTERS):
        used = int(time_per_computer[computer])
        percent = 100 * used / MAX_TIME
        if percent < 99.0:
            print(f"  C{computer+1:3d}: {used:5d}/{MAX_TIME} min  ({percent:5.1f}%)")
    avg_utilization = 100 * total_used / total_capacity
    print(f"\n  Prosecna iskoriscenje: {avg_utilization:.2f}%")

    print("\n── Iskoriscenje traznje servisa ─────────────────────────────────────────")
    allocated_per_service = chromosome.sum(axis=1)
    services_at_max = int(np.sum(allocated_per_service == DEMAND))
    services_unused = int(np.sum(allocated_per_service == 0))
    print(f"  Servisi sa 100% iskoriscenjem: {services_at_max}/{NUM_SERVICES}")
    print(f"  Nekorisceni servisi: {services_unused}/{NUM_SERVICES}")

    print("=" * 80)


# ── Multi-Run ─────────────────────────────────────────────────────────────────

NUM_RUNS = 15

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"rezultati-{PROBLEM_SIZE}")
os.makedirs(RESULTS_DIR, exist_ok=True)


def save_results(run_profits, run_times, best_overall_profit):
    median_profit = statistics.median(run_profits)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"random-{POPULATION_SIZE}-{GENERATIONS}-{NUM_RUNS}runs-{timestamp}.txt"
    filepath = os.path.join(RESULTS_DIR, filename)

    with open(filepath, "w") as file:
        file.write(f"Random Search Parametri:\n")
        file.write(f"  Velicina populacije: {POPULATION_SIZE}\n")
        file.write(f"  Broj generacija: {GENERATIONS}\n")
        file.write(f"  Ukupno evaluacija po pokretanju: {POPULATION_SIZE * GENERATIONS}\n")
        file.write(f"  Broj pokretanja: {NUM_RUNS}\n")
        file.write(f"\n{'='*80}\n")
        file.write(f"  REZULTATI SVIH POKRETANJA\n")
        file.write(f"{'='*80}\n\n")

        for run_index, (profit, elapsed) in enumerate(zip(run_profits, run_times)):
            file.write(f"  Pokretanje {run_index + 1:3d}: {profit:,} dinara  ({elapsed:.2f}s)\n")

        file.write(f"\n{'='*80}\n")
        file.write(f"  MEDIJANA: {median_profit:,.1f} dinara\n")
        file.write(f"  NAJBOLJI: {best_overall_profit:,} dinara\n")
        file.write(f"  NAJGORI:  {min(run_profits):,} dinara\n")
        total_time = sum(run_times)
        file.write(f"\n  UKUPNO VREME:  {total_time:.2f}s\n")
        file.write(f"  PROSECNO VREME: {statistics.mean(run_times):.2f}s\n")
        file.write(f"{'='*80}\n")

    print(f"\nRezultati sacuvani u: {filepath}")
    return filepath


def save_convergence_plot(all_histories):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"random-{POPULATION_SIZE}-{GENERATIONS}-{NUM_RUNS}runs-{timestamp}.png"
    filepath = os.path.join(RESULTS_DIR, filename)

    plt.figure(figsize=(10, 6))
    for run_index, history in enumerate(all_histories):
        running_best = np.maximum.accumulate(history)
        x_values = [(generation + 1) * POPULATION_SIZE for generation in range(len(history))]
        plt.plot(x_values, running_best, linewidth=1, alpha=0.7, label=f"Pokretanje {run_index + 1}")

    plt.xlabel("Broj generacija x velicina populacije (broj evaluacija)")
    plt.ylabel("Vrednost funkcije cilja (zarada)")
    plt.title(f"Konvergencija Random Search — {PROBLEM_SIZE}, pop={POPULATION_SIZE}, gen={GENERATIONS}")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(filepath, dpi=120)
    plt.close()
    print(f"Grafik sacuvan u: {filepath}")
    return filepath


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_profits = []
    run_times = []
    all_histories = []
    best_overall_chromosome = None
    best_overall_profit = 0

    for run_index in range(NUM_RUNS):
        print(f"\n{'#'*80}")
        print(f"  POKRETANJE {run_index + 1}/{NUM_RUNS}")
        print(f"{'#'*80}")

        random.seed(run_index)
        np.random.seed(run_index)
        start_time = time.time()
        best_chromosome, best_profit, generation_best_history = run_random_search()
        all_histories.append(generation_best_history)
        elapsed_time = time.time() - start_time
        run_profits.append(best_profit)
        run_times.append(elapsed_time)

        if best_profit > best_overall_profit:
            best_overall_profit = best_profit
            best_overall_chromosome = best_chromosome

        print(f"\n  Pokretanje {run_index + 1} zavrseno | Zarada: {best_profit:,} | Vreme: {elapsed_time:.2f}s")

    median_profit = statistics.median(run_profits)

    print(f"\n{'='*80}")
    print(f"  SUMARNI REZULTATI ({NUM_RUNS} pokretanja)")
    print(f"{'='*80}")
    for run_index, (profit, elapsed) in enumerate(zip(run_profits, run_times)):
        print(f"  Pokretanje {run_index + 1:3d}: {profit:,} dinara  ({elapsed:.2f}s)")
    print(f"\n  MEDIJANA: {median_profit:,.1f} dinara")
    print(f"  NAJBOLJI: {best_overall_profit:,} dinara")
    print(f"  NAJGORI:  {min(run_profits):,} dinara")
    print(f"  PROSEK:   {statistics.mean(run_profits):,.1f} dinara")
    if len(run_profits) > 1:
        print(f"  STD DEV:  {statistics.stdev(run_profits):,.1f} dinara")
    total_time = sum(run_times)
    print(f"\n  UKUPNO VREME:  {total_time:.2f}s")
    print(f"  PROSECNO VREME: {statistics.mean(run_times):.2f}s")
    print(f"{'='*80}")

    print_solution(best_overall_chromosome, best_overall_profit)
    save_results(run_profits, run_times, best_overall_profit)
    save_convergence_plot(all_histories)
