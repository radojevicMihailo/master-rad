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
    """Load matrix from a CSV file (skip header row and first column)."""
    matrix = []
    with open(filepath, "r") as file:
        reader = csv.reader(file)
        next(reader)  # skip header
        for row in reader:
            matrix.append([int(value) for value in row[1:]])
    return matrix


def load_demand_csv(filepath):
    """Load demand values from a CSV file (Servis_N,value per line, no header)."""
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
    """Load matrix from a tab-separated file (skip header row and first column)."""
    matrix = []
    with open(filepath, "r") as file:
        lines = file.readlines()
        for line in lines[1:]:  # skip header
            parts = line.strip().split("\t")
            if len(parts) > 1:
                row = [int(value) for value in parts[1:]]
                matrix.append(row)
    return matrix


def load_demand_tsv(filepath):
    """Load demand values from a tab-separated file (skip header, take second column)."""
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
    """Load TIME, PROFIT, DEMAND matrices based on problem size and file format."""
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
print(f"Traznja po servisu: {DEMAND[0]} (uniformna)")

# ── GA Hyperparameters ────────────────────────────────

POPULATION_SIZE = 100
GENERATIONS = 200
TOURNAMENT_SIZE = 10
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.15
ELITE_COUNT = 10

print("\nGA Hiperparametri:")
print(f"  Velicina populacije: {POPULATION_SIZE}")
print(f"  Broj generacija: {GENERATIONS}")
print(f"  Velicina turnira: {TOURNAMENT_SIZE}")
print(f"  Verovatnoca ukrstanja: {CROSSOVER_RATE:.2f}")
print(f"  Verovatnoca mutacije: {MUTATION_RATE:.2f}")
print(f"  Elitizam: {ELITE_COUNT} najboljih zadrzavano")

# ── Precomputed Helpers ───────────────────────────────────────────────────────

PROFIT_PER_MINUTE = PROFIT.astype(np.float64) / TIME.astype(np.float64)

MAX_UNITS = (MAX_TIME // TIME).astype(np.int32)

# Top profit/minute pairs for greedy init (precomputed once)
_flat_indices = np.argsort(PROFIT_PER_MINUTE.ravel())[::-1]
SORTED_PAIRS = list(zip(*np.unravel_index(_flat_indices, PROFIT_PER_MINUTE.shape)))

# For each computer, services sorted by profit/minute (worst first) — used in repair
SERVICES_BY_RATIO_PER_COMPUTER = [
    np.argsort(PROFIT_PER_MINUTE[:, computer]).tolist()
    for computer in range(NUM_COMPUTERS)
]

# For each computer, services sorted by profit/minute (best first) — used in fill
SERVICES_BEST_FIRST_PER_COMPUTER = [
    np.argsort(PROFIT_PER_MINUTE[:, computer])[::-1].tolist()
    for computer in range(NUM_COMPUTERS)
]


# ── Repair ────────────────────────────────────────────────────────────────────

def repair(chromosome):
    """Ensure chromosome satisfies all time and demand constraints."""
    # 1. Clip negatives
    np.clip(chromosome, 0, None, out=chromosome)

    # 2. Enforce demand constraints (row-wise)
    row_totals = chromosome.sum(axis=1)
    over_demand = row_totals > DEMAND
    if over_demand.any():
        ratios = DEMAND[over_demand].astype(np.float64) / row_totals[over_demand]
        chromosome[over_demand] = (chromosome[over_demand] * ratios[:, np.newaxis]).astype(np.int32)

    # 3. Enforce time constraints (column-wise)
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

    # 4. Fill leftover time with best profit/minute services
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


# ── Selection ─────────────────────────────────────────────────────────────────

def tournament_select(population, fitnesses, tournament_size=TOURNAMENT_SIZE):
    indices = random.sample(range(len(population)), tournament_size)
    best_index = max(indices, key=lambda idx: fitnesses[idx])
    return population[best_index].copy()


# ── Crossover ─────────────────────────────────────────────────────────────────

def one_point_crossover(parent1, parent2):
    if random.random() > CROSSOVER_RATE:
        return parent1.copy(), parent2.copy()

    total_genes = NUM_SERVICES * NUM_COMPUTERS
    crossover_point = random.randint(1, total_genes - 1)

    flat_parent1 = parent1.ravel()
    flat_parent2 = parent2.ravel()

    flat_child1 = np.concatenate([flat_parent1[:crossover_point], flat_parent2[crossover_point:]])
    flat_child2 = np.concatenate([flat_parent2[:crossover_point], flat_parent1[crossover_point:]])

    child1 = flat_child1.reshape(NUM_SERVICES, NUM_COMPUTERS).copy()
    child2 = flat_child2.reshape(NUM_SERVICES, NUM_COMPUTERS).copy()

    return repair(child1), repair(child2)


# ── Mutation ──────────────────────────────────────────────────────────────────

def mutate(chromosome):
    if random.random() < MUTATION_RATE:
        service = random.randint(0, NUM_SERVICES - 1)
        computer = random.randint(0, NUM_COMPUTERS - 1)
        chromosome[service, computer] = random.randint(0, int(MAX_UNITS[service, computer]))

    return repair(chromosome)


# ── Main GA Loop ──────────────────────────────────────────────────────────────

def run_ga():
    print("\nInicijalizacija populacije...")
    population = init_population(POPULATION_SIZE)
    best_ever = None
    best_fitness_ever = 0
    stagnation_counter = 0

    print("Pokretanje evolucije...\n")
    generation_best_history = []
    for generation in range(GENERATIONS):
        fitnesses = [fitness(chromosome) for chromosome in population]

        generation_best_index = max(range(len(fitnesses)), key=lambda idx: fitnesses[idx])
        generation_best_fitness = fitnesses[generation_best_index]
        generation_best_history.append(generation_best_fitness)

        if generation_best_fitness > best_fitness_ever:
            best_fitness_ever = generation_best_fitness
            best_ever = population[generation_best_index].copy()
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        if generation % 10 == 0:
            print(f"Gen {generation:4d} | Best: {generation_best_fitness:,}")

        # Diversity injection on stagnation
        if stagnation_counter > 0 and stagnation_counter % 100 == 0:
            sorted_indices = sorted(range(len(fitnesses)), key=lambda idx: fitnesses[idx])
            num_replace = POPULATION_SIZE // 5
            for idx in sorted_indices[:num_replace]:
                population[idx] = init_random()

        # Elitism: keep top ELITE_COUNT
        elite_indices = sorted(range(len(fitnesses)), key=lambda idx: fitnesses[idx], reverse=True)[:ELITE_COUNT]
        new_population = [population[idx].copy() for idx in elite_indices]

        # Fill the rest via selection, crossover, mutation
        while len(new_population) < POPULATION_SIZE:
            parent1 = tournament_select(population, fitnesses)
            parent2 = tournament_select(population, fitnesses)
            child1, child2 = one_point_crossover(parent1, parent2)
            child1 = mutate(child1)
            child2 = mutate(child2)
            new_population.append(child1)
            if len(new_population) < POPULATION_SIZE:
                new_population.append(child2)

        population = new_population

    return best_ever, best_fitness_ever, generation_best_history

# ── Multi-Run with Median ────────────────────────────────────────────────────

NUM_RUNS = 15

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"rezultati-{PROBLEM_SIZE}")
os.makedirs(RESULTS_DIR, exist_ok=True)


def save_results(run_profits, run_times, best_overall_profit, all_histories):
    """Save all run results and median to file."""
    median_profit = statistics.median(run_profits)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"ga-{POPULATION_SIZE}-{GENERATIONS}-{ELITE_COUNT}-{NUM_RUNS}runs-{timestamp}.txt"
    filepath = os.path.join(RESULTS_DIR, filename)

    with open(filepath, "w") as file:
        file.write(f"GA Hiperparametri:\n")
        file.write(f"  Velicina populacije: {POPULATION_SIZE}\n")
        file.write(f"  Broj generacija: {GENERATIONS}\n")
        file.write(f"  Velicina turnira: {TOURNAMENT_SIZE}\n")
        file.write(f"  Verovatnoca ukrstanja: {CROSSOVER_RATE:.2f}\n")
        file.write(f"  Verovatnoca mutacije: {MUTATION_RATE:.2f}\n")
        file.write(f"  Elitizam: {ELITE_COUNT} najboljih zadrzavano\n")
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

        file.write(f"\n{'='*80}\n")
        file.write(f"  NAJBOLJI PO GENERACIJAMA\n")
        file.write(f"{'='*80}\n")
        for run_index, history in enumerate(all_histories):
            file.write(f"\n  Pokretanje {run_index + 1}:\n")
            for generation, best in enumerate(history):
                file.write(f"    Gen {generation:4d}: {best:,}\n")

    print(f"\nRezultati sacuvani u: {filepath}")
    return filepath


def save_convergence_plot(all_histories):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"ga-{POPULATION_SIZE}-{GENERATIONS}-{ELITE_COUNT}-{NUM_RUNS}runs-{timestamp}.png"
    filepath = os.path.join(RESULTS_DIR, filename)

    plt.figure(figsize=(10, 6))
    for run_index, history in enumerate(all_histories):
        x_values = [(generation + 1) * POPULATION_SIZE for generation in range(len(history))]
        plt.plot(x_values, history, linewidth=1, alpha=0.7, label=f"Pokretanje {run_index + 1}")

    plt.xlabel("Broj generacija x velicina populacije (broj evaluacija)")
    plt.ylabel("Vrednost funkcije cilja (zarada)")
    plt.title(f"Konvergencija GA — {PROBLEM_SIZE}, pop={POPULATION_SIZE}, gen={GENERATIONS}")
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
    best_overall_profit = 0

    for run_index in range(NUM_RUNS):
        print(f"\n{'#'*80}")
        print(f"  POKRETANJE {run_index + 1}/{NUM_RUNS}")
        print(f"{'#'*80}")

        random.seed(run_index)
        np.random.seed(run_index)
        start_time = time.time()
        best_chromosome, best_profit, generation_best_history = run_ga()
        all_histories.append(generation_best_history)
        elapsed_time = time.time() - start_time
        run_profits.append(best_profit)
        run_times.append(elapsed_time)

        if best_profit > best_overall_profit:
            best_overall_profit = best_profit

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
    total_time = sum(run_times)
    print(f"\n  UKUPNO VREME:  {total_time:.2f}s")
    print(f"  PROSECNO VREME: {statistics.mean(run_times):.2f}s")
    print(f"{'='*80}")

    print("\n" + "=" * 80)
    print(f"  MAKSIMALNA ZARADA: {best_overall_profit:,} dinara")
    print("=" * 80)
    save_results(run_profits, run_times, best_overall_profit, all_histories)
    save_convergence_plot(all_histories)