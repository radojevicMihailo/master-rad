import csv
import random
import os
import statistics
from datetime import datetime

import numpy as np

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

POPULATION_SIZE = 1000
GENERATIONS = 500
TOURNAMENT_SIZE = 5
CROSSOVER_RATE = 0.85
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

def init_greedy():
    chromosome = np.zeros((NUM_SERVICES, NUM_COMPUTERS), dtype=np.int32)
    remaining_demand = DEMAND.copy()
    remaining_time = np.full(NUM_COMPUTERS, MAX_TIME, dtype=np.int32)

    for service, computer in SORTED_PAIRS:
        if remaining_demand[service] <= 0 or remaining_time[computer] < TIME[service, computer]:
            continue
        max_by_time = remaining_time[computer] // TIME[service, computer]
        max_by_demand = remaining_demand[service]
        value = min(int(max_by_time), int(max_by_demand))
        if value > 0:
            value = max(0, value - random.randint(0, value // 3 + 1))
            chromosome[service, computer] = value
            remaining_demand[service] -= value
            remaining_time[computer] -= value * TIME[service, computer]

    return repair(chromosome)


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

def uniform_crossover(parent1, parent2):
    if random.random() > CROSSOVER_RATE:
        return parent1.copy(), parent2.copy()

    mask = np.random.randint(0, 2, size=(NUM_SERVICES, NUM_COMPUTERS), dtype=np.bool_)
    child1 = np.where(mask, parent1, parent2)
    child2 = np.where(mask, parent2, parent1)

    return repair(child1), repair(child2)


# ── Mutation ──────────────────────────────────────────────────────────────────

def mutate(chromosome):
    # Operator 1: Point mutation
    per_cell_probability = MUTATION_RATE / NUM_COMPUTERS
    mutation_mask = np.random.random((NUM_SERVICES, NUM_COMPUTERS)) < per_cell_probability
    if mutation_mask.any():
        deltas_max = np.maximum(1, chromosome // 3)
        # Generate random deltas in range [-delta_max, +delta_max] for each cell
        random_deltas = (np.random.random(chromosome.shape) * (2 * deltas_max + 1) - deltas_max).astype(np.int32)
        chromosome[mutation_mask] += random_deltas[mutation_mask]
        np.clip(chromosome, 0, MAX_UNITS, out=chromosome)

    # Operator 2: Column swap (5%)
    if random.random() < 0.05:
        computer1, computer2 = random.sample(range(NUM_COMPUTERS), 2)
        chromosome[:, [computer1, computer2]] = chromosome[:, [computer2, computer1]]

    # Operator 3: Row redistribution (5%)
    if random.random() < 0.05:
        service = random.randint(0, NUM_SERVICES - 1)
        total = int(chromosome[service].sum())
        if total > 0:
            weights = PROFIT_PER_MINUTE[service]
            weights_sum = weights.sum()
            chromosome[service] = (total * weights / weights_sum).astype(np.int32)

    return repair(chromosome)


# ── Main GA Loop ──────────────────────────────────────────────────────────────

def run_ga():
    print("\nInicijalizacija populacije...")
    population = init_population(POPULATION_SIZE)
    best_ever = None
    best_fitness_ever = 0
    stagnation_counter = 0

    print("Pokretanje evolucije...\n")
    for generation in range(GENERATIONS):
        fitnesses = [fitness(chromosome) for chromosome in population]

        generation_best_index = max(range(len(fitnesses)), key=lambda idx: fitnesses[idx])
        generation_best_fitness = fitnesses[generation_best_index]

        if generation_best_fitness > best_fitness_ever:
            best_fitness_ever = generation_best_fitness
            best_ever = population[generation_best_index].copy()
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        if generation % 25 == 0:
            average_fitness = sum(fitnesses) / len(fitnesses)
            print(f"Gen {generation:4d} | Best: {generation_best_fitness:,} | Avg: {average_fitness:,.1f} | "
                  f"All-time best: {best_fitness_ever:,}")

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
            child1, child2 = uniform_crossover(parent1, parent2)
            child1 = mutate(child1)
            child2 = mutate(child2)
            new_population.append(child1)
            if len(new_population) < POPULATION_SIZE:
                new_population.append(child2)

        population = new_population

    return best_ever, best_fitness_ever


# ── Output ────────────────────────────────────────────────────────────────────

def print_solution(chromosome, total_profit):
    print("\n" + "=" * 80)
    print(f"  MAKSIMALNA ZARADA: {total_profit:,} dinara")
    print("=" * 80)

    # Computer utilization
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

    # Service demand utilization
    print("\n── Iskoriscenje traznje servisa ─────────────────────────────────────────")
    allocated_per_service = chromosome.sum(axis=1)
    services_at_max = int(np.sum(allocated_per_service == DEMAND))
    services_unused = int(np.sum(allocated_per_service == 0))
    print(f"  Servisi sa 100% iskoriscenjem: {services_at_max}/{NUM_SERVICES}")
    print(f"  Nekorisceni servisi: {services_unused}/{NUM_SERVICES}")

    # Partially used services
    print("\n  Delimicno korisceni servisi:")
    for service in range(NUM_SERVICES):
        allocated = int(allocated_per_service[service])
        if 0 < allocated < DEMAND[service]:
            percent = 100 * allocated / DEMAND[service]
            print(f"    S{service+1:3d}: {allocated:5d}/{DEMAND[service]:5d}  ({percent:5.1f}%)")

    # Top 15 most profitable assignments
    print("\n── Top 15 najisplativijih dodela ────────────────────────────────────────")
    contributions = PROFIT * chromosome
    nonzero = np.argwhere(chromosome > 0)
    assignments = [
        (int(contributions[service, computer]), service, computer, int(chromosome[service, computer]), int(PROFIT[service, computer]))
        for service, computer in nonzero
    ]
    assignments.sort(reverse=True)
    print(f"  {'Servis':<10} {'Racunar':<10} {'Kolicina':>10} {'Cena/kom':>10} {'Doprinos':>12}")
    print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*12}")
    for contribution, service, computer, quantity, price in assignments[:15]:
        print(f"  S{service+1:<9} C{computer+1:<9} {quantity:>10,} {price:>10} {contribution:>12,}")

    print("=" * 80)


# ── Multi-Run with Median ────────────────────────────────────────────────────

NUM_RUNS = 10

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"rezultati-{PROBLEM_SIZE}")
os.makedirs(RESULTS_DIR, exist_ok=True)


def save_results(run_profits, best_overall_profit, best_overall_chromosome):
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

        for run_index, profit in enumerate(run_profits):
            file.write(f"  Pokretanje {run_index + 1:3d}: {profit:,} dinara\n")

        file.write(f"\n{'='*80}\n")
        file.write(f"  MEDIJANA: {median_profit:,.1f} dinara\n")
        file.write(f"  NAJBOLJI: {best_overall_profit:,} dinara\n")
        file.write(f"  NAJGORI:  {min(run_profits):,} dinara\n")
        file.write(f"  PROSEK:   {statistics.mean(run_profits):,.1f} dinara\n")
        if len(run_profits) > 1:
            file.write(f"  STD DEV:  {statistics.stdev(run_profits):,.1f} dinara\n")
        file.write(f"{'='*80}\n")

    print(f"\nRezultati sacuvani u: {filepath}")
    return filepath


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_profits = []
    best_overall_chromosome = None
    best_overall_profit = 0

    for run_index in range(NUM_RUNS):
        print(f"\n{'#'*80}")
        print(f"  POKRETANJE {run_index + 1}/{NUM_RUNS}")
        print(f"{'#'*80}")

        random.seed(run_index)
        np.random.seed(run_index)
        best_chromosome, best_profit = run_ga()
        run_profits.append(best_profit)

        if best_profit > best_overall_profit:
            best_overall_profit = best_profit
            best_overall_chromosome = best_chromosome

        print(f"\n  Pokretanje {run_index + 1} zavrseno | Zarada: {best_profit:,}")

    median_profit = statistics.median(run_profits)

    print(f"\n{'='*80}")
    print(f"  SUMARNI REZULTATI ({NUM_RUNS} pokretanja)")
    print(f"{'='*80}")
    for run_index, profit in enumerate(run_profits):
        print(f"  Pokretanje {run_index + 1:3d}: {profit:,} dinara")
    print(f"\n  MEDIJANA: {median_profit:,.1f} dinara")
    print(f"  NAJBOLJI: {best_overall_profit:,} dinara")
    print(f"  NAJGORI:  {min(run_profits):,} dinara")
    print(f"  PROSEK:   {statistics.mean(run_profits):,.1f} dinara")
    if len(run_profits) > 1:
        print(f"  STD DEV:  {statistics.stdev(run_profits):,.1f} dinara")
    print(f"{'='*80}")

    print_solution(best_overall_chromosome, best_overall_profit)
    save_results(run_profits, best_overall_profit, best_overall_chromosome)
