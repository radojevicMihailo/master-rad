import random
import os

# ── Load Data from Files ──────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "podaci 100x100")

def load_matrix(filename):
    """Load a 100x100 matrix from a tab-separated file (skip header row and first column)."""
    filepath = os.path.join(DATA_DIR, filename)
    matrix = []
    with open(filepath, "r") as file:
        lines = file.readlines()
        for line in lines[1:]:  # skip header
            parts = line.strip().split("\t")
            if len(parts) > 1:
                row = [int(value) for value in parts[1:]]  # skip service name column
                matrix.append(row)
    return matrix

def load_demand(filename):
    """Load demand values from a tab-separated file (skip header, take second column)."""
    filepath = os.path.join(DATA_DIR, filename)
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

print("Ucitavanje podataka...")
TIME = load_matrix("vremena_izvrsavanja.txt")
PROFIT = load_matrix("zarada_po_servisu.txt")
DEMAND = load_demand("zahtevi_za_servisima.txt")

NUM_SERVICES = len(TIME)
NUM_COMPUTERS = len(TIME[0])
MAX_TIME = 2880

print(f"Dimenzije problema: {NUM_SERVICES} servisa x {NUM_COMPUTERS} racunara")
print(f"Traznja po servisu: {DEMAND[0]} (uniformna)")

# ── GA Hyperparameters (adjusted for 100x100) ────────────────────────────────

POPULATION_SIZE = 300
GENERATIONS = 2000
TOURNAMENT_SIZE = 5
CROSSOVER_RATE = 0.85
MUTATION_RATE = 0.15
ELITE_COUNT = 15

print("\nGA Hyperparameters:")
print(f"  Velicina populacije: {POPULATION_SIZE}")
print(f"  Generacije: {GENERATIONS}")
print(f"  Tournament size: {TOURNAMENT_SIZE}")
print(f"  Crossover rate: {CROSSOVER_RATE:.2f}")
print(f"  Mutation rate: {MUTATION_RATE:.2f}")
print(f"  Elitizam: {ELITE_COUNT} najboljih zadrzavano")

# ── Precomputed Helpers ───────────────────────────────────────────────────────

PROFIT_PER_MINUTE = [
    [PROFIT[service][computer] / TIME[service][computer] for computer in range(NUM_COMPUTERS)]
    for service in range(NUM_SERVICES)
]

MAX_UNITS = [
    [MAX_TIME // TIME[service][computer] for computer in range(NUM_COMPUTERS)]
    for service in range(NUM_SERVICES)
]

# Top profit/minute pairs for greedy init (precomputed once)
SORTED_PAIRS = sorted(
    [(service, computer) for service in range(NUM_SERVICES) for computer in range(NUM_COMPUTERS)],
    key=lambda pair: PROFIT_PER_MINUTE[pair[0]][pair[1]],
    reverse=True,
)

# For each computer, services sorted by profit/minute (worst first) — used in repair
SERVICES_BY_RATIO_PER_COMPUTER = [
    sorted(range(NUM_SERVICES), key=lambda service: PROFIT_PER_MINUTE[service][computer])
    for computer in range(NUM_COMPUTERS)
]

# For each computer, services sorted by profit/minute (best first) — used in fill
SERVICES_BEST_FIRST_PER_COMPUTER = [
    sorted(range(NUM_SERVICES), key=lambda service: PROFIT_PER_MINUTE[service][computer], reverse=True)
    for computer in range(NUM_COMPUTERS)
]


# ── Repair ────────────────────────────────────────────────────────────────────

def repair(chromosome):
    """Ensure chromosome satisfies all time and demand constraints."""
    # 1. Clip negatives
    for service in range(NUM_SERVICES):
        for computer in range(NUM_COMPUTERS):
            if chromosome[service][computer] < 0:
                chromosome[service][computer] = 0

    # 2. Enforce demand constraints (row-wise)
    for service in range(NUM_SERVICES):
        total = sum(chromosome[service])
        if total > DEMAND[service]:
            ratio = DEMAND[service] / total
            for computer in range(NUM_COMPUTERS):
                chromosome[service][computer] = int(chromosome[service][computer] * ratio)

    # 3. Enforce time constraints (column-wise)
    for computer in range(NUM_COMPUTERS):
        used = sum(TIME[service][computer] * chromosome[service][computer] for service in range(NUM_SERVICES))
        if used > MAX_TIME:
            # Remove from worst profit/minute services first
            for service in SERVICES_BY_RATIO_PER_COMPUTER[computer]:
                if used <= MAX_TIME:
                    break
                if chromosome[service][computer] > 0:
                    # Remove as many units as needed from this service
                    excess = used - MAX_TIME
                    units_to_remove = min(chromosome[service][computer], (excess + TIME[service][computer] - 1) // TIME[service][computer])
                    chromosome[service][computer] -= units_to_remove
                    used -= units_to_remove * TIME[service][computer]

    # 4. Fill leftover time with best profit/minute services
    for computer in range(NUM_COMPUTERS):
        used = sum(TIME[service][computer] * chromosome[service][computer] for service in range(NUM_SERVICES))
        remaining = MAX_TIME - used
        for service in SERVICES_BEST_FIRST_PER_COMPUTER[computer]:
            if remaining < TIME[service][computer]:
                continue
            demand_left = DEMAND[service] - sum(chromosome[service])
            if demand_left <= 0:
                continue
            add = min(demand_left, remaining // TIME[service][computer])
            if add > 0:
                chromosome[service][computer] += add
                remaining -= add * TIME[service][computer]
            if remaining < 5:  # no time for any service
                break

    return chromosome


# ── Fitness ───────────────────────────────────────────────────────────────────

def fitness(chromosome):
    total = 0
    for service in range(NUM_SERVICES):
        for computer in range(NUM_COMPUTERS):
            total += PROFIT[service][computer] * chromosome[service][computer]
    return total


# ── Initialization ────────────────────────────────────────────────────────────

def init_greedy():
    chromosome = [[0] * NUM_COMPUTERS for _ in range(NUM_SERVICES)]
    remaining_demand = DEMAND[:]
    remaining_time = [MAX_TIME] * NUM_COMPUTERS

    for service, computer in SORTED_PAIRS:
        if remaining_demand[service] <= 0 or remaining_time[computer] < TIME[service][computer]:
            continue
        max_by_time = remaining_time[computer] // TIME[service][computer]
        max_by_demand = remaining_demand[service]
        value = min(max_by_time, max_by_demand)
        if value > 0:
            value = max(0, value - random.randint(0, value // 3 + 1))
            chromosome[service][computer] = value
            remaining_demand[service] -= value
            remaining_time[computer] -= value * TIME[service][computer]

    return repair(chromosome)


def init_random():
    chromosome = [
        [random.randint(0, min(DEMAND[service], MAX_UNITS[service][computer]))
         for computer in range(NUM_COMPUTERS)]
        for service in range(NUM_SERVICES)
    ]
    return repair(chromosome)


def init_population(size):
    greedy_count = int(0.3 * size)
    population = [init_greedy() for _ in range(greedy_count)]
    population += [init_random() for _ in range(size - greedy_count)]
    return population


# ── Selection ─────────────────────────────────────────────────────────────────

def tournament_select(population, fitnesses, tournament_size=TOURNAMENT_SIZE):
    indices = random.sample(range(len(population)), tournament_size)
    best_index = max(indices, key=lambda idx: fitnesses[idx])
    return [row[:] for row in population[best_index]]


# ── Crossover ─────────────────────────────────────────────────────────────────

def uniform_crossover(parent1, parent2):
    if random.random() > CROSSOVER_RATE:
        return [row[:] for row in parent1], [row[:] for row in parent2]

    child1 = [[0] * NUM_COMPUTERS for _ in range(NUM_SERVICES)]
    child2 = [[0] * NUM_COMPUTERS for _ in range(NUM_SERVICES)]

    for service in range(NUM_SERVICES):
        for computer in range(NUM_COMPUTERS):
            if random.random() < 0.5:
                child1[service][computer] = parent1[service][computer]
                child2[service][computer] = parent2[service][computer]
            else:
                child1[service][computer] = parent2[service][computer]
                child2[service][computer] = parent1[service][computer]

    return repair(child1), repair(child2)


# ── Mutation ──────────────────────────────────────────────────────────────────

def mutate(chromosome):
    # Operator 1: Point mutation (sparser for 100x100)
    per_cell_probability = MUTATION_RATE / NUM_COMPUTERS
    for service in range(NUM_SERVICES):
        for computer in range(NUM_COMPUTERS):
            if random.random() < per_cell_probability:
                delta = max(1, chromosome[service][computer] // 3)
                chromosome[service][computer] += random.randint(-delta, delta)
                chromosome[service][computer] = max(0, min(chromosome[service][computer], MAX_UNITS[service][computer]))

    # Operator 2: Column swap (5%)
    if random.random() < 0.05:
        computer1, computer2 = random.sample(range(NUM_COMPUTERS), 2)
        for service in range(NUM_SERVICES):
            chromosome[service][computer1], chromosome[service][computer2] = chromosome[service][computer2], chromosome[service][computer1]

    # Operator 3: Row redistribution (5%)
    if random.random() < 0.05:
        service = random.randint(0, NUM_SERVICES - 1)
        total = sum(chromosome[service])
        if total > 0:
            weights = [PROFIT_PER_MINUTE[service][computer] for computer in range(NUM_COMPUTERS)]
            weights_sum = sum(weights)
            chromosome[service] = [int(total * weight / weights_sum) for weight in weights]

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
            best_ever = [row[:] for row in population[generation_best_index]]
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
        new_population = [[row[:] for row in population[idx]] for idx in elite_indices]

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
    total_used = 0
    total_capacity = NUM_COMPUTERS * MAX_TIME
    for computer in range(NUM_COMPUTERS):
        used = sum(TIME[service][computer] * chromosome[service][computer] for service in range(NUM_SERVICES))
        total_used += used
        percent = 100 * used / MAX_TIME
        if percent < 99.0:
            print(f"  C{computer+1:3d}: {used:5d}/{MAX_TIME} min  ({percent:5.1f}%)")
    avg_utilization = 100 * total_used / total_capacity
    print(f"\n  Prosecna iskoriscenje: {avg_utilization:.2f}%")

    # Service demand utilization
    print("\n── Iskoriscenje traznje servisa ─────────────────────────────────────────")
    services_at_max = 0
    services_unused = 0
    for service in range(NUM_SERVICES):
        allocated = sum(chromosome[service])
        if allocated == DEMAND[service]:
            services_at_max += 1
        elif allocated == 0:
            services_unused += 1
    print(f"  Servisi sa 100% iskoriscenjem: {services_at_max}/{NUM_SERVICES}")
    print(f"  Nekorisceni servisi: {services_unused}/{NUM_SERVICES}")

    # Partially used services
    print("\n  Delimicno korisceni servisi:")
    for service in range(NUM_SERVICES):
        allocated = sum(chromosome[service])
        if 0 < allocated < DEMAND[service]:
            percent = 100 * allocated / DEMAND[service]
            print(f"    S{service+1:3d}: {allocated:5d}/{DEMAND[service]:5d}  ({percent:5.1f}%)")

    # Top 15 most profitable assignments
    print("\n── Top 15 najisplativijih dodela ────────────────────────────────────────")
    assignments = [
        (PROFIT[service][computer] * chromosome[service][computer], service, computer, chromosome[service][computer], PROFIT[service][computer])
        for service in range(NUM_SERVICES)
        for computer in range(NUM_COMPUTERS)
        if chromosome[service][computer] > 0
    ]
    assignments.sort(reverse=True)
    print(f"  {'Servis':<10} {'Racunar':<10} {'Kolicina':>10} {'Cena/kom':>10} {'Doprinos':>12}")
    print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*12}")
    for contribution, service, computer, quantity, price in assignments[:15]:
        print(f"  S{service+1:<9} C{computer+1:<9} {quantity:>10,} {price:>10} {contribution:>12,}")

    print("=" * 80)


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    random.seed(42)
    best_chromosome, best_profit = run_ga()
    print_solution(best_chromosome, best_profit)
