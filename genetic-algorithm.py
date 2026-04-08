import random

# ── Constants ─────────────────────────────────────────────────────────────────

SERVICE_COUNT = 10
COMPUTER_COUNT = 10
MAX_TIME = 2880  # 48 hours in minutes

# Time matrix: TIME[i][j] = minutes to run service i on computer j
TIME = [
    [ 5,  7,  4, 10,  6,  8,  5,  9,  7,  6],  # S1
    [ 6, 12,  8, 15, 10,  7,  9, 11,  8, 14],  # S2
    [13, 14,  9, 17, 12, 15, 10, 11, 13, 16],  # S3
    [ 8,  5,  7,  6,  9, 10,  4, 12, 15,  7],  # S4
    [10, 10, 10, 10, 10, 10, 10, 10, 10, 10],  # S5
    [ 5, 15,  5, 15,  5, 15,  5, 15,  5, 15],  # S6
    [12,  8, 14,  9, 11,  7, 13, 10,  6,  8],  # S7
    [ 7,  7,  7,  7,  7,  7,  7,  7,  7,  7],  # S8
    [15, 12, 10,  8, 14,  9, 11, 13,  7, 16],  # S9
    [ 9, 11, 13, 15,  7,  8, 10, 12, 14,  6],  # S10
]

# Profit matrix: PROFIT[i][j] = dinars per execution of service i on computer j
PROFIT = [
    [10,  8,  6,  9,  7, 11, 10,  8,  9,  7],  # S1
    [18, 20, 15, 17, 19, 16, 18, 20, 15, 17],  # S2
    [15, 16, 13, 17, 14, 18, 15, 16, 13, 17],  # S3
    [12, 14, 11, 13, 15, 12, 14, 11, 13, 15],  # S4
    [25, 22, 20, 24, 21, 23, 25, 22, 20, 24],  # S5
    [10, 10, 10, 10, 10, 10, 10, 10, 10, 10],  # S6
    [30, 28, 32, 29, 31, 27, 30, 28, 32, 29],  # S7
    [14, 15, 16, 17, 18, 19, 20, 21, 22, 23],  # S8
    [40, 35, 38, 42, 37, 39, 40, 35, 38, 42],  # S9
    [22, 24, 26, 28, 20, 21, 23, 25, 27, 29],  # S10
]

# Maximum demand per service over 48 hours
DEMAND = [1000, 600, 500, 800, 400, 1200, 300, 700, 200, 450]

# GA hyperparameters
POPULATION_SIZE = 200
GENERATIONS = 1000
TOURNAMENT_SIZE = 5
CROSSOVER_RATE = 0.85
MUTATION_RATE = 0.15
ELITE_COUNT = 10

# Precomputed helpers
PROFIT_PER_MIN = [
    [PROFIT[i][j] / TIME[i][j] for j in range(COMPUTER_COUNT)]
    for i in range(SERVICE_COUNT)
]

MAX_UNITS = [
    [MAX_TIME // TIME[i][j] for j in range(COMPUTER_COUNT)]
    for i in range(SERVICE_COUNT)
]

# All (i, j) pairs sorted by profit-per-minute descending (used in greedy init)
SORTED_PAIRS = sorted(
    [(i, j) for i in range(SERVICE_COUNT) for j in range(COMPUTER_COUNT)],
    key=lambda p: PROFIT_PER_MIN[p[0]][p[1]],
    reverse=True,
)


# ── Repair ────────────────────────────────────────────────────────────────────

def repair(chromosome):
    """Ensure chromosome satisfies all time and demand constraints (in-place + return)."""
    # 1. Clip negatives
    for service in range(SERVICE_COUNT):
        for computer in range(COMPUTER_COUNT):
            if chromosome[service][computer] < 0:
                chromosome[service][computer] = 0

    # 2. Enforce demand constraints (row-wise)
    for service in range(SERVICE_COUNT):
        total = sum(chromosome[service])
        if total > DEMAND[service]:
            for computer in range(COMPUTER_COUNT):
                chromosome[service][computer] = int(chromosome[service][computer] * DEMAND[service] / total)

    # 3. Enforce time constraints (column-wise)
    for computer in range(COMPUTER_COUNT):
        used = sum(TIME[service][computer] * chromosome[service][computer] for service in range(SERVICE_COUNT))
        while used > MAX_TIME:
            worst_service = -1
            worst_ratio = float('inf')
            for service in range(SERVICE_COUNT):
                if chromosome[service][computer] > 0 and PROFIT_PER_MIN[service][computer] < worst_ratio:
                    worst_ratio = PROFIT_PER_MIN[service][computer]
                    worst_service = service
            if worst_service == -1:
                break
            chromosome[worst_service][computer] -= 1
            used -= TIME[worst_service][computer]

    # 4. Fill leftover time with the best profit/min services that have remaining demand
    for computer in range(COMPUTER_COUNT):
        used = sum(TIME[service][computer] * chromosome[service][computer] for service in range(SERVICE_COUNT))
        remaining = MAX_TIME - used
        # Services sorted by profit/min on this computer, best first
        service_order = sorted(range(SERVICE_COUNT), key=lambda service: PROFIT_PER_MIN[service][computer], reverse=True)
        for service in service_order:
            if remaining < TIME[service][computer]:
                continue
            demand_left = DEMAND[service] - sum(chromosome[service])
            if demand_left <= 0:
                continue
            add = min(demand_left, remaining // TIME[service][computer])
            if add > 0:
                chromosome[service][computer] += add
                remaining -= add * TIME[service][computer]

    return chromosome


# ── Fitness ───────────────────────────────────────────────────────────────────

def fitness(chromosome):
    return sum(
        PROFIT[service][computer] * chromosome[service][computer]
        for service in range(SERVICE_COUNT)
        for computer in range(COMPUTER_COUNT)
    )


# ── Initialization ────────────────────────────────────────────────────────────

def init_greedy():
    chromosome = [[0] * COMPUTER_COUNT for _ in range(SERVICE_COUNT)]
    remaining_demand = DEMAND[:]
    remaining_time = [MAX_TIME] * COMPUTER_COUNT

    for service, computer in SORTED_PAIRS:
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
         for computer in range(COMPUTER_COUNT)]
        for service in range(SERVICE_COUNT)
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

    child1 = [[0] * COMPUTER_COUNT for _ in range(SERVICE_COUNT)]
    child2 = [[0] * COMPUTER_COUNT for _ in range(SERVICE_COUNT)]

    for service in range(SERVICE_COUNT):
        for computer in range(COMPUTER_COUNT):
            if random.random() < 0.5:
                child1[service][computer], child2[service][computer] = parent1[service][computer], parent2[service][computer]
            else:
                child1[service][computer], child2[service][computer] = parent2[service][computer], parent1[service][computer]

    return repair(child1), repair(child2)


# ── Mutation ──────────────────────────────────────────────────────────────────

def mutate(chromosome):
    # Operator 1: Point mutation
    per_cell_probability = MUTATION_RATE / COMPUTER_COUNT
    for service in range(SERVICE_COUNT):
        for computer in range(COMPUTER_COUNT):
            if random.random() < per_cell_probability:
                delta = max(1, chromosome[service][computer] // 3)
                chromosome[service][computer] += random.randint(-delta, delta)
                chromosome[service][computer] = max(0, min(chromosome[service][computer], MAX_UNITS[service][computer]))

    # Operator 2: Column swap (5% chance)
    if random.random() < 0.05:
        computer1, computer2 = random.sample(range(COMPUTER_COUNT), 2)
        for service in range(SERVICE_COUNT):
            chromosome[service][computer1], chromosome[service][computer2] = chromosome[service][computer2], chromosome[service][computer1]

    # Operator 3: Row redistribution (5% chance)
    if random.random() < 0.05:
        service = random.randint(0, SERVICE_COUNT - 1)
        total = sum(chromosome[service])
        if total > 0:
            weights = [PROFIT_PER_MIN[service][computer] for computer in range(COMPUTER_COUNT)]
            weights_sum = sum(weights)
            chromosome[service] = [int(total * weight / weights_sum) for weight in weights]

    return repair(chromosome)


# ── Main GA Loop ──────────────────────────────────────────────────────────────

def run_ga():
    population = init_population(POPULATION_SIZE)
    best_ever = None
    best_fitness_ever = 0
    stagnation_counter = 0

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

        if generation % 50 == 0:
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
    print("\n" + "=" * 72)
    print("  OPTIMALNA ALOKACIJA SERVISA  (x[i][j] = broj izvrsavanja)")
    print("=" * 72)

    header = "         " + "".join(f"  C{computer+1:2d}" for computer in range(COMPUTER_COUNT))
    print(header)
    print("-" * 72)
    for service in range(SERVICE_COUNT):
        row = f"  S{service+1:2d}   " + "".join(f"{chromosome[service][computer]:5d}" for computer in range(COMPUTER_COUNT))
        print(row)

    print("=" * 72)
    print(f"\n  MAKSIMALNA ZARADA: {total_profit:,} dinara\n")

    print("── Iskoriscenje racunara ──────────────────────────────────────────────")
    for computer in range(COMPUTER_COUNT):
        used = sum(TIME[service][computer] * chromosome[service][computer] for service in range(SERVICE_COUNT))
        percent = 100 * used / MAX_TIME
        bar = "#" * int(percent / 5)
        print(f"  C{computer+1:2d}: {used:5d}/{MAX_TIME} min  ({percent:5.1f}%)  {bar}")

    print()
    print("── Iskoriscenje potraznje servisa ─────────────────────────────────────")
    for service in range(SERVICE_COUNT):
        allocated = sum(chromosome[service])
        percent = 100 * allocated / DEMAND[service]
        bar = "#" * int(percent / 5)
        print(f"  S{service+1:2d}: {allocated:5d}/{DEMAND[service]:5d}  ({percent:5.1f}%)  {bar}")

    print()
    print("── Top 10 najisplativijih dodela ──────────────────────────────────────")
    assignments = [
        (PROFIT[service][computer] * chromosome[service][computer], service, computer, chromosome[service][computer], PROFIT[service][computer])
        for service in range(SERVICE_COUNT)
        for computer in range(COMPUTER_COUNT)
        if chromosome[service][computer] > 0
    ]
    assignments.sort(reverse=True)
    print(f"  {'Servis':<8} {'Racunar':<10} {'Kolicina':>10} {'Cena/kom':>10} {'Doprinos':>12}")
    print(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*12}")
    for contribution, service, computer, quantity, price in assignments[:10]:
        print(f"  S{service+1:<7} C{computer+1:<9} {quantity:>10,} {price:>10} {contribution:>12,}")

    print("=" * 72)


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    random.seed(42)
    best_chromosome, best_profit = run_ga()
    print_solution(best_chromosome, best_profit)
