import random

# ── Constants ─────────────────────────────────────────────────────────────────

NUM_SERVICES  = 10
NUM_COMPUTERS = 10
MAX_TIME      = 2880  # 48 hours in minutes

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
POP_SIZE        = 200
GENERATIONS     = 1000
TOURNAMENT_SIZE = 5
CROSSOVER_RATE  = 0.85
MUTATION_RATE   = 0.15
ELITE_COUNT     = 10

# Precomputed helpers
PROFIT_PER_MIN = [
    [PROFIT[i][j] / TIME[i][j] for j in range(NUM_COMPUTERS)]
    for i in range(NUM_SERVICES)
]

MAX_UNITS = [
    [MAX_TIME // TIME[i][j] for j in range(NUM_COMPUTERS)]
    for i in range(NUM_SERVICES)
]

# All (i, j) pairs sorted by profit-per-minute descending (used in greedy init)
SORTED_PAIRS = sorted(
    [(i, j) for i in range(NUM_SERVICES) for j in range(NUM_COMPUTERS)],
    key=lambda p: PROFIT_PER_MIN[p[0]][p[1]],
    reverse=True,
)


# ── Repair ────────────────────────────────────────────────────────────────────

def repair(ch):
    """Ensure ch satisfies all time and demand constraints (in-place + return)."""
    # 1. Clip negatives
    for i in range(NUM_SERVICES):
        for j in range(NUM_COMPUTERS):
            if ch[i][j] < 0:
                ch[i][j] = 0

    # 2. Enforce demand constraints (row-wise)
    for i in range(NUM_SERVICES):
        total = sum(ch[i])
        if total > DEMAND[i]:
            for j in range(NUM_COMPUTERS):
                ch[i][j] = int(ch[i][j] * DEMAND[i] / total)

    # 3. Enforce time constraints (column-wise)
    for j in range(NUM_COMPUTERS):
        used = sum(TIME[i][j] * ch[i][j] for i in range(NUM_SERVICES))
        while used > MAX_TIME:
            worst_i     = -1
            worst_ratio = float('inf')
            for i in range(NUM_SERVICES):
                if ch[i][j] > 0 and PROFIT_PER_MIN[i][j] < worst_ratio:
                    worst_ratio = PROFIT_PER_MIN[i][j]
                    worst_i     = i
            if worst_i == -1:
                break
            ch[worst_i][j] -= 1
            used           -= TIME[worst_i][j]

    # 4. Fill leftover time with the best profit/min services that have remaining demand
    for j in range(NUM_COMPUTERS):
        used = sum(TIME[i][j] * ch[i][j] for i in range(NUM_SERVICES))
        remaining = MAX_TIME - used
        # Services sorted by profit/min on this computer, best first
        order = sorted(range(NUM_SERVICES), key=lambda i: PROFIT_PER_MIN[i][j], reverse=True)
        for i in order:
            if remaining < TIME[i][j]:
                continue
            demand_left = DEMAND[i] - sum(ch[i])
            if demand_left <= 0:
                continue
            add = min(demand_left, remaining // TIME[i][j])
            if add > 0:
                ch[i][j]  += add
                remaining -= add * TIME[i][j]

    return ch


# ── Fitness ───────────────────────────────────────────────────────────────────

def fitness(ch):
    return sum(
        PROFIT[i][j] * ch[i][j]
        for i in range(NUM_SERVICES)
        for j in range(NUM_COMPUTERS)
    )


# ── Initialization ────────────────────────────────────────────────────────────

def init_greedy():
    ch               = [[0] * NUM_COMPUTERS for _ in range(NUM_SERVICES)]
    remaining_demand = DEMAND[:]
    remaining_time   = [MAX_TIME] * NUM_COMPUTERS

    for i, j in SORTED_PAIRS:
        max_by_time   = remaining_time[j] // TIME[i][j]
        max_by_demand = remaining_demand[i]
        val           = min(max_by_time, max_by_demand)
        if val > 0:
            val            = max(0, val - random.randint(0, val // 3 + 1))
            ch[i][j]       = val
            remaining_demand[i] -= val
            remaining_time[j]   -= val * TIME[i][j]

    return repair(ch)


def init_random():
    ch = [
        [random.randint(0, min(DEMAND[i], MAX_UNITS[i][j]))
         for j in range(NUM_COMPUTERS)]
        for i in range(NUM_SERVICES)
    ]
    return repair(ch)


def init_population(size):
    n_greedy   = int(0.3 * size)
    population = [init_greedy() for _ in range(n_greedy)]
    population += [init_random() for _ in range(size - n_greedy)]
    return population


# ── Selection ─────────────────────────────────────────────────────────────────

def tournament_select(population, fitnesses, k=TOURNAMENT_SIZE):
    indices = random.sample(range(len(population)), k)
    best    = max(indices, key=lambda idx: fitnesses[idx])
    return [row[:] for row in population[best]]


# ── Crossover ─────────────────────────────────────────────────────────────────

def uniform_crossover(p1, p2):
    if random.random() > CROSSOVER_RATE:
        return [row[:] for row in p1], [row[:] for row in p2]

    c1 = [[0] * NUM_COMPUTERS for _ in range(NUM_SERVICES)]
    c2 = [[0] * NUM_COMPUTERS for _ in range(NUM_SERVICES)]

    for i in range(NUM_SERVICES):
        for j in range(NUM_COMPUTERS):
            if random.random() < 0.5:
                c1[i][j], c2[i][j] = p1[i][j], p2[i][j]
            else:
                c1[i][j], c2[i][j] = p2[i][j], p1[i][j]

    return repair(c1), repair(c2)


# ── Mutation ──────────────────────────────────────────────────────────────────

def mutate(ch):
    # Operator 1: Point mutation
    per_cell_prob = MUTATION_RATE / NUM_COMPUTERS
    for i in range(NUM_SERVICES):
        for j in range(NUM_COMPUTERS):
            if random.random() < per_cell_prob:
                delta     = max(1, ch[i][j] // 3)
                ch[i][j] += random.randint(-delta, delta)
                ch[i][j]  = max(0, min(ch[i][j], MAX_UNITS[i][j]))

    # Operator 2: Column swap (5% chance)
    if random.random() < 0.05:
        j1, j2 = random.sample(range(NUM_COMPUTERS), 2)
        for i in range(NUM_SERVICES):
            ch[i][j1], ch[i][j2] = ch[i][j2], ch[i][j1]

    # Operator 3: Row redistribution (5% chance)
    if random.random() < 0.05:
        i       = random.randint(0, NUM_SERVICES - 1)
        total   = sum(ch[i])
        if total > 0:
            weights = [PROFIT_PER_MIN[i][j] for j in range(NUM_COMPUTERS)]
            w_sum   = sum(weights)
            ch[i]   = [int(total * w / w_sum) for w in weights]

    return repair(ch)


# ── Local Search (Hill Climbing) ──────────────────────────────────────────────

def local_search(ch, iterations=50):
    """Improve a solution by trying small swaps and reallocations."""
    best_fit = fitness(ch)

    for _ in range(iterations):
        improved = False

        # Move 1: For each computer, try replacing worst-ratio service with best-ratio
        for j in range(NUM_COMPUTERS):
            used = sum(TIME[i][j] * ch[i][j] for i in range(NUM_SERVICES))
            services_here = sorted(
                [i for i in range(NUM_SERVICES) if ch[i][j] > 0],
                key=lambda i: PROFIT_PER_MIN[i][j]
            )
            for i_remove in services_here:
                freed = TIME[i_remove][j]
                profit_lost = PROFIT[i_remove][j]
                # Try adding a unit of a better service
                for i_add in range(NUM_SERVICES):
                    if i_add == i_remove:
                        continue
                    demand_left = DEMAND[i_add] - sum(ch[i_add])
                    if demand_left <= 0:
                        continue
                    # How many units of i_add can we fit in freed time + remaining?
                    remaining = MAX_TIME - used + freed
                    can_add = min(demand_left, remaining // TIME[i_add][j])
                    if can_add <= 0:
                        continue
                    profit_gained = can_add * PROFIT[i_add][j]
                    if profit_gained > profit_lost:
                        ch[i_remove][j] -= 1
                        ch[i_add][j]    += can_add
                        best_fit = best_fit - profit_lost + profit_gained
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break

        # Move 2: Shift units of a service from a worse computer to a better one
        if not improved:
            for i in range(NUM_SERVICES):
                # Sort computers by profit/time ratio for this service
                comps = sorted(range(NUM_COMPUTERS), key=lambda j: PROFIT_PER_MIN[i][j])
                for j_from in comps:
                    if ch[i][j_from] == 0:
                        continue
                    for j_to in reversed(comps):
                        if j_to == j_from:
                            continue
                        if PROFIT_PER_MIN[i][j_to] <= PROFIT_PER_MIN[i][j_from]:
                            break
                        used_to = sum(TIME[s][j_to] * ch[s][j_to] for s in range(NUM_SERVICES))
                        free_to = MAX_TIME - used_to
                        can_move = min(ch[i][j_from], free_to // TIME[i][j_to])
                        if can_move > 0:
                            gain = can_move * (PROFIT[i][j_to] - PROFIT[i][j_from])
                            if gain > 0:
                                ch[i][j_from] -= can_move
                                ch[i][j_to]   += can_move
                                best_fit      += gain
                                improved       = True
                                break
                    if improved:
                        break
                if improved:
                    break

        if not improved:
            break

    # Final fill pass
    repair(ch)
    return ch


# ── Main GA Loop ──────────────────────────────────────────────────────────────

def run_ga():
    population         = init_population(POP_SIZE)
    best_ever          = None
    best_fitness_ever  = 0
    stagnation_counter = 0

    for gen in range(GENERATIONS):
        fitnesses = [fitness(ch) for ch in population]

        gen_best_idx = max(range(len(fitnesses)), key=lambda idx: fitnesses[idx])
        gen_best_fit = fitnesses[gen_best_idx]

        if gen_best_fit > best_fitness_ever:
            best_fitness_ever  = gen_best_fit
            best_ever          = [row[:] for row in population[gen_best_idx]]
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        if gen % 50 == 0:
            avg_fit = sum(fitnesses) / len(fitnesses)
            print(f"Gen {gen:4d} | Best: {gen_best_fit:,} | Avg: {avg_fit:,.1f} | "
                  f"All-time best: {best_fitness_ever:,}")

        # Diversity injection on stagnation
        if stagnation_counter > 0 and stagnation_counter % 100 == 0:
            sorted_idx  = sorted(range(len(fitnesses)), key=lambda idx: fitnesses[idx])
            num_replace = POP_SIZE // 5
            for idx in sorted_idx[:num_replace]:
                population[idx] = init_random()

        # Elitism: keep top ELITE_COUNT, apply local search to the best one
        elite_idx = sorted(range(len(fitnesses)), key=lambda idx: fitnesses[idx], reverse=True)[:ELITE_COUNT]
        new_pop   = [[row[:] for row in population[idx]] for idx in elite_idx]
        # Local search on top 3 elites every 10 generations
        if gen % 10 == 0:
            for e in range(min(3, len(new_pop))):
                new_pop[e] = local_search(new_pop[e])
                f = fitness(new_pop[e])
                if f > best_fitness_ever:
                    best_fitness_ever  = f
                    best_ever          = [row[:] for row in new_pop[e]]
                    stagnation_counter = 0

        # Fill the rest via selection, crossover, mutation
        while len(new_pop) < POP_SIZE:
            p1     = tournament_select(population, fitnesses)
            p2     = tournament_select(population, fitnesses)
            c1, c2 = uniform_crossover(p1, p2)
            c1     = mutate(c1)
            c2     = mutate(c2)
            new_pop.append(c1)
            if len(new_pop) < POP_SIZE:
                new_pop.append(c2)

        population = new_pop

    # Final intensive local search on best solution
    print("\nPrimena lokalne pretrage na najbolje resenje...")
    best_ever = local_search(best_ever, iterations=200)
    best_fitness_ever = fitness(best_ever)
    print(f"Zarada posle lokalne pretrage: {best_fitness_ever:,}")

    return best_ever, best_fitness_ever


# ── Output ────────────────────────────────────────────────────────────────────

def print_solution(ch, total_profit):
    print("\n" + "=" * 72)
    print("  OPTIMALNA ALOKACIJA SERVISA  (x[i][j] = broj izvrsavanja)")
    print("=" * 72)

    header = "         " + "".join(f"  C{j+1:2d}" for j in range(NUM_COMPUTERS))
    print(header)
    print("-" * 72)
    for i in range(NUM_SERVICES):
        row = f"  S{i+1:2d}   " + "".join(f"{ch[i][j]:5d}" for j in range(NUM_COMPUTERS))
        print(row)

    print("=" * 72)
    print(f"\n  MAKSIMALNA ZARADA: {total_profit:,} dinara\n")

    print("── Iskoriscenje racunara ──────────────────────────────────────────────")
    for j in range(NUM_COMPUTERS):
        used = sum(TIME[i][j] * ch[i][j] for i in range(NUM_SERVICES))
        pct  = 100 * used / MAX_TIME
        bar  = "#" * int(pct / 5)
        print(f"  C{j+1:2d}: {used:5d}/{MAX_TIME} min  ({pct:5.1f}%)  {bar}")

    print()
    print("── Iskoriscenje potraznje servisa ─────────────────────────────────────")
    for i in range(NUM_SERVICES):
        alloc = sum(ch[i])
        pct   = 100 * alloc / DEMAND[i]
        bar   = "#" * int(pct / 5)
        print(f"  S{i+1:2d}: {alloc:5d}/{DEMAND[i]:5d}  ({pct:5.1f}%)  {bar}")

    print()
    print("── Top 10 najisplativijih dodela ──────────────────────────────────────")
    assignments = [
        (PROFIT[i][j] * ch[i][j], i, j, ch[i][j], PROFIT[i][j])
        for i in range(NUM_SERVICES)
        for j in range(NUM_COMPUTERS)
        if ch[i][j] > 0
    ]
    assignments.sort(reverse=True)
    print(f"  {'Servis':<8} {'Racunar':<10} {'Kolicina':>10} {'Cena/kom':>10} {'Doprinos':>12}")
    print(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*12}")
    for contrib, i, j, qty, price in assignments[:10]:
        print(f"  S{i+1:<7} C{j+1:<9} {qty:>10,} {price:>10} {contrib:>12,}")

    print("=" * 72)


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    random.seed(42)
    best_chromosome, best_profit = run_ga()
    print_solution(best_chromosome, best_profit)
