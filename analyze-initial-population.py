# Analiza kvaliteta pocetne populacije (slucajno resenje + procedura popravke).
# Reprodukuje repair() iz genetic-algorithm.cpp / random-search.cpp u cistom Python-u
# i meri: kvalitet popravljenog slucajnog resenja u odnosu na ILP optimum,
# iskoriscenost trazje i vremena, i gustinu (udeo nenultih gena) hromozoma.
#
# Pokretanje: python3 analyze-initial-population.py

import glob
import os
import random

MAX_TIME = 2880
NUM_SAMPLES = 3

OPTIMUMS = {
    "10x10": 89847,
    "100x100": 2023188,
    "1000x1000": 20307561,
}


def load_instance(problem_size):
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"podaci-{problem_size}")
    time_path = glob.glob(os.path.join(base_dir, "vremena*"))[0]
    profit_path = glob.glob(os.path.join(base_dir, "zarada*"))[0]
    demand_path = glob.glob(os.path.join(base_dir, "zahtevi*"))[0]
    delimiter = "," if time_path.endswith(".csv") else "\t"

    def load_matrix(path):
        lines = open(path).read().strip().splitlines()[1:]
        return [[int(value) for value in line.split(delimiter)[1:] if value.strip()] for line in lines]

    time_matrix = load_matrix(time_path)
    profit_matrix = load_matrix(profit_path)
    demand = [
        int(line.split(delimiter)[1])
        for line in open(demand_path).read().strip().splitlines()
        if len(line.split(delimiter)) > 1
    ]
    return time_matrix, profit_matrix, demand


def repair(solution, time_matrix, profit_matrix, demand, order_worst_first, order_best_first):
    num_services = len(time_matrix)
    num_computers = len(time_matrix[0])

    # 1) trazenje po redovima: proporcionalno smanjivanje
    for service in range(num_services):
        row_total = sum(solution[service])
        if row_total > demand[service]:
            ratio = demand[service] / row_total
            solution[service] = [int(value * ratio) for value in solution[service]]

    row_sums = [sum(solution[service]) for service in range(num_services)]

    for computer in range(num_computers):
        # 2) vreme po kolonama: uklanjanje najnerentabilnijih jedinica
        used_time = sum(time_matrix[service][computer] * solution[service][computer] for service in range(num_services))
        if used_time > MAX_TIME:
            for service in order_worst_first[computer]:
                if used_time <= MAX_TIME:
                    break
                if solution[service][computer] > 0:
                    excess = used_time - MAX_TIME
                    unit_time = time_matrix[service][computer]
                    units_to_remove = min(solution[service][computer], -(-excess // unit_time))
                    solution[service][computer] -= units_to_remove
                    row_sums[service] -= units_to_remove
                    used_time -= units_to_remove * unit_time

        # 3) popunjavanje preostalog vremena najrentabilnijim jedinicama
        remaining_time = MAX_TIME - used_time
        for service in order_best_first[computer]:
            unit_time = time_matrix[service][computer]
            if remaining_time < unit_time:
                continue
            demand_left = demand[service] - row_sums[service]
            if demand_left <= 0:
                continue
            units_to_add = min(demand_left, remaining_time // unit_time)
            if units_to_add > 0:
                solution[service][computer] += units_to_add
                row_sums[service] += units_to_add
                remaining_time -= units_to_add * unit_time
            if remaining_time < 5:
                break

    return solution


def analyze(problem_size):
    time_matrix, profit_matrix, demand = load_instance(problem_size)
    num_services = len(time_matrix)
    num_computers = len(time_matrix[0])
    optimum = OPTIMUMS[problem_size]

    profit_per_minute = [
        [profit_matrix[service][computer] / time_matrix[service][computer] for computer in range(num_computers)]
        for service in range(num_services)
    ]
    order_worst_first = [
        sorted(range(num_services), key=lambda service: profit_per_minute[service][computer])
        for computer in range(num_computers)
    ]
    order_best_first = [list(reversed(order)) for order in order_worst_first]

    upper_bound = [
        [min(demand[service], MAX_TIME // time_matrix[service][computer]) for computer in range(num_computers)]
        for service in range(num_services)
    ]

    generator = random.Random(42)
    fitness_values = []
    demand_utilizations = []
    time_utilizations = []
    nonzero_fractions = []

    for _ in range(NUM_SAMPLES):
        solution = [
            [generator.randint(0, upper_bound[service][computer]) for computer in range(num_computers)]
            for service in range(num_services)
        ]
        solution = repair(solution, time_matrix, profit_matrix, demand, order_worst_first, order_best_first)

        fitness = sum(
            profit_matrix[service][computer] * solution[service][computer]
            for service in range(num_services)
            for computer in range(num_computers)
        )
        used_time = sum(
            time_matrix[service][computer] * solution[service][computer]
            for service in range(num_services)
            for computer in range(num_computers)
        )
        used_demand = sum(sum(solution[service]) for service in range(num_services))
        nonzero_count = sum(
            1 for service in range(num_services) for computer in range(num_computers) if solution[service][computer] > 0
        )

        fitness_values.append(fitness)
        demand_utilizations.append(used_demand / sum(demand))
        time_utilizations.append(used_time / (num_computers * MAX_TIME))
        nonzero_fractions.append(nonzero_count / (num_services * num_computers))

    average_fitness = sum(fitness_values) / len(fitness_values)
    print(
        f"{problem_size}: popravljeno slucajno resenje ~{average_fitness:.0f} = {100 * average_fitness / optimum:.1f}% optimuma | "
        f"iskoriscena traznja {100 * sum(demand_utilizations) / NUM_SAMPLES:.1f}% | "
        f"iskorisceno vreme {100 * sum(time_utilizations) / NUM_SAMPLES:.1f}% | "
        f"nenulti geni {100 * sum(nonzero_fractions) / NUM_SAMPLES:.1f}%"
    )


if __name__ == "__main__":
    for size in ["10x10", "100x100", "1000x1000"]:
        analyze(size)
