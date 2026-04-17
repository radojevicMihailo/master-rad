import os
import time
from datetime import datetime

import pulp

# ── Configuration ────────────────────────────────────────────────────────────

PROBLEM_SIZE = "10x10"
MAX_TIME = 2880
TIME_LIMIT_SECONDS = 1000  # solver time limit

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, f"podaci-{PROBLEM_SIZE}")
RESULTS_DIR = os.path.join(BASE_DIR, f"rezultati-{PROBLEM_SIZE}")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Load Data ────────────────────────────────────────────────────────────────

def load_matrix_tsv(filename):
    """Load matrix from tab-separated file (skip header row and first column)."""
    filepath = os.path.join(DATA_DIR, filename)
    matrix = []
    with open(filepath, "r") as file:
        lines = file.readlines()
        for line in lines[1:]:
            parts = line.strip().split("\t")
            if len(parts) > 1:
                row = [int(value) for value in parts[1:]]
                matrix.append(row)
    return matrix


def load_matrix_csv(filename):
    """Load matrix from CSV file (skip header row and first column)."""
    filepath = os.path.join(DATA_DIR, filename)
    matrix = []
    with open(filepath, "r") as file:
        lines = file.readlines()
        for line in lines[1:]:
            parts = line.strip().split(",")
            if len(parts) > 1:
                row = [int(value) for value in parts[1:]]
                matrix.append(row)
    return matrix


def load_demand_tsv(filename):
    """Load demand values from tab-separated file (skip header, take second column)."""
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


def load_demand_csv(filename):
    """Load demand values from CSV file (second column)."""
    filepath = os.path.join(DATA_DIR, filename)
    demand = []
    with open(filepath, "r") as file:
        for line in file.readlines():
            parts = line.strip().split(",")
            if len(parts) >= 2:
                try:
                    demand.append(int(parts[1]))
                except ValueError:
                    continue
    return demand


def load_data():
    """Load data based on problem size (TSV for 10x10/100x100, CSV for 1000x1000)."""
    if PROBLEM_SIZE == "1000x1000":
        time_matrix = load_matrix_csv("vremena_izvrsavanja_1000.csv")
        profit_matrix = load_matrix_csv("zarada_po_servisu_1000.csv")
        demand = load_demand_csv("zahtevi_za_servisima_1000.csv")
    else:
        time_matrix = load_matrix_tsv("vremena_izvrsavanja.txt")
        profit_matrix = load_matrix_tsv("zarada_po_servisu.txt")
        demand = load_demand_tsv("zahtevi_za_servisima.txt")
    return time_matrix, profit_matrix, demand


# ── Build and Solve ILP ──────────────────────────────────────────────────────

def solve_ilp(time_matrix, profit_matrix, demand):
    """Build and solve ILP model using PuLP."""
    num_services = len(time_matrix)
    num_computers = len(time_matrix[0])

    print(f"Dimenzije problema: {num_services} servisa x {num_computers} racunara")
    print(f"Traznja po servisu: {demand[0]} (uniformna)")
    print(f"Maksimalno vreme po racunaru: {MAX_TIME}")
    print(f"Vremenski limit solvera: {TIME_LIMIT_SECONDS}s")
    print(f"Ukupno promenljivih: {num_services * num_computers}")

    # Create model
    model = pulp.LpProblem("Maksimizacija_zarade", pulp.LpMaximize)

    # Decision variables: x[s][c] = number of units of service s on computer c
    print("\nKreiranje promenljivih...")
    x = {}
    for service in range(num_services):
        for computer in range(num_computers):
            max_units = MAX_TIME // time_matrix[service][computer]
            x[service, computer] = pulp.LpVariable(
                f"x_{service}_{computer}",
                lowBound=0,
                upBound=min(demand[service], max_units),
                cat="Integer",
            )

    # Objective: maximize total profit
    print("Postavljanje funkcije cilja...")
    model += pulp.lpSum(
        profit_matrix[service][computer] * x[service, computer]
        for service in range(num_services)
        for computer in range(num_computers)
    )

    # Constraint 1: demand per service
    print("Dodavanje ogranicenja traznje...")
    for service in range(num_services):
        model += (
            pulp.lpSum(x[service, computer] for computer in range(num_computers)) <= demand[service],
            f"demand_{service}",
        )

    # Constraint 2: time per computer
    print("Dodavanje ogranicenja vremena...")
    for computer in range(num_computers):
        model += (
            pulp.lpSum(
                time_matrix[service][computer] * x[service, computer]
                for service in range(num_services)
            ) <= MAX_TIME,
            f"time_{computer}",
        )

    # Solve
    print("\nResavanje ILP problema...")
    start_time = time.time()

    solver = pulp.HiGHS(msg=1, timeLimit=TIME_LIMIT_SECONDS)
    print("Solver: HiGHS")

    model.solve(solver)
    solve_time = time.time() - start_time

    # Results
    status = pulp.LpStatus[model.status]
    objective_value = int(pulp.value(model.objective)) if model.status == 1 else None

    print(f"\nStatus: {status}")
    print(f"Vreme resavanja: {solve_time:.2f}s")
    if objective_value is not None:
        print(f"Optimalna zarada: {objective_value:,} dinara")

    # Extract solution matrix
    solution = [[0] * num_computers for _ in range(num_services)]
    if model.status == 1:
        for service in range(num_services):
            for computer in range(num_computers):
                value = x[service, computer].varValue
                if value is not None and value > 0:
                    solution[service][computer] = int(value)

    return status, objective_value, solve_time, solution, num_services, num_computers


# ── Output ───────────────────────────────────────────────────────────────────

def print_solution(solution, objective_value, num_services, num_computers, time_matrix, demand):
    """Print solution details."""
    print("\n" + "=" * 80)
    print(f"  OPTIMALNA ZARADA (ILP): {objective_value:,} dinara")
    print("=" * 80)

    # Computer utilization
    print("\n── Iskoriscenje racunara ────────────────────────────────────────────────")
    total_used = 0
    total_capacity = num_computers * MAX_TIME
    for computer in range(num_computers):
        used = sum(time_matrix[service][computer] * solution[service][computer] for service in range(num_services))
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
    for service in range(num_services):
        allocated = sum(solution[service])
        if allocated == demand[service]:
            services_at_max += 1
        elif allocated == 0:
            services_unused += 1
    print(f"  Servisi sa 100% iskoriscenjem: {services_at_max}/{num_services}")
    print(f"  Nekorisceni servisi: {services_unused}/{num_services}")

    print("=" * 80)


def save_results(status, objective_value, solve_time, solution, num_services, num_computers, time_matrix, demand):
    """Save results to file."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"ilp-{PROBLEM_SIZE}-{timestamp}.txt"
    filepath = os.path.join(RESULTS_DIR, filename)

    with open(filepath, "w") as file:
        file.write(f"ILP Solver Rezultati\n")
        file.write(f"{'='*80}\n")
        file.write(f"  Dimenzije: {num_services} servisa x {num_computers} racunara\n")
        file.write(f"  Maksimalno vreme: {MAX_TIME}\n")
        file.write(f"  Vremenski limit: {TIME_LIMIT_SECONDS}s\n")
        file.write(f"  Status: {status}\n")
        file.write(f"  Vreme resavanja: {solve_time:.2f}s\n")

        if objective_value is not None:
            file.write(f"\n  OPTIMALNA ZARADA: {objective_value:,} dinara\n")

            total_used = 0
            total_capacity = num_computers * MAX_TIME
            for computer in range(num_computers):
                used = sum(time_matrix[service][computer] * solution[service][computer] for service in range(num_services))
                total_used += used
            avg_utilization = 100 * total_used / total_capacity
            file.write(f"  Prosecna iskoriscenje: {avg_utilization:.2f}%\n")

            services_at_max = sum(1 for service in range(num_services) if sum(solution[service]) == demand[service])
            file.write(f"  Servisi sa 100% iskoriscenjem: {services_at_max}/{num_services}\n")

        file.write(f"{'='*80}\n")

    print(f"\nRezultati sacuvani u: {filepath}")
    return filepath


# ── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Ucitavanje podataka za {PROBLEM_SIZE}...")
    time_matrix, profit_matrix, demand = load_data()

    status, objective_value, solve_time, solution, num_services, num_computers = solve_ilp(
        time_matrix, profit_matrix, demand
    )

    if objective_value is not None:
        print_solution(solution, objective_value, num_services, num_computers, time_matrix, demand)

    save_results(status, objective_value, solve_time, solution, num_services, num_computers, time_matrix, demand)
