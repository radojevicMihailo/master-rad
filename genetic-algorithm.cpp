// Genetic Algorithm — C++ port of genetic-algorithm.py
// Build: g++ -O2 -std=c++17 genetic-algorithm.cpp -o ga
// Run:   ./ga

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// ── Hyperparameters ──────────────────────────────────────────────────────────
static const std::string PROBLEM_SIZE = "1000x1000";
static const int POPULATION_SIZE = 10000;
static const int GENERATIONS = 100;
static const int TOURNAMENT_SIZE = 10;
static const double CROSSOVER_RATE = 0.8;
static const double MUTATION_RATE = 0.15;
static const int MUTATION_GENES = 1000; // cells changed per mutation event (scale up for big problems)
static const int ELITE_COUNT = 100;
static const int NUM_RUNS = 3;
static const int MAX_TIME = 2880;

// ── Globals (problem data) ───────────────────────────────────────────────────
static int NUM_SERVICES = 0;
static int NUM_COMPUTERS = 0;
static std::vector<std::vector<int>> TIME_MAT;
static std::vector<std::vector<int>> PROFIT_MAT;
static std::vector<int> DEMAND;
static std::vector<std::vector<int>> MAX_UNITS;
static std::vector<std::vector<double>> PROFIT_PER_MINUTE;
static std::vector<std::vector<int>> SERVICES_BY_RATIO_PER_COMPUTER;   // worst first
static std::vector<std::vector<int>> SERVICES_BEST_FIRST_PER_COMPUTER; // best first

// Chromosome: flat row-major [service * NUM_COMPUTERS + computer]
// int16_t: cell values bounded by min(DEMAND, MAX_UNITS) << 32767 for current datasets.
using Gene = int16_t;
using Chromosome = std::vector<Gene>;

inline Gene &cell(Chromosome &c, int s, int k) { return c[s * NUM_COMPUTERS + k]; }
inline Gene cell(const Chromosome &c, int s, int k) { return c[s * NUM_COMPUTERS + k]; }

// ── RNG ──────────────────────────────────────────────────────────────────────
static std::mt19937 rng(std::random_device{}());
inline int randint(int low, int high) { return std::uniform_int_distribution<int>(low, high)(rng); }
inline double randreal() { return std::uniform_real_distribution<double>(0.0, 1.0)(rng); }

// ── Data loading ─────────────────────────────────────────────────────────────
static std::vector<std::vector<int>> load_matrix(const std::string &path, char delim, bool skip_header)
{
    std::vector<std::vector<int>> matrix;
    std::ifstream file(path);
    if (!file)
    {
        std::cerr << "Cannot open " << path << "\n";
        std::exit(1);
    }
    std::string line;
    if (skip_header)
        std::getline(file, line);
    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string cell_str;
        std::vector<int> row;
        bool first = true;
        while (std::getline(ss, cell_str, delim))
        {
            if (first)
            {
                first = false;
                continue;
            } // skip first col label
            try
            {
                row.push_back(std::stoi(cell_str));
            }
            catch (...)
            {
            }
        }
        if (!row.empty())
            matrix.push_back(row);
    }
    return matrix;
}

static std::vector<int> load_demand(const std::string &path, char delim, bool skip_header)
{
    std::vector<int> demand;
    std::ifstream file(path);
    if (!file)
    {
        std::cerr << "Cannot open " << path << "\n";
        std::exit(1);
    }
    std::string line;
    if (skip_header)
        std::getline(file, line);
    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string label, value_str;
        if (std::getline(ss, label, delim) && std::getline(ss, value_str, delim))
        {
            try
            {
                demand.push_back(std::stoi(value_str));
            }
            catch (...)
            {
            }
        }
    }
    return demand;
}

static void load_data()
{
    fs::path base = fs::path(__FILE__).parent_path();
    fs::path dir = base / ("podaci-" + PROBLEM_SIZE);

    std::string size_number = PROBLEM_SIZE.substr(0, PROBLEM_SIZE.find('x'));
    fs::path csv_time = dir / ("vremena_izvrsavanja_" + size_number + ".csv");
    fs::path csv_profit = dir / ("zarada_po_servisu_" + size_number + ".csv");
    fs::path csv_demand = dir / ("zahtevi_za_servisima_" + size_number + ".csv");

    if (fs::exists(csv_time))
    {
        TIME_MAT = load_matrix(csv_time.string(), ',', true);
        PROFIT_MAT = load_matrix(csv_profit.string(), ',', true);
        DEMAND = load_demand(csv_demand.string(), ',', false);
    }
    else
    {
        TIME_MAT = load_matrix((dir / "vremena_izvrsavanja.txt").string(), '\t', true);
        PROFIT_MAT = load_matrix((dir / "zarada_po_servisu.txt").string(), '\t', true);
        DEMAND = load_demand((dir / "zahtevi_za_servisima.txt").string(), '\t', false);
    }

    NUM_SERVICES = (int)TIME_MAT.size();
    NUM_COMPUTERS = (int)TIME_MAT[0].size();

    MAX_UNITS.assign(NUM_SERVICES, std::vector<int>(NUM_COMPUTERS));
    PROFIT_PER_MINUTE.assign(NUM_SERVICES, std::vector<double>(NUM_COMPUTERS));
    for (int s = 0; s < NUM_SERVICES; ++s)
        for (int k = 0; k < NUM_COMPUTERS; ++k)
        {
            MAX_UNITS[s][k] = MAX_TIME / TIME_MAT[s][k];
            PROFIT_PER_MINUTE[s][k] = (double)PROFIT_MAT[s][k] / (double)TIME_MAT[s][k];
        }

    SERVICES_BY_RATIO_PER_COMPUTER.assign(NUM_COMPUTERS, {});
    SERVICES_BEST_FIRST_PER_COMPUTER.assign(NUM_COMPUTERS, {});
    for (int k = 0; k < NUM_COMPUTERS; ++k)
    {
        std::vector<int> idx(NUM_SERVICES);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [k](int a, int b)
                  { return PROFIT_PER_MINUTE[a][k] < PROFIT_PER_MINUTE[b][k]; });
        SERVICES_BY_RATIO_PER_COMPUTER[k] = idx;
        std::reverse(idx.begin(), idx.end());
        SERVICES_BEST_FIRST_PER_COMPUTER[k] = idx;
    }
}

// ── Repair ───────────────────────────────────────────────────────────────────
static void repair(Chromosome &chromosome)
{
    // 1. Clip negatives
    for (auto &gene : chromosome)
        if (gene < 0)
            gene = 0;

    // 2. Demand constraints (row-wise)
    for (int s = 0; s < NUM_SERVICES; ++s)
    {
        long long total = 0;
        for (int k = 0; k < NUM_COMPUTERS; ++k)
            total += cell(chromosome, s, k);
        if (total > DEMAND[s])
        {
            double ratio = (double)DEMAND[s] / (double)total;
            for (int k = 0; k < NUM_COMPUTERS; ++k)
                cell(chromosome, s, k) = (Gene)(cell(chromosome, s, k) * ratio);
        }
    }

    // 3. Time constraints (column-wise)
    for (int k = 0; k < NUM_COMPUTERS; ++k)
    {
        long long used = 0;
        for (int s = 0; s < NUM_SERVICES; ++s)
            used += (long long)TIME_MAT[s][k] * cell(chromosome, s, k);
        if (used > MAX_TIME)
        {
            for (int s : SERVICES_BY_RATIO_PER_COMPUTER[k])
            {
                if (used <= MAX_TIME)
                    break;
                if (cell(chromosome, s, k) > 0)
                {
                    long long excess = used - MAX_TIME;
                    int t = TIME_MAT[s][k];
                    int units_to_remove = (int)std::min<long long>(cell(chromosome, s, k), (excess + t - 1) / t);
                    cell(chromosome, s, k) = (Gene)(cell(chromosome, s, k) - units_to_remove);
                    used -= (long long)units_to_remove * t;
                }
            }
        }
    }

    // 4. Fill leftover time with best profit/minute services
    for (int k = 0; k < NUM_COMPUTERS; ++k)
    {
        long long used = 0;
        for (int s = 0; s < NUM_SERVICES; ++s)
            used += (long long)TIME_MAT[s][k] * cell(chromosome, s, k);
        long long remaining = MAX_TIME - used;
        for (int s : SERVICES_BEST_FIRST_PER_COMPUTER[k])
        {
            int t = TIME_MAT[s][k];
            if (remaining < t)
                continue;
            long long row_sum = 0;
            for (int kk = 0; kk < NUM_COMPUTERS; ++kk)
                row_sum += cell(chromosome, s, kk);
            long long demand_left = DEMAND[s] - row_sum;
            if (demand_left <= 0)
                continue;
            int add = (int)std::min<long long>(demand_left, remaining / t);
            if (add > 0)
            {
                cell(chromosome, s, k) = (Gene)(cell(chromosome, s, k) + add);
                remaining -= (long long)add * t;
            }
            if (remaining < 5)
                break;
        }
    }
}

// ── Fitness ──────────────────────────────────────────────────────────────────
static long long fitness(const Chromosome &chromosome)
{
    long long total = 0;
    for (int s = 0; s < NUM_SERVICES; ++s)
        for (int k = 0; k < NUM_COMPUTERS; ++k)
            total += (long long)PROFIT_MAT[s][k] * cell(chromosome, s, k);
    return total;
}

// ── Init ─────────────────────────────────────────────────────────────────────
static Chromosome init_random()
{
    Chromosome chromosome(NUM_SERVICES * NUM_COMPUTERS, 0);
    for (int s = 0; s < NUM_SERVICES; ++s)
    {
        for (int k = 0; k < NUM_COMPUTERS; ++k)
        {
            int upper = std::min(DEMAND[s], MAX_UNITS[s][k]);
            cell(chromosome, s, k) = (Gene)randint(0, upper);
        }
    }
    repair(chromosome);
    return chromosome;
}

// ── Selection ────────────────────────────────────────────────────────────────
static int tournament_select_index(const std::vector<long long> &fitnesses)
{
    int best_index = randint(0, (int)fitnesses.size() - 1);
    for (int i = 1; i < TOURNAMENT_SIZE; ++i)
    {
        int candidate = randint(0, (int)fitnesses.size() - 1);
        if (fitnesses[candidate] > fitnesses[best_index])
            best_index = candidate;
    }
    return best_index;
}

// ── Crossover ────────────────────────────────────────────────────────────────
static void one_point_crossover(const Chromosome &parent1, const Chromosome &parent2,
                                Chromosome &child1, Chromosome &child2)
{
    child1 = parent1;
    child2 = parent2;
    if (randreal() > CROSSOVER_RATE)
        return;
    int total_genes = NUM_SERVICES * NUM_COMPUTERS;
    int crossover_point = randint(1, total_genes - 1);
    for (int i = crossover_point; i < total_genes; ++i)
    {
        child1[i] = parent2[i];
        child2[i] = parent1[i];
    }
    repair(child1);
    repair(child2);
}

// ── Mutation ─────────────────────────────────────────────────────────────────
static void mutate(Chromosome &chromosome)
{
    if (randreal() < MUTATION_RATE)
    {
        for (int m = 0; m < MUTATION_GENES; ++m)
        {
            int s = randint(0, NUM_SERVICES - 1);
            int k = randint(0, NUM_COMPUTERS - 1);
            cell(chromosome, s, k) = (Gene)randint(0, MAX_UNITS[s][k]);
        }
        repair(chromosome);
    }
}

// ── GA Loop ──────────────────────────────────────────────────────────────────
struct RunResult
{
    Chromosome best;
    long long best_fitness;
    std::vector<long long> history;
    double elapsed_seconds;
};

static RunResult run_ga()
{
    std::cout << "\nInicijalizacija populacije...\n";
    std::vector<Chromosome> population(POPULATION_SIZE);
    for (auto &chromosome : population)
        chromosome = init_random();

    Chromosome best_ever;
    long long best_fitness_ever = 0;
    int stagnation_counter = 0;
    std::vector<long long> generation_best_history;

    auto start_time = std::chrono::steady_clock::now();
    std::cout << "Pokretanje evolucije...\n\n";

    for (int generation = 0; generation < GENERATIONS; ++generation)
    {
        std::vector<long long> fitnesses(POPULATION_SIZE);
        for (int i = 0; i < POPULATION_SIZE; ++i)
            fitnesses[i] = fitness(population[i]);

        int generation_best_index = (int)(std::max_element(fitnesses.begin(), fitnesses.end()) - fitnesses.begin());
        long long generation_best_fitness = fitnesses[generation_best_index];
        generation_best_history.push_back(generation_best_fitness);

        if (generation_best_fitness > best_fitness_ever)
        {
            best_fitness_ever = generation_best_fitness;
            best_ever = population[generation_best_index];
            stagnation_counter = 0;
        }
        else
        {
            stagnation_counter++;
        }

        if (generation % 10 == 0)
        {
            auto now = std::time(nullptr);
            char timebuf[16];
            std::strftime(timebuf, sizeof(timebuf), "%H:%M:%S", std::localtime(&now));
            std::cout << "[" << timebuf << "] Gen " << std::setw(4) << generation
                      << " | Best: " << generation_best_fitness << "\n";
        }

        // Diversity injection
        if (stagnation_counter > 0 && stagnation_counter % 100 == 0)
        {
            std::vector<int> sorted_indices(POPULATION_SIZE);
            std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
            std::sort(sorted_indices.begin(), sorted_indices.end(),
                      [&](int a, int b)
                      { return fitnesses[a] < fitnesses[b]; });
            int num_replace = POPULATION_SIZE / 5;
            for (int i = 0; i < num_replace; ++i)
                population[sorted_indices[i]] = init_random();
        }

        // Elitism
        std::vector<int> elite_indices(POPULATION_SIZE);
        std::iota(elite_indices.begin(), elite_indices.end(), 0);
        std::partial_sort(elite_indices.begin(), elite_indices.begin() + ELITE_COUNT, elite_indices.end(),
                          [&](int a, int b)
                          { return fitnesses[a] > fitnesses[b]; });

        std::vector<Chromosome> new_population;
        new_population.reserve(POPULATION_SIZE);
        for (int i = 0; i < ELITE_COUNT; ++i)
            new_population.push_back(population[elite_indices[i]]);

        while ((int)new_population.size() < POPULATION_SIZE)
        {
            int parent1_index = tournament_select_index(fitnesses);
            int parent2_index = tournament_select_index(fitnesses);
            Chromosome child1, child2;
            one_point_crossover(population[parent1_index], population[parent2_index], child1, child2);
            mutate(child1);
            mutate(child2);
            new_population.push_back(std::move(child1));
            if ((int)new_population.size() < POPULATION_SIZE)
                new_population.push_back(std::move(child2));
        }

        population = std::move(new_population);
    }

    auto end_time = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(end_time - start_time).count();

    return {std::move(best_ever), best_fitness_ever, std::move(generation_best_history), elapsed};
}

// ── Results writing ──────────────────────────────────────────────────────────
static std::string timestamp_str()
{
    auto now = std::time(nullptr);
    char buffer[32];
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%d_%H-%M-%S", std::localtime(&now));
    return buffer;
}

static void save_results(const std::vector<RunResult> &runs)
{
    fs::path base = fs::path(__FILE__).parent_path();
    fs::path results_dir = base / ("rezultati-" + PROBLEM_SIZE);
    fs::path txt_dir = results_dir / "rezultati";
    fs::create_directories(txt_dir);

    std::ostringstream name;
    name << "ga-cpp-" << POPULATION_SIZE << "-" << GENERATIONS << "-"
         << ELITE_COUNT << ".txt";
    fs::path filepath = txt_dir / name.str();

    std::ofstream file(filepath);
    file << "GA Hiperparametri (C++):\n"
         << "  Velicina populacije: " << POPULATION_SIZE << "\n"
         << "  Broj generacija: " << GENERATIONS << "\n"
         << "  Velicina turnira: " << TOURNAMENT_SIZE << "\n"
         << "  Verovatnoca ukrstanja: " << CROSSOVER_RATE << "\n"
         << "  Verovatnoca mutacije: " << MUTATION_RATE << "\n"
         << "  Elitizam: " << ELITE_COUNT << "\n"
         << "  Broj pokretanja: " << NUM_RUNS << "\n\n";

    long long best_overall = 0, worst_overall = LLONG_MAX;
    double total_time = 0;
    std::vector<long long> profits;
    for (size_t i = 0; i < runs.size(); ++i)
    {
        file << "  Pokretanje " << (i + 1) << ": " << runs[i].best_fitness
             << " dinara  (" << std::fixed << std::setprecision(2) << runs[i].elapsed_seconds << "s)\n";
        profits.push_back(runs[i].best_fitness);
        best_overall = std::max(best_overall, runs[i].best_fitness);
        worst_overall = std::min(worst_overall, runs[i].best_fitness);
        total_time += runs[i].elapsed_seconds;
    }
    std::sort(profits.begin(), profits.end());
    double median = profits.size() % 2 ? (double)profits[profits.size() / 2]
                                       : (profits[profits.size() / 2 - 1] + profits[profits.size() / 2]) / 2.0;
    double mean = 0.0;
    for (long long p : profits)
        mean += (double)p;
    mean /= (double)profits.size();
    double variance = 0.0;
    for (long long p : profits)
        variance += ((double)p - mean) * ((double)p - mean);
    variance /= (double)profits.size();
    double stdev = std::sqrt(variance);

    file << "\n  PROSEK:   " << std::fixed << std::setprecision(1) << mean << "\n"
         << "  MEDIJANA: " << median << "\n"
         << "  NAJBOLJI: " << best_overall << "\n"
         << "  NAJGORI:  " << worst_overall << "\n"
         << "  STDEV:    " << stdev << "\n"
         << "  UKUPNO VREME: " << total_time << "s\n"
         << "  PROSECNO VREME: " << total_time / runs.size() << "s\n\n";

    file << "NAJBOLJI PO GENERACIJAMA\n";
    for (size_t i = 0; i < runs.size(); ++i)
    {
        file << "\n  Pokretanje " << (i + 1) << ":\n";
        for (size_t g = 0; g < runs[i].history.size(); ++g)
            file << "    Gen " << std::setw(4) << g << ": " << runs[i].history[g] << "\n";
    }

    std::cout << "\nRezultati sacuvani u: " << filepath << "\n";

    // Convergence CSV (for plotting later with python/matplotlib or gnuplot)
    std::ostringstream csv_name;
    csv_name << "ga-cpp-convergence-" << POPULATION_SIZE << "-" << GENERATIONS << "-"
             << ELITE_COUNT << ".csv";
    fs::path csv_dir = results_dir / "svi-podaci";
    fs::create_directories(csv_dir);
    fs::path csv_path = csv_dir / csv_name.str();
    std::ofstream csv(csv_path);
    csv << "run,generation,evaluations,best_fitness\n";
    for (size_t i = 0; i < runs.size(); ++i)
        for (size_t g = 0; g < runs[i].history.size(); ++g)
            csv << (i + 1) << "," << g << "," << ((g + 1) * POPULATION_SIZE) << "," << runs[i].history[g] << "\n";
    std::cout << "Konvergencija (CSV): " << csv_path << "\n";
}

// ── Entry Point ──────────────────────────────────────────────────────────────
int main()
{
    std::cout << "Ucitavanje podataka za " << PROBLEM_SIZE << "...\n";
    load_data();
    std::cout << "Dimenzije problema: " << NUM_SERVICES << " servisa x "
              << NUM_COMPUTERS << " racunara\n";
    std::cout << "Traznja po servisu: " << DEMAND[0] << " (uniformna)\n";

    std::cout << "\nGA Hiperparametri:\n"
              << "  Velicina populacije: " << POPULATION_SIZE << "\n"
              << "  Broj generacija: " << GENERATIONS << "\n"
              << "  Velicina turnira: " << TOURNAMENT_SIZE << "\n"
              << "  Verovatnoca ukrstanja: " << CROSSOVER_RATE << "\n"
              << "  Verovatnoca mutacije: " << MUTATION_RATE << "\n"
              << "  Elitizam: " << ELITE_COUNT << "\n";

    std::vector<RunResult> runs;
    for (int r = 0; r < NUM_RUNS; ++r)
    {
        std::cout << "\n===== Pokretanje " << (r + 1) << "/" << NUM_RUNS << " =====\n";
        runs.push_back(run_ga());
        std::cout << "\nNajbolja zarada: " << runs.back().best_fitness
                  << " | Vreme: " << std::fixed << std::setprecision(2) << runs.back().elapsed_seconds << "s\n";
    }

    save_results(runs);
    return 0;
}
