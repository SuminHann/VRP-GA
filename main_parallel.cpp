#pragma comment (lib, "msmpi.lib")
#include <mpi.h>
#include <tbb/tbb.h>
#include <omp.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
#include <chrono>
#include <limits>
#include <cstdlib>

using namespace std;

// Function prototypes
vector<vector<int>> populate(int numLocations, int populationSize);
double calcDistance(pair<int, int> start, pair<int, int> end);
double calcRouteDistance(const vector<pair<int, int>>& locations, const vector<int>& route);
vector<double> calcFitness(const vector<pair<int, int>>& locations, const vector<vector<int>>& routes);
vector<int> selectParent(const vector<vector<int>>& routes, const vector<double>& fitness);
vector<int> crossover(const vector<int>& parent1, const vector<int>& parent2);
void mutate(vector<int>& route, double mutationRate);
void introduceDiversity(vector<vector<int>>& routes, int diversitySize, int numLocations);

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    // MPI setup
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    auto start_time = chrono::high_resolution_clock::now();

    // Configuration
    int numLocations = 20;
    int total_population_size = 10000;
    int generations = 100;
    double mutationRate = 0.2;
    int local_population_size = total_population_size / world_size;

    vector<pair<int, int>> locations = {
        {0, 0}, {1, 3}, {4, 3}, {6, 1}, {3, 0}, {2, 6}, {5, 5}, {8, 8},
        {9, 4}, {7, 2}, {10, 1}, {12, 3}, {13, 7}, {11, 9}, {6, 9},
        {4, 7}, {2, 8}, {0, 5}, {3, 4}, {7, 6}
    };

    // Initialize local population
    vector<vector<int>> local_routes = populate(numLocations, local_population_size);
    vector<int> global_best_route;
    double global_best_fitness = numeric_limits<double>::max();

    for (int gen = 0; gen < generations; ++gen) {
        // 1. Fitness evaluation (OpenMP)
        vector<double> local_fitness = calcFitness(locations, local_routes);

        // 2. Genetic operations (TBB)
        vector<vector<int>> new_routes(local_population_size);
        tbb::parallel_for(tbb::blocked_range<size_t>(0, local_population_size),
            [&](tbb::blocked_range<size_t> range) {
                for (size_t i = range.begin(); i < range.end(); ++i) {
                    vector<int> parent1 = selectParent(local_routes, local_fitness);
                    vector<int> parent2 = selectParent(local_routes, local_fitness);
                    vector<int> child = crossover(parent1, parent2);
                    mutate(child, mutationRate);
                    new_routes[i] = child;
                }
            });

        local_routes = new_routes;

        // 3. Periodic diversity introduction
        if (gen % 10 == 0) {
            if (world_rank == 0) {
                introduceDiversity(local_routes, local_population_size / 4, numLocations);
            }
        }

        // 4. Find local best route
        vector<int> local_best_route = local_routes[0];
        double local_best_fitness = calcRouteDistance(locations, local_best_route);
        for (size_t i = 1; i < local_routes.size(); ++i) {
            double fitness = calcRouteDistance(locations, local_routes[i]);
            if (fitness < local_best_fitness) {
                local_best_fitness = fitness;
                local_best_route = local_routes[i];
            }
        }

        // 5. Exchange best individuals across processes (MPI)
        double global_fitness;
        MPI_Allreduce(&local_best_fitness, &global_fitness, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

        if (local_best_fitness == global_fitness) {
            global_best_route = local_best_route;
        }

        global_best_fitness = global_fitness;

        // Output progress
        if (world_rank == 0 && gen % 10 == 0) {
            cout << "Generation " << gen << ": Best Fitness = " << global_best_fitness << endl;
        }
    }

    // Display the optimal route and fitness
    if (world_rank == 0) {
        cout << "\nOptimal Route: Depot -> ";
        for (int loc : global_best_route) {
            cout << loc << " -> ";
        }
        cout << "Depot\n";
        cout << "Optimal Fitness (Distance): " << global_best_fitness << endl;

        auto end_time = chrono::high_resolution_clock::now();
        double elapsed_time = chrono::duration<double>(end_time - start_time).count();
        cout << "Execution Time: " << elapsed_time << " seconds" << endl;
    }

    MPI_Finalize();
    return 0;
}

// Population initialization
vector<vector<int>> populate(int numLocations, int populationSize) {
    vector<vector<int>> routes(populationSize);
    vector<int> baseRoute(numLocations - 1);
    iota(baseRoute.begin(), baseRoute.end(), 1);

    random_device rd;
    mt19937 gen(rd());

    for (int i = 0; i < populationSize; ++i) {
        shuffle(baseRoute.begin(), baseRoute.end(), gen);
        routes[i] = baseRoute;
    }
    return routes;
}

// Distance calculation
double calcDistance(pair<int, int> start, pair<int, int> end) {
    return sqrt(pow(end.first - start.first, 2) + pow(end.second - start.second, 2));
}

double calcRouteDistance(const vector<pair<int, int>>& locations, const vector<int>& route) {
    double totalDist = calcDistance(locations[0], locations[route[0]]);
    for (size_t i = 1; i < route.size(); ++i) {
        totalDist += calcDistance(locations[route[i - 1]], locations[route[i]]);
    }
    totalDist += calcDistance(locations[route.back()], locations[0]);
    return totalDist;
}

// Fitness calculation (OpenMP)
vector<double> calcFitness(const vector<pair<int, int>>& locations, const vector<vector<int>>& routes) {
    vector<double> fitness(routes.size());
#pragma omp parallel for
    for (size_t i = 0; i < routes.size(); ++i) {
        fitness[i] = calcRouteDistance(locations, routes[i]);
    }
    return fitness;
}

// Parent selection
vector<int> selectParent(const vector<vector<int>>& routes, const vector<double>& fitness) {
    int tournamentSize = 5;
    double bestFitness = numeric_limits<double>::max();
    vector<int> bestRoute;

    for (int i = 0; i < tournamentSize; ++i) {
        int randomIndex = rand() % routes.size();
        if (fitness[randomIndex] < bestFitness) {
            bestFitness = fitness[randomIndex];
            bestRoute = routes[randomIndex];
        }
    }
    return bestRoute;
}

// Crossover
vector<int> crossover(const vector<int>& parent1, const vector<int>& parent2) {
    vector<int> child;
    int split = rand() % parent1.size();
    child.insert(child.end(), parent1.begin(), parent1.begin() + split);
    for (int gene : parent2) {
        if (find(child.begin(), child.end(), gene) == child.end()) {
            child.push_back(gene);
        }
    }
    return child;
}

// Mutation
void mutate(vector<int>& route, double mutationRate) {
    if ((rand() / (double)RAND_MAX) < mutationRate) {
        int idx1 = rand() % route.size();
        int idx2 = rand() % route.size();
        swap(route[idx1], route[idx2]);
    }
}

// Introduce diversity
void introduceDiversity(vector<vector<int>>& routes, int diversitySize, int numLocations) {
    vector<int> baseRoute(numLocations - 1);
    iota(baseRoute.begin(), baseRoute.end(), 1);

    random_device rd;
    mt19937 gen(rd());

    for (int i = 0; i < diversitySize; ++i) {
        shuffle(baseRoute.begin(), baseRoute.end(), gen);
        routes[i] = baseRoute;
    }
}
