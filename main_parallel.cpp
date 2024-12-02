#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <limits>
#include <cstdlib>
#include <chrono>
#include <omp.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <mpi.h>
#include <numeric>

#define __cdecl
#define __stdcall

using namespace std;

// Function prototypes
vector<pair<int, int>> initializeLocations(int numLocations);
vector<double> calcFitness(vector<pair<int, int>>& locations, vector<vector<int>>& routes);
vector<vector<int>> populate(int numLocations, int populationSize);
double calcRouteDistance(const vector<pair<int, int>>& locations, const vector<int>& route);
vector<int> selectParent(const vector<vector<int>>& routes, const vector<double>& fitness);
vector<int> crossover(const vector<int>& parent1, const vector<int>& parent2);
void mutate(vector<int>& route, double mutationRate);

// Function definitions
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Configuration
    int numLocations = 20;
    int total_population_size = 100000;
    int generations = 100;
    double mutationRate = 0.2;
    int local_population_size = total_population_size / world_size;

    // Timer
    auto start_time = chrono::high_resolution_clock::now();

    // Initialize locations
    vector<pair<int, int>> locations = {
        {0, 0}, {1, 3}, {4, 3}, {6, 1}, {3, 0}, {2, 6}, {5, 5}, {8, 8},
        {9, 4}, {7, 2}, {10, 1}, {12, 3}, {13, 7}, {11, 9}, {6, 9},
        {4, 7}, {2, 8}, {0, 5}, {3, 4}, {7, 6}
    };

    // Initialize routes for the local population
    vector<vector<int>> local_routes = populate(numLocations, local_population_size);

    // Global best route tracking
    vector<int> global_best_route;
    double global_best_fitness = numeric_limits<double>::max();

    // Main generation loop
    for (int gen = 0; gen < generations; ++gen) {
        // Each rank calculates the fitness of its local population
        vector<double> local_fitness = calcFitness(locations, local_routes);
        

        // Step 2: Perform genetic operations using TBB with dynamic task granularity
        vector<vector<int>> new_routes(local_population_size);
        int chunk_size = max(100, local_population_size / world_size); // Adjust chunk size dynamically
        tbb::parallel_for(tbb::blocked_range<size_t>(0, local_population_size, chunk_size),
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

        // Step 3: Synchronize results using MPI with diverse routes
        if (gen % 10 == 0) {
            vector<double> local_best_fitnesses(world_size);
            vector<int> global_routes(local_population_size * world_size);

            // Share diverse routes among processes
            MPI_Reduce(local_routes[0].data(), global_routes.data(), numLocations, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);

            // Update global best
            double local_best_fitness = *min_element(local_fitness.begin(), local_fitness.end());
            MPI_Allreduce(&local_best_fitness, &global_best_fitness, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

            // Find the global best route from all gathered routes
            if (local_best_fitness == global_best_fitness) {
                auto best_route_index = min_element(local_fitness.begin(), local_fitness.end()) - local_fitness.begin();
                global_best_route = local_routes[best_route_index];
            }

            if (world_rank == 0) {
                cout << "Generation " << gen << ": Best Fitness = " << global_best_fitness << endl;
            }
        }
    }

    // Output results
    if (world_rank == 0) {
        cout << "\nOptimal Route: Depot -> ";
        for (int loc : global_best_route) {
            cout << loc << " -> ";
        }
        cout << "Depot" << endl;

        cout << "Optimal Fitness (Distance): " << global_best_fitness << endl;

        auto end_time = chrono::high_resolution_clock::now();
        double execution_time = chrono::duration<double>(end_time - start_time).count();
        cout << "Execution Time: " << execution_time << " seconds" << endl;
    }

    MPI_Finalize();
    return 0;
}

// Function to initialize random locations
vector<pair<int, int>> initializeLocations(int numLocations) {
    vector<pair<int, int>> locations;
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, 10);

    locations.push_back({ 0, 0 }); // Depot
    for (int i = 0; i < numLocations; ++i) {
        locations.push_back({ dis(gen), dis(gen) });
    }
    return locations;
}

vector<double> calcFitness(vector<pair<int, int>>& locations, vector<vector<int>>& routes) {
    vector<double> fitness;
#pragma omp parallel for
    for (int i = 0; i < routes.size(); i++) {
        double d = calcRouteDistance(locations, routes[i]);
        fitness.push_back(d);
    }
    return fitness;
}

// Function to populate the initial routes
vector<vector<int>> populate(int numLocations, int populationSize) {
    vector<vector<int>> routes;
    vector<int> route(numLocations - 1);
    iota(route.begin(), route.end(), 1); // Fill with 1 to numLocations - 1

    random_device rd;
    mt19937 gen(rd());

#pragma omp parallel for
    for (int i = 0; i < populationSize; ++i) {
        shuffle(route.begin(), route.end(), gen);
        routes.push_back(route);
    }
    return routes;
}

// Function to calculate route distance
double calcRouteDistance(const vector<pair<int, int>>& locations, const vector<int>& route) {
    double distance = 0;
    int prev = 0; // Depot
#pragma omp parallel for
    for (int loc : route) {
        distance += sqrt(pow(locations[loc].first - locations[prev].first, 2) +
            pow(locations[loc].second - locations[prev].second, 2));
        prev = loc;
    }
    // Return to depot
    distance += sqrt(pow(locations[0].first - locations[prev].first, 2) +
        pow(locations[0].second - locations[prev].second, 2));
    return distance;
}

// Parent selection using tournament selection
vector<int> selectParent(const vector<vector<int>>& routes, const vector<double>& fitness) {
    int tournamentSize = 5;
    vector<int> best_route;
    double best_fitness = numeric_limits<double>::max();
#pragma omp parallel for
    for (int i = 0; i < tournamentSize; ++i) {
        int idx = rand() % routes.size();
        if (fitness[idx] < best_fitness) {
            best_fitness = fitness[idx];
            best_route = routes[idx];
        }
    }
    return best_route;
}

// Crossover operator
vector<int> crossover(const vector<int>& parent1, const vector<int>& parent2) {
    vector<int> child(parent1.size(), -1);
    int start = rand() % parent1.size();
    int end = rand() % parent1.size();
    if (start > end) swap(start, end);
#pragma omp parallel for
    for (int i = start; i <= end; ++i) {
        child[i] = parent1[i];
    }

    int child_index = 0;
    for (int gene : parent2) {
        if (find(child.begin(), child.end(), gene) == child.end()) {
            while (child[child_index] != -1) ++child_index;
            child[child_index] = gene;
        }
    }
    return child;
}

// Mutation operator
void mutate(vector<int>& route, double mutationRate) {
    if ((rand() / (double)RAND_MAX) < mutationRate) {
        int idx1 = rand() % route.size();
        int idx2 = rand() % route.size();
        swap(route[idx1], route[idx2]);
    }
}
