#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <limits>
#include <cstdlib>
#include <chrono>

using namespace std;

vector< pair <int, int> > initializeLocations (int numLocations);
vector<vector<int> > populate (int populationSize, int routeSize);
double calcDistance (pair<int, int> start, pair<int, int> end);
double calcRouteDistance (vector<pair<int, int> >& locations, vector<int>& route);
vector<double> calcFitness (vector<pair<int, int> >& locations, vector<vector<int> >& routes);
vector<int> selectParent (vector<vector<int> >& routes, vector<double>& fitness);
vector<int> crossover (vector<int>& parent1, vector<int>& parent2);
void mutate(vector<int>& route, double mutationRate);
void replaceWorstRouteWithChild(vector<vector<int>>& routes, vector<double>& fitness, vector<int>& child, vector<pair<int, int>>& locations);

int main () {

    auto start = std::chrono::high_resolution_clock::now();


    int numLocations = 20;
    int routesSize = 100000;
    int generations = 100;
    double mutationRate = 0.2;

    srand(time(0));

    // vector<pair<int, int> > locations = initializeLocations(numLocations);
vector<pair<int, int> > locations = {
    {0, 0}, {1, 3}, {4, 3}, {6, 1}, {3, 0}, {2, 6}, {5, 5}, {8, 8},
    {9, 4}, {7, 2}, {10, 1}, {12, 3}, {13, 7}, {11, 9}, {6, 9},
    {4, 7}, {2, 8}, {0, 5}, {3, 4}, {7, 6}
};
    vector<vector<int> > routes = populate(numLocations, routesSize);



    // for (int i = 0; i < locations.size(); i++) {
    //     cout << "{" << locations[i].first << ", " << locations[i].second << "}" << endl;
    // }

    // for (int i = 0; i < routes.size(); i++) {
    //     cout << "routes " << i << ": ";
    //     for (int j = 0; j < routes[i].size(); j++) {
    //         cout << routes[i][j];
    //     }
    //     cout << endl;
    // }

    // cout << "fitness " << ": ";
    // for (int i = 0; i < fitness.size(); i++) {
    //     cout << fitness[i] << " ";
    // }
    // cout << endl;

    vector<double> fitness = calcFitness(locations, routes);

    for (int generation = 0; generation < generations; generation++) {

        vector<int> selectedParent1 = selectParent(routes, fitness);
        vector<int> selectedParent2 = selectParent(routes, fitness);
        
        vector<int> child = crossover(selectedParent1, selectedParent2);
        
        mutate(child, mutationRate);

        replaceWorstRouteWithChild(routes, fitness, child, locations);

        double bestFitness = *min_element(fitness.begin(), fitness.end());
        cout << "Generation " << generation + 1 << ": Best Fitness = " << bestFitness << endl;

    }
    vector<double> finalFitness = calcFitness(locations, routes);
    int bestIndex = min_element(finalFitness.begin(), finalFitness.end()) - finalFitness.begin();
    vector<int> bestRoute = routes[bestIndex];

    cout << "\nOptimal Route: Depot -> ";
    for (int loc : bestRoute) {
        cout << loc << " -> ";
    }
    cout << "Depot" << endl;

    cout << "Optimal Distance: " << finalFitness[bestIndex] << endl;

    auto end = std::chrono::high_resolution_clock::now();
    double time = std::chrono::duration<double>(end - start).count();
    cout << "Time taken: " << time << endl;

    return 0;
}

vector< pair <int, int> > initializeLocations (int numLocations) {
    vector<pair<int, int> > locations;
    locations.push_back(make_pair(0, 0));

    for (int i = 0; i < numLocations; i++) {
        locations.push_back(make_pair(rand() % 10, rand() % 10));
    }

    return locations;
}

vector<vector<int>> populate(int numLocations, int routesSize) {
    vector<vector<int>> routes;
    vector<int> route;

    for (int i = 0; i < numLocations; i++) {
        route.push_back(i + 1);
    }

    random_device rd;
    mt19937 gen(rd()); 

    for (int i = 0; i < routesSize; i++) {
        shuffle(route.begin(), route.end(), gen); 
        routes.push_back(route);
    }

    return routes;
}


double calcDistance (pair<int, int> start, pair<int, int> end) {
    return sqrt(pow((end.first - start.first), 2) + pow((end.second - start.second), 2));
}

double calcRouteDistance (vector<pair<int, int> >& locations, vector<int>&  route) {
    double totalDist = 0;

    totalDist += calcDistance(locations[0], locations[route[0]]);

    for (int i = 1; i < route.size(); i++) {
        totalDist += calcDistance(locations[route[i - 1]], locations[route[i]]);
    }

    totalDist += calcDistance(locations[route.back()], locations[0]);

    return totalDist;
}

vector<double> calcFitness (vector<pair<int, int> >& locations, vector<vector<int> >& routes) {
    vector<double> fitness;

    for (int i = 0; i < routes.size(); i++) {
        double d = calcRouteDistance(locations, routes[i]);
        fitness.push_back(d);
    }

    return fitness;
}

vector<int> selectParent(vector<vector<int>>& routes, vector<double>& fitness) {
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


vector<int> crossover (vector<int>& parent1, vector<int>& parent2) {
    vector<int> child;
    int randomPoint = rand() % parent1.size();

    for (int i = 0; i < randomPoint; i++) {
        child.push_back(parent1[i]);
    }

    for (int i = 0; i < parent2.size(); i++) {
        if (find(child.begin(), child.end(), parent2[i]) == child.end()) {
            child.push_back(parent2[i]);
        }
    }

    return child;
}

void mutate(vector<int>& route, double mutationRate) {
    if ((rand() / (double)RAND_MAX) < mutationRate) {
        int idx1 = rand() % route.size();
        int idx2 = rand() % route.size();
        swap(route[idx1], route[idx2]);
    }
}


void replaceWorstRouteWithChild(vector<vector<int>>& routes, vector<double>& fitness, vector<int>& child, vector<pair<int, int>>& locations) {
    double maxFitness = -1;
    int worstIndex = 0;

    for (int i = 0; i < fitness.size(); i++) {
        if (fitness[i] > maxFitness) {
            maxFitness = fitness[i];
            worstIndex = i;
        }
    }

    routes[worstIndex] = child;
    fitness[worstIndex] = calcRouteDistance(locations, child);
}


