import random
import numpy as np
import math
from multiprocessing import Pool

# Generate random cities
def generate_cities(num_cities):
    return [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(num_cities)]

# Euclidean distance between two cities
def distance(city1, city2):
    return math.sqrt((city2[0] - city1[0]) ** 2 + (city2[1] - city1[1]) ** 2)

# Total distance of a route
def total_distance(route, cities):
    dist = 0
    for i in range(len(route) - 1):
        dist += distance(cities[route[i]], cities[route[i+1]])
    dist += distance(cities[route[-1]], cities[route[0]])  # return to start city
    return dist

# Fitness function (inverse of the total distance)
def fitness(route, cities):
    return 1 / total_distance(route, cities)

# Initialize a population (random routes)
def initialize_population(num_cells, num_cities):
    return [random.sample(range(num_cities), num_cities) for _ in range(num_cells)]

# Get neighbors of a cell (simplified to adjacent cells in the list)
def get_neighbors(population, index):
    neighbors = []
    if index > 0:
        neighbors.append(population[index - 1])
    if index < len(population) - 1:
        neighbors.append(population[index + 1])
    return neighbors

# Update state of a cell by moving towards the best neighbor (simplified approach)
def update_state(cell, neighbors, cities):
    best_neighbor = min(neighbors, key=lambda x: total_distance(x, cities))
    # Randomly swap a portion of the route to simulate an update
    swap_indices = random.sample(range(len(cell)), 2)
    new_cell = cell[:]
    new_cell[swap_indices[0]], new_cell[swap_indices[1]] = new_cell[swap_indices[1]], new_cell[swap_indices[0]]
    return new_cell

# Parallel fitness evaluation function
def evaluate_cell(cell, cities):
    return fitness(cell, cities)

# Parallel update function (simplified)
def parallel_update_population(population, cities):
    with Pool() as pool:
        fitness_values = pool.starmap(evaluate_cell, [(cell, cities) for cell in population])

    updated_population = []
    for i in range(len(population)):
        neighbors = get_neighbors(population, i)
        updated_population.append(update_state(population[i], neighbors, cities))

    return updated_population

# Main parallel cellular algorithm
def parallel_cellular_algorithm(num_cities, num_cells, iterations):
    cities = generate_cities(num_cities)
    population = initialize_population(num_cells, num_cities)

    for _ in range(iterations):
        population = parallel_update_population(population, cities)

    # Track the best solution found
    best_solution = min(population, key=lambda x: total_distance(x, cities))
    return best_solution, total_distance(best_solution, cities)

# Running the algorithm
best_route, best_distance = parallel_cellular_algorithm(10, 50, 100)

print(f"Best Route: {best_route}")
print(f"Total Distance: {best_distance}")
