import numpy as np
import random
import matplotlib.pyplot as plt

# Step 1: Define the Problem (cities and coordinates)
num_cities = 10
cities = []

# Minimum distance between cities to avoid clustering
min_distance = 5

# Generate cities ensuring they are sufficiently far apart
while len(cities) < num_cities:
    # Generate a random city
    new_city = np.random.rand(1, 2) * 100
    # Check if it's sufficiently far from all existing cities
    if all(np.linalg.norm(new_city - np.array(city)) >= min_distance for city in cities):
        cities.append(new_city[0])

cities = np.array(cities)

# Function to calculate the distance matrix
def calculate_distance_matrix(cities):
    n = len(cities)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist[i][j] = np.linalg.norm(cities[i] - cities[j])
    return dist

distance_matrix = calculate_distance_matrix(cities)

# Step 2: Initialize Parameters for ACO
num_ants = 20
num_iterations = 100
alpha = 1.0          # pheromone importance
beta = 5.0           # heuristic importance
rho = 0.5            # evaporation rate
Q = 100              # constant for pheromone update
initial_pheromone = 1.0
pheromone = np.ones((num_cities, num_cities)) * initial_pheromone

# Step 3: Construct Solutions
# Function to calculate probability of visiting city j from city i
def probability(i, j, pheromone, distance_matrix, visited):
    if j in visited:
        return 0  # Don't revisit cities
    pher = pheromone[i][j] ** alpha
    heuristic = (1.0 / distance_matrix[i][j]) ** beta
    return pher * heuristic

# Function to construct a solution (tour) for an ant
def construct_solution(pheromone, distance_matrix):
    solution = []
    visited = set()
    current_city = random.randint(0, num_cities - 1)
    solution.append(current_city)
    visited.add(current_city)

    while len(visited) < num_cities:
        probs = []
        for j in range(num_cities):
            prob = probability(current_city, j, pheromone, distance_matrix, visited)
            probs.append(prob)
        probs = np.array(probs)
        probs /= probs.sum()  # normalize to form probability distribution
        next_city = np.random.choice(range(num_cities), p=probs)
        solution.append(next_city)
        visited.add(next_city)
        current_city = next_city

    return solution

# Step 4: Update Pheromones
# Function to update pheromones based on the solutions found
def update_pheromones(pheromone, all_solutions, distance_matrix):
    pheromone *= (1 - rho)  # Evaporate pheromones

    for path, length in all_solutions:
        for i in range(len(path)):
            from_city = path[i]
            to_city = path[(i + 1) % num_cities]  # Return to the starting city
            pheromone[from_city][to_city] += Q / length
            pheromone[to_city][from_city] += Q / length  # Undirected graph

# Step 5: Path Length Calculation
# Function to calculate the total length of a tour
def path_length(path, distance_matrix):
    length = 0
    for i in range(len(path)):
        from_city = path[i]
        to_city = path[(i + 1) % num_cities]  # Return to the starting city
        length += distance_matrix[from_city][to_city]
    return length

# Step 6: Main ACO Loop
best_path = None
best_length = float('inf')

for iteration in range(num_iterations):
    all_solutions = []
    for ant in range(num_ants):
        solution = construct_solution(pheromone, distance_matrix)
        length = path_length(solution, distance_matrix)
        all_solutions.append((solution, length))

        if length < best_length:
            best_length = length
            best_path = solution

    update_pheromones(pheromone, all_solutions, distance_matrix)
    print(f"Iteration {iteration+1}/{num_iterations} - Best Length: {best_length:.2f}")

# Step 7: Output the Best Solution
print("Best tour found:", best_path)
print("Shortest path length:", best_length)

# Step 8: (Optional) Plotting the Best Tour and All Cities
def plot_tour(cities, path):
    plt.figure(figsize=(8, 6))

    # Plot all the cities
    plt.scatter(cities[:, 0], cities[:, 1], color='red', marker='o', s=100, label="Cities")
    for i, (x, y) in enumerate(cities):
        plt.text(x + 1, y + 1, str(i), color='black', fontsize=12)

    # Plot the best route found by the ants (with arrows to show direction)
    tour = path + [path[0]]  # Return to the start
    for i in range(len(path)):
        start = cities[path[i]]
        end = cities[path[(i + 1) % num_cities]]
        plt.arrow(start[0], start[1], end[0] - start[0], end[1] - start[1],
                  head_width=2, head_length=3, fc='blue', ec='blue')  # Blue arrows to indicate direction

    plt.title("Best TSP Tour Found by ACO")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.show()

plot_tour(cities, best_path)
