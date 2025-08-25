import numpy as np

def fitness_function(x):
    return x ** 2

def initialize_population(pop_size, bit_length):
    population = np.random.randint(0, 2, (pop_size, bit_length))
    return population

def selection(population, fitness):
    avg_fitness = np.mean(fitness)
    expected_output_before_rounding = fitness / avg_fitness
    expected_output = np.round(expected_output_before_rounding).astype(int)
    
    total_fitness = np.sum(fitness)
    probabilities = fitness / total_fitness
    probability_percentage = probabilities * 100
    
    print(f"Expected Output before rounding: {expected_output_before_rounding}")
    print(f"Selection Probabilities: {probabilities}")
    print(f"Selection Probability Percentages: {probability_percentage}")
    
    selected_population = []
    for idx, count in enumerate(expected_output):
        selected_population.extend([population[idx]] * count)
    
    selected_population = np.array(selected_population)
    
    pop_size = len(population)
    if len(selected_population) > pop_size:
        selected_population = selected_population[np.random.choice(len(selected_population), pop_size, replace=False)]
    elif len(selected_population) < pop_size:
        extra_indices = np.random.choice(len(population), pop_size - len(selected_population))
        selected_population = np.vstack([selected_population, population[extra_indices]])
    
    return selected_population, expected_output, probabilities, probability_percentage

def crossover(population, crossover_point):
    offspring = []
    pop_size = len(population)
    
    for i in range(0, pop_size - 1, 2):
        parent1 = population[i]
        parent2 = population[i + 1]
        child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
        offspring.extend([child1, child2])
    
    if pop_size % 2 == 1:
        offspring.append(population[-1])
    
    return np.array(offspring)

def mutation(population, mutation_rate):
    mutated_population = population.copy()
    pop_size = len(population)
    
    for i in range(pop_size):
        if np.random.rand() < mutation_rate:
            rand_idx = np.random.randint(0, pop_size)
            random_chrom = population[rand_idx]
            mutated_population[i] = np.bitwise_xor(mutated_population[i], random_chrom)
    
    return mutated_population

def binary_to_decimal(binary):
    return int(''.join(map(str, binary)), 2)

def genetic_algorithm(pop_size=4, bit_length=5, generations=5, mutation_rate=0.05, crossover_point=2):
    population = initialize_population(pop_size, bit_length)
    
    for generation in range(generations):
        print(f"\nGeneration {generation + 1}:")
        decimal_values = np.array([binary_to_decimal(ind) for ind in population])
        fitness_values = fitness_function(decimal_values)
        avg_fitness = np.mean(fitness_values)
        max_fitness = np.max(fitness_values)
        
        print(f"Population:\n{population}")
        print(f"Fitness Values: {fitness_values}")
        print(f"Avg Fitness: {avg_fitness}, Max Fitness: {max_fitness}")
        
        selected_population, expected_output, probabilities, probability_percentage = selection(population, fitness_values)
        print(f"Expected Output (rounded): {expected_output}")
        
        offspring = crossover(selected_population, crossover_point)
        print(f"Offspring after Crossover:\n{offspring}")
        
        mutated_population = mutation(offspring, mutation_rate)
        print(f"Mutated Population:\n{mutated_population}")
        
        population = mutated_population
    
    return population

final_population = genetic_algorithm()
print(f"\nFinal Population:\n{final_population}")
