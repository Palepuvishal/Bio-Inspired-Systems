import numpy as np
import random
from typing import List, Tuple, Dict, Any

# --- 1. Configuration and Synthetic Data Setup ---

# Synthetic financial data: 100 examples with 3 features and a target relative return
NUM_EXAMPLES = 100
FEATURES = ['Earnings_Surprise', 'Volatility', 'Volume_Change']
OPERATORS = ['>', '<']

# Simulate stock data: Features and a 12-week actual relative return
DATA = {
    'Earnings_Surprise': np.random.uniform(-0.15, 0.20, NUM_EXAMPLES), 
    'Volatility': np.random.uniform(0.01, 0.15, NUM_EXAMPLES),        
    'Volume_Change': np.random.uniform(-0.50, 0.50, NUM_EXAMPLES),     
    'Actual_Return': np.random.uniform(-0.05, 0.10, NUM_EXAMPLES)    # The 12-week relative return
}
DATA_SIZE = len(DATA['Actual_Return'])


# --- 2. Rule and Individual Representation (Chromosome) ---

class Rule:
    """Represents a single GA chromosome (prediction rule)."""
    def __init__(self, conditions: List[Tuple[str, str, float]], prediction: str):
        # conditions: list of (feature, operator, threshold) tuples
        self.conditions = conditions
        # prediction: 'Up' or 'Down' (relative to market)
        self.prediction = prediction
        self.fitness = -np.inf # Average Magnitude Score

    def __repr__(self):
        cond_str = " AND ".join([f"{f} {op} {t:.3f}" for f, op, t in self.conditions])
        return f"IF [{cond_str}] THEN Prediction = {self.prediction} (Fitness: {self.fitness:.4f})"

# --- 3. Genetic Algorithm Class ---

class FinancialGATrainer:
    """Implements the GA lifecycle for rule generation."""
    def __init__(self, pop_size=100, max_generations=50, cross_rate=0.8, mut_rate=0.1):
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.cross_rate = cross_rate
        self.mut_rate = mut_rate
        self.population: List[Rule] = []

    # --- Step 2: Initialization ---
    def _initialize_rule(self, num_conditions=3) -> Rule:
        """Creates a random prediction rule."""
        conditions = []
        for _ in range(num_conditions):
            feature = random.choice(FEATURES)
            op = random.choice(OPERATORS)
            # Threshold chosen randomly within feature's range
            min_val = np.min(DATA[feature])
            max_val = np.max(DATA[feature])
            threshold = random.uniform(min_val, max_val)
            conditions.append((feature, op, threshold))
        
        prediction = random.choice(['Up', 'Down'])
        return Rule(conditions, prediction)

    # --- Step 3: Fitness Function (Average Magnitude Score) ---
    def _calculate_fitness(self, rule: Rule) -> float:
        """Calculates the Average Magnitude Score for a rule."""
        
        scores = []
        
        for i in range(DATA_SIZE):
            # Check if all conditions are met (Rule application/Prediction)
            is_predicted = True
            for feature, op, threshold in rule.conditions:
                value = DATA[feature][i]
                if op == '>' and not (value > threshold):
                    is_predicted = False; break
                elif op == '<' and not (value < threshold):
                    is_predicted = False; break
            
            if is_predicted:
                actual_relative_return = DATA['Actual_Return'][i]
                actual_direction = 'Up' if actual_relative_return > 0 else 'Down'

                magnitude = abs(actual_relative_return)
                
                # Score logic: +|return| for correct, -|return| for incorrect
                if rule.prediction == actual_direction:
                    scores.append(magnitude)
                else:
                    scores.append(-magnitude)
        
        # Score is averaged over ALL examples (NUM_EXAMPLES), not just predicted ones.
        if not scores:
            return 0.0 # No predictions made (zero magnitude score)
        
        # Scale scores back up to a percentage base for better comparison/logging
        average_score = np.sum(scores) / NUM_EXAMPLES * 100 
        return average_score

    # --- Step 4: Selection (Roulette Wheel) ---
    def _selection(self) -> Rule:
        """Selects a rule based on its relative fitness."""
        fitnesses = np.array([r.fitness for r in self.population])
        # Adjust fitness to be non-negative for probability calculation
        adjusted_fitness = fitnesses - np.min(fitnesses) + 1e-6 
        
        probabilities = adjusted_fitness / np.sum(adjusted_fitness)
        
        idx = np.random.choice(len(self.population), p=probabilities)
        return self.population[idx]

    # --- Step 5: Crossover (Single-point on conditions) ---
    def _crossover(self, parent1: Rule, parent2: Rule) -> Rule:
        """Performs single-point crossover on the rule conditions."""
        len1, len2 = len(parent1.conditions), len(parent2.conditions)
        if min(len1, len2) < 2:
            return random.choice([parent1, parent2]) # No room for crossover

        crossover_point = random.randint(1, min(len1, len2) - 1) 

        child_conditions = (
            parent1.conditions[:crossover_point] + 
            parent2.conditions[crossover_point:]
        )
        child_prediction = random.choice([parent1.prediction, parent2.prediction])
        
        return Rule(child_conditions, child_prediction)

    # --- Step 6: Mutation ---
    def _mutation(self, rule: Rule) -> Rule:
        """Applies random, small changes to the rule's conditions or prediction."""
        
        new_conditions = list(rule.conditions)
        
        # Mutate a condition (e.g., change threshold, operator, or feature)
        if new_conditions and random.random() < 0.8:
            idx = random.randrange(len(new_conditions))
            cond = list(new_conditions[idx])
            
            if random.random() < 0.2: # Feature Mutation
                cond[0] = random.choice(FEATURES)
            elif random.random() < 0.2: # Operator Mutation
                cond[1] = random.choice(OPERATORS)
            else: # Threshold Mutation
                feature = cond[0]
                min_val = np.min(DATA[feature])
                max_val = np.max(DATA[feature])
                cond[2] = random.uniform(min_val, max_val)

            new_conditions[idx] = tuple(cond)
        
        # Mutate prediction ('Up' <-> 'Down')
        new_prediction = rule.prediction
        if random.random() < 0.05:
            new_prediction = 'Down' if rule.prediction == 'Up' else 'Up'

        return Rule(new_conditions, new_prediction)

    # --- 4. Execution Flow ---
    def run(self):
        """Runs the main GA loop (Selection -> Crossover -> Mutation -> Replacement)."""
        
        # Initialize population
        self.population = [self._initialize_rule() for _ in range(self.pop_size)]
        
        best_rule = None
        best_fitness = -np.inf
        
        print(f"--- Starting Genetic Algorithm for {self.max_generations} Generations ---")
        print("Goal: Maximize Average Magnitude Score (in %)")
        
        for gen in range(self.max_generations):
            # Evaluate current population (Step 3)
            for rule in self.population:
                rule.fitness = self._calculate_fitness(rule)

            # Find current best rule (for Elitism)
            current_best = max(self.population, key=lambda r: r.fitness)
            if current_best.fitness > best_fitness:
                best_fitness = current_best.fitness
                best_rule = current_best
            
            # Print generation summary
            avg_fitness = np.mean([r.fitness for r in self.population])
            print(f"Gen {gen+1:02d}: Max Fitness = {best_fitness:.2f}% | Avg Fitness = {avg_fitness:.2f}%")

            # Create the next generation (Step 7: Replacement)
            next_population: List[Rule] = [best_rule] # Elitism: preserve the best rule
            
            while len(next_population) < self.pop_size:
                parent1 = self._selection()
                parent2 = self._selection()

                child = parent1

                # Crossover (Step 5)
                if random.random() < self.cross_rate:
                    child = self._crossover(parent1, parent2)
                
                # Mutation (Step 6)
                if random.random() < self.mut_rate:
                    child = self._mutation(child)
                
                next_population.append(child)

            self.population = next_population
        
        print("\n--- GA Run Complete ---")
        return best_rule

# --- Run the Simulation ---
if __name__ == "__main__":
    # Parameters simulate a modest search (200 individuals, 100 generations)
    ga_solver = FinancialGATrainer(pop_size=200, max_generations=100)
    final_rule = ga_solver.run()

    print("\nFinal Best Rule Generated (Approximating the paper's output):")
    print(final_rule)
