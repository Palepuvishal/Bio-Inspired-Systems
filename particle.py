import numpy as np

# Objective function
def f(x):
    return -x**2 + 20*x + 5

def particle_swarm_optimization(num_particles=9, num_iterations=5, pos_bounds=(0, 15)):
    # Generate random initial positions within bounds
    positions = np.random.uniform(pos_bounds[0], pos_bounds[1], size=num_particles)
    velocities = np.zeros_like(positions)

    # Initialize personal bests
    pbest_positions = positions.copy()
    pbest_scores = f(positions)

    # Initialize global best
    gbest_position = pbest_positions[np.argmax(pbest_scores)]

    # PSO hyperparameters
    c1 = c2 = 1
    w = 1

    print("Initial positions:\n", positions.round(4))
    print("Initial function values:\n", f(positions).round(4))
    print("Initial global best position:", round(gbest_position, 4), "with value:", round(f(gbest_position), 4))
    print("-" * 60)

    # Iterate
    for t in range(num_iterations):
        r1 = np.random.rand()
        r2 = np.random.rand()

        for i in range(len(positions)):
            velocities[i] = (
                w * velocities[i] +
                c1 * r1 * (pbest_positions[i] - positions[i]) +
                c2 * r2 * (gbest_position - positions[i])
            )

        # Update positions
        positions += velocities
        scores = f(positions)

        # Update personal bests
        for i in range(len(positions)):
            if scores[i] > pbest_scores[i]:
                pbest_positions[i] = positions[i]
                pbest_scores[i] = scores[i]

        # Update global best
        gbest_position = pbest_positions[np.argmax(pbest_scores)]

        # Display iteration results
        print(f"Iteration {t + 1}")
        print("r1 =", round(r1, 4), ", r2 =", round(r2, 4))
        print("Positions:", positions.round(4))
        print("Velocities:", velocities.round(4))
        print("Function values:", scores.round(4))
        print("Global best position:", round(gbest_position, 4), "with value:", round(f(gbest_position), 4))
        print("-" * 60)

# Run the PSO function
particle_swarm_optimization()
