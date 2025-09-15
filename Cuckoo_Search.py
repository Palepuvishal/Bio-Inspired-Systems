"""
Cuckoo Search Algorithm (CSA) adapted for TSP (permutation-based)

- Input: distance matrix D (NxN)
- Output: best permutation (tour) and its length
- Author: concise CSA→TSP implementation
"""

import numpy as np

# ---------------------------
# Utility / operator functions
# ---------------------------

def euclidean_distance_matrix(coords):
    coords = np.asarray(coords, dtype=float)
    n = coords.shape[0]
    dif = coords.reshape(n,1,-1) - coords.reshape(1,n,-1)
    return np.sqrt((dif**2).sum(axis=2))

def tour_length_from_perm(perm, D):
    perm = np.asarray(perm, dtype=int)
    return D[perm, np.roll(perm, -1)].sum()

def random_perm(n):
    return np.random.permutation(n)

def levy_k(beta, dim, cap=None):
    """
    Discrete heavy-tailed sampler (analogous to Lévy flight length).
    Returns an integer k >= 1 indicating how many permutation operators to apply.
    Uses a Pareto-like discrete sampling.
    """
    if cap is None:
        cap = max(1, dim // 2)
    # sample from pareto (numpy pareto's shape = alpha), shift by +1
    x = 1 + int(np.random.pareto(beta))
    return min(max(1, x), cap)

def two_opt_inversion(perm):
    """Random 2-opt: reverse a random subsequence."""
    n = len(perm)
    i, j = np.random.choice(n, 2, replace=False)
    if i > j:
        i, j = j, i
    new = perm.copy()
    new[i:j+1] = new[i:j+1][::-1]
    return new

def swap_two_positions(perm):
    """Swap two random positions."""
    n = len(perm)
    a, b = np.random.choice(n, 2, replace=False)
    new = perm.copy()
    new[a], new[b] = new[b], new[a]
    return new

def insertion_move(perm):
    """Remove an element and insert it at a random position."""
    n = len(perm)
    a, b = np.random.choice(n, 2, replace=False)
    new = list(perm)
    val = new.pop(a)
    new.insert(b, val)
    return np.array(new, dtype=int)

def guided_insert_from_best(curr, best):
    """
    Guided move: pick a random block from 'best' and insert it into 'curr' (preserving order),
    keeping a valid permutation. This nudges 'curr' towards 'best'.
    """
    n = len(curr)
    if n < 4:
        return two_opt_inversion(curr)
    i, j = np.random.choice(n, 2, replace=False)
    if i > j:
        i, j = j, i
    block = best[i:j+1].tolist()
    curr_list = [c for c in curr if c not in block]
    pos = np.random.randint(0, len(curr_list)+1)
    new = curr_list[:pos] + block + curr_list[pos:]
    return np.array(new, dtype=int)

def levy_move_permutation(perm, best, k):
    """
    Apply k permutation operators to perm.
    Operators are chosen probabilistically:
    - using guided insert occasionally to pull toward 'best'
    - using 2-opt, swap, insertion for diversity
    """
    new = perm.copy()
    for _ in range(k):
        r = np.random.rand()
        if r < 0.2:
            new = guided_insert_from_best(new, best)
        elif r < 0.55:
            new = two_opt_inversion(new)
        elif r < 0.8:
            new = insertion_move(new)
        else:
            new = swap_two_positions(new)
    return new

# ---------------------------
# Main CSA-TSP implementation
# ---------------------------

def cuckoo_search_tsp(D, n_nests=30, p_a=0.25, beta=1.5,
                      max_gen=1000, seed=None, verbose=False):
    """
    Cuckoo Search for TSP (permutation variant).
    D        : NxN distance matrix
    n_nests  : population size (number of nests)
    p_a      : discovery probability (fraction replaced each generation)
    beta     : Lévy-like exponent (discrete heavy-tail, e.g., 1.5)
    max_gen  : number of generations
    seed     : random seed (optional)
    Returns: best_perm (1D numpy array) and best_length (float)
    """
    if seed is not None:
        np.random.seed(seed)

    n_cities = D.shape[0]
    # initialize nests (list of numpy arrays)
    nests = [random_perm(n_cities) for _ in range(n_nests)]
    fitness = np.array([tour_length_from_perm(n, D) for n in nests])
    best_idx = np.argmin(fitness)
    best = nests[best_idx].copy()
    best_f = fitness[best_idx]

    if verbose:
        print(f"Initial best length: {best_f:.6f}")

    for gen in range(1, max_gen + 1):
        # 1) Generate cuckoo proposals via discrete Lévy moves
        for i in range(n_nests):
            k = levy_k(beta, n_cities)
            candidate = levy_move_permutation(nests[i], best, k)
            f_new = tour_length_from_perm(candidate, D)
            if f_new < fitness[i]:
                nests[i] = candidate
                fitness[i] = f_new
                if f_new < best_f:
                    best_f = f_new
                    best = candidate.copy()

        # 2) Abandon fraction p_a of nests and replace with new random permutations
        replace_mask = np.random.rand(n_nests) < p_a
        for i in np.where(replace_mask)[0]:
            nests[i] = random_perm(n_cities)
            fitness[i] = tour_length_from_perm(nests[i], D)
            if fitness[i] < best_f:
                best_f = fitness[i]
                best = nests[i].copy()

        if verbose and (gen % (max(1, max_gen//10)) == 0 or gen == 1):
            print(f"Gen {gen:4d}  best = {best_f:.6f}")

    return best, best_f

# ---------------------------
# Example usage (if run as a script)
# ---------------------------

if __name__ == "__main__":
    # Example with random 30 cities
    n_cities = 30
    coords = np.random.rand(n_cities, 2) * 100.0
    D = euclidean_distance_matrix(coords)

    best_perm, best_len = cuckoo_search_tsp(
        D,
        n_nests=40,
        p_a=0.25,
        beta=1.5,
        max_gen=2000,
        seed=123,
        verbose=True
    )

    print("\nBest length:", best_len)
    print("Best tour:", best_perm.tolist())
