import numpy as np


def tour_length(perm, D):
    n = len(perm)
    length = 0
    for i in range(n):
        city_from = perm[i]
        city_to = perm[(i + 1) % n]
        length += D[city_from, city_to]
    return length

def random_perm(dim):
    return np.random.permutation(dim)

def two_opt_inversion(perm, a, b):
    new_perm = perm.copy()
    sub_sequence = new_perm[a:b+1]
    new_perm[a:b+1] = sub_sequence[::-1]
    return new_perm

def insertion_move(perm, a, b):
    new_perm = list(perm.copy())
    city_to_move = new_perm.pop(a)
    new_perm.insert(b, city_to_move)
    return np.array(new_perm)

def swap_move(perm, a, b):
    new_perm = perm.copy()
    new_perm[a], new_perm[b] = new_perm[b], new_perm[a]
    return new_perm

def guided_move(curr_perm, best_perm):
    n = len(curr_perm)
    best_city_idx = np.random.randint(n)
    best_city = best_perm[best_city_idx]
    
    curr_city_idx = np.where(curr_perm == best_city)[0][0]
    
    new_perm = list(curr_perm.copy())
    new_perm.pop(curr_city_idx)
    new_perm.insert(best_city_idx, best_city)
    
    return np.array(new_perm)

def levy_k(beta, dim):
    
    sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
             (np.math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
    u = np.random.normal(0, sigma)
    v = np.random.normal(0, 1)
    
    X = np.abs(u / (v**(1 / beta)))
    
    k = np.floor(X)
    
    k = max(1, k)
    k = min(k, np.floor(dim / 2))
    
    return int(k)

def levy_move(perm, best_perm, k):
    new_perm = perm.copy()
    dim = len(perm)
    
    for _ in range(k):
        r = np.random.rand()
        
        if r < 0.2:
            new_perm = guided_move(new_perm, best_perm)
        elif r < 0.55:
            a, b = np.random.choice(dim, 2, replace=False)
            a, b = min(a, b), max(a, b)
            new_perm = two_opt_inversion(new_perm, a, b)
        elif r < 0.8:
            a, b = np.random.choice(dim, 2, replace=False)
            new_perm = insertion_move(new_perm, a, b)
        else:
            a, b = np.random.choice(dim, 2, replace=False)
            new_perm = swap_move(new_perm, a, b)
            
    return new_perm


def cuckoo_search_tsp(D, n_nests, p_a, beta, max_gen):
    dim = D.shape[0]

    nests = np.array([random_perm(dim) for _ in range(n_nests)])

    F = np.array([tour_length(nest, D) for nest in nests])

    best_idx = np.argmin(F)
    best_perm = nests[best_idx].copy()
    best_length = F[best_idx]
    
    print(f"Initial Best Length: {best_length:.4f}")

    for gen in range(1, max_gen + 1):
        
        for i in range(n_nests):
            k = levy_k(beta, dim)
            
            candidate_perm = levy_move(nests[i], best_perm, k)
            f_new = tour_length(candidate_perm, D)

            if f_new < F[i]:
                nests[i] = candidate_perm.copy()
                F[i] = f_new

                if f_new < best_length:
                    best_perm = candidate_perm.copy()
                    best_length = f_new
        
        for i in range(n_nests):
            if np.random.rand() < p_a:
                nests[i] = random_perm(dim)
                F[i] = tour_length(nests[i], D)
                
                if F[i] < best_length:
                    best_perm = nests[i].copy()
                    best_length = F[i]
        
        if gen % 10 == 0 or gen == max_gen:
            print(f"Generation {gen}/{max_gen} | Current Best Length: {best_length:.4f}")

    return best_perm, best_length


if __name__ == '__main__':
    np.random.seed(42)
    num_cities = 10
    coords = np.random.rand(num_cities, 2) * 100
    
    D = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(num_cities):
            if i != j:
                D[i, j] = np.linalg.norm(coords[i] - coords[j])
    
    N_NESTS = 25
    P_A = 0.25
    BETA = 1.5
    MAX_GEN = 100
    
    print(f"Starting Cuckoo Search for TSP with {num_cities} cities...")
    
    final_tour, final_length = cuckoo_search_tsp(D, N_NESTS, P_A, BETA, MAX_GEN)
    
    print("\n=============================================")
    print("Optimization Complete")
    print(f"Final Shortest Path Length: {final_length:.4f}")
    print(f"Best Tour Found: {final_tour.tolist()}")
    print("=============================================")
