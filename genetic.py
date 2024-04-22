import time

import numpy as np

# decode bitstring to numbers
def decode(bounds, n_bits, bitstring):
    decoded = list()
    largest = 2**n_bits
    for i in range(len(bounds)):
        start, end = i * n_bits, (i * n_bits)+n_bits
        substring = bitstring[start:end]
        chars = ''.join([str(s) for s in substring])
        # convert base 2 to base 10
        integer = int(chars, 2)
        value = bounds[i][0] + (integer/largest) * (bounds[i][1] - bounds[i][0])
        decoded.append(value)
    return decoded

# tournament selection
def selection(pop, scores, rng, k=3):
    selection_ix = rng.integers(len(pop))
    for ix in rng.integers(0, len(pop), k-1):
        if scores[ix] < scores[selection_ix]:
            selection_ix = ix
    return pop[selection_ix]

# crossover two parents to create two children
def crossover(p1, p2, r_cross, rng):
    c1, c2 = p1.copy(), p2.copy()
    if rng.random() < r_cross:
        pt = rng.integers(1, len(p1)-2)
        c1 = p1[:pt] + p2[pt:]
        c2 = p2[:pt] + p1[pt:]
    return (c1, c2)

# mutation operator
def mutation(bitstring, r_mut, rng):
    for i in range(len(bitstring)):
        if rng.random() < r_mut:
            bitstring[i] = 1 - bitstring[i]

# genetic algorithm
def genetic_algorithm(objective, bounds, n_bits, n_iter, n_pop, r_cross, r_mut, rng):
    pop = [rng.integers(0, 2, n_bits*len(bounds)).tolist() for _ in range(n_pop)]
    best, best_eval = 0, objective(decode(bounds, n_bits, pop[0]))

    first_trial = True

    # go through different generations
    for gen in range(n_iter):
        found_new_best = False

        if first_trial:
            s_time = time.time()            

        decoded = [decode(bounds, n_bits, p) for p in pop]
        scores = [objective(d) for d in decoded]

        if first_trial:
            print(f"Estimated generation runtime: {time.time()-s_time:.2f}s/generation")
            first_trial = False

        for i in range(n_pop):
            if scores[i] < best_eval:
                best, best_eval = pop[i], scores[i]
                best_decoded = decoded[i]
                found_new_best = True

        print(f"Gen {gen}: best f({best_decoded}) = {best_eval:.4f} {'**' if found_new_best else ''}")

        # select parents
        selected = [selection(pop, scores, rng) for _ in range(n_pop)]
        children = list()
        for i in range(0, n_pop, 2):
            p1, p2 = selected[i], selected[i+1]
            for c in crossover(p1, p2, r_cross, rng):
                mutation(c, r_mut, rng)
                children.append(c)
        pop = children

    return [np.array(best), best_eval]

def optimize(objective, lbs, ubs, n_iter=100, n_pop=100, seed=None):
    """ 
    Optimizes objective
    :param objective: black-box function that takes in tuple for bounds
    """
    bounds = [[lb, ub] for (lb,ub) in zip(lbs, ubs)]
    n_bits = 16
    r_cross = 0.9
    r_mut = 1./(n_bits * len(bounds))
    rng = np.random.default_rng(seed)

    best, score = genetic_algorithm(objective, bounds, n_bits, n_iter, n_pop, r_cross, r_mut, rng)
    x = np.array(decode(bounds, n_bits, best))

    return x
