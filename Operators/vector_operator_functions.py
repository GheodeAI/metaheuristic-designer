import math
import random
import numpy as np
import scipy as sp
import scipy.stats


def xorMask(vector, n, mode="byte"):
    """
    Applies an XOR operation between a random number and the input vector.
    """

    mask_pos = np.hstack([np.ones(n), np.zeros(vector.size - n)]).astype(bool)
    np.random.shuffle(mask_pos)

    if mode == "bin":
        mask = mask_pos
    elif mode == "byte":
        mask = np.random.randint(1, 0xFF, size=vector.shape) * mask_pos
    elif mode == "int":
        mask = np.random.randint(1, 0xFFFF, size=vector.shape) * mask_pos

    return vector ^ mask


def xorCross(vector1, vector2):
    """
    Applies the XOR operation between each component of the input vectors.
    """

    return vector1 ^ vector2


def permutation(vector, n):
    """
    Randomly permutes 'n' of the components of the input vector.
    """

    mask_pos = np.hstack([np.ones(n), np.zeros(vector.size - n)]).astype(bool)
    np.random.shuffle(mask_pos)

    if np.count_nonzero(mask_pos==1) < 2:
        mask[random.sample(range(mask_pos.size), 2)] = 1 
    np.random.shuffle(vector[mask_pos])
    return vector


def mutateRand(vector, population, params):
    """
    Adds random noise with a given probability distribution to 'n' components of the input vector.
    """

    method = params["method"]
    n = round(params["N"])

    low = params["Low"] if "Low" in params else -1
    up = params["Up"] if "Low" in params else 1
    strength = params["F"] if "F" in params else 1
    
    mask_pos = np.hstack([np.ones(n), np.zeros(vector.size - n)]).astype(bool)
    np.random.shuffle(mask_pos)
    
    rand_vec = sampleDistribution(method, n, 0, strength, low, up)
    
    vector[mask_pos] = vector[mask_pos] + rand_vec
    return vector


def mutateSample(vector, population, params):
    """
    Replaces 'n' components of the input vector with a random value sampled from a given probability distribution.
    """

    method = params["method"]
    n = round(params["N"])

    low = params["Low"] if "Low" in params else -1
    up = params["Up"] if "Low" in params else 1
    strength = params["F"] if "F" in params else 1

    mask_pos = np.hstack([np.ones(n), np.zeros(vector.size - n)]).astype(bool)
    np.random.shuffle(mask_pos)
    popul_matrix = np.vstack([i.vector for i in population])
    mean = popul_matrix.mean(axis=0)[mask_pos]
    std = (popul_matrix.std(axis=0)[mask_pos] + 1e-6)*strength # ensure there will be some standard deviation
    
    rand_vec = sampleDistribution(method, n, mean, std, low, up)
    
    vector[mask_pos] = rand_vec
    return vector


def randSample(vector, population, params):
    """
    Picks a vector with components sampled from a probability distribution.
    """

    method = params["method"]
    
    low = params["Low"] if "Low" in params else -1
    up = params["Up"] if "Low" in params else 1
    strength = params["F"] if "F" in params else 1

    popul_matrix = np.vstack([i.vector for i in population])
    mean = popul_matrix.mean(axis=0)
    std = (popul_matrix.std(axis=0) + 1e-6)*strength # ensure there will be some standard deviation
    
    rand_vec = sampleDistribution(method, vector.shape, mean, std, low, up)
    
    return rand_vec


def randNoise(vector, params):
    """
    Adds random noise with a given probability distribution to all components of the input vector.
    """

    method = params["method"]

    low = params["Low"] if "Low" in params else -1
    up = params["Up"] if "Low" in params else 1
    strength = params["F"] if "F" in params else 1
    
    noise = sampleDistribution(method, vector.shape, 0, strength, low, up)
    
    return vector + noise

"""
-Distribución zeta
-Distribución hipergeométrica
-Distribución geomética
-Distribución de Boltzman
-Distribución de Pascal (binomial negativa)
"""

def sampleDistribution(method, n, mean=0, strength=0.01, low=0, up=1):
    """
    Takes 'n' samples from a given probablility distribution and returns them as a vector.
    """

    sample = 0 
    if method == "gauss":
        sample = np.random.normal(mean, strength, size=n)
    elif method == "uniform":
        sample = np.random.uniform(low, up, size=n)
    elif method == "cauchy":
        sample = sp.stats.cauchy.rvs(mean, strength, size=n)
    elif method == "laplace":
        sample = sp.stats.laplace.rvs(mean, strength, size=n)
    elif method == "poisson":
        sample = sp.stats.poisson.rvs(strength, size=n)
    elif method == "bernouli":
        sample = sp.stats.bernoulli.rvs(strength, size=n)
    else:
        print(f"Error: distribution \"{method}\" not defined")
        exit(1)
    return sample


def laplace(vector, strength):
    """
    Adds random noise following a Laplace distribution to the vector.
    """

    return randNoise(vector, {"method":"laplace", "F":strength})


def cauchy(vector, strength):
    """
    Adds random noise following a Cauchy distribution to the vector.
    """
    
    return randNoise(vector, {"method":"cauchy", "F":strength})


def gaussian(vector, strength):
    """
    Adds random noise following a Gaussian distribution to the vector.
    """
    
    return randNoise(vector, {"method":"gauss", "F":strength})


def uniform(vector, low, up):
    """
    Adds random noise following an Uniform distribution to the vector.
    """
    
    return randNoise(vector, {"method":"uniform", "Low":low, "Up":up})


def poisson(vector, mu):
    """
    Adds random noise following a Poisson distribution to the vector.
    """
    
    return randNoise(vector, {"method":"poisson", "F":mu})

# def bernoulli(vector, p):
#     """
#     Adds random noise following a Poisson distribution to the vector.
#     """
    
#     return randNoise(vector, {"method":"Bernoulli", "F":p})



def sample_1_sigma(vector, n, epsilon, tau):
    """
    Replaces 'n' components of the input vector with a value sampled from the mutate 1 sigma function.

    In future, it should be integrated in mutateSample and sampleDistribution functions, considering 
    np.exp(tau * N(0,1)) as a distribution function with a min value of epsilon.
    """

    mask_pos = np.hstack([np.ones(n), np.zeros(vector.size - n)]).astype(bool)
    np.random.shuffle(mask_pos)

    sampled = np.array([mutate_1_sigma(vector[pos], epsilon, tau) for pos in mask_pos])
    vector[mask_pos] = sampled[mask_pos]
    return vector


def mutate_1_sigma(sigma, epsilon, tau):
    """
    Mutate a sigma value in base of tau param, where epsilon is de minimum value that a sigma can have.
    """

    return max(epsilon, np.exp(tau * np.random.normal()))


def mutate_n_sigmas(list_sigmas, epsilon, tau, tau_multiple):
    """
    Mutate a list of sigmas values in base of tau and tau_multiple params, where epsilon is de minimum value that a sigma can have.
    """

    base_tau = tau * np.random.normal()
    new_sigmas = [max(epsilon, sigma * np.exp(base_tau + tau_multiple * np.random.normal())) for sigma in list_sigmas]
    return new_sigmas


def cross1p(vector1, vector2):
    """
    Performs a 1 point cross between two vectors.
    """
    
    cross_point = random.randrange(0, vector1.size)
    return np.hstack([vector1[:cross_point], vector2[cross_point:]])


def cross2p(vector1, vector2):
    """
    Performs a 2 point cross between two vectors.
    """
    
    cross_point1 = random.randrange(0, vector1.size-2)
    cross_point2 = random.randrange(cross_point1, vector1.size)
    return np.hstack([vector1[:cross_point1], vector2[cross_point1:cross_point2], vector1[cross_point2:]])


def crossMp(vector1, vector2):
    """
    Performs a multipoint cross between two vectors.
    """
    
    mask_pos = 1*(np.random.rand(vector1.size) > 0.5)
    aux = np.copy(vector1)
    aux[mask_pos==1] = vector2[mask_pos==1]
    return aux


def multiCross(vector, population, n_ind):
    """
    Performs a multipoint cross between the vector and 'n-1' individuals of the population
    """

    if n_ind >= len(population):
        n_ind = len(population)
    
    other_parents = random.sample(population, n_ind-1)
    mask_pos = np.random.randint(n_ind, size=vector.size) - 1
    for i in range(0, n_ind-1):
        vector[mask_pos==i] = other_parents[i].vector[mask_pos==i]
    return vector


def crossInterAvg(vector, population, n_ind):
    """
    Performs an intermediate average crossover between the vector and 'n-1' individuals the population
    """

    if n_ind >= len(population):
        n_ind = len(population)
    
    other_parents = random.sample(population, n_ind-1)
    parents = [parent.vector for parent in other_parents] + [vector]
    return np.mean(parents, axis=0)

def weightedAverage(vector1, vector2, alpha):
    """
    Performs a weighted average between the two given vectors
    """

    return alpha*vector1 + (1-alpha)*vector2


def blxalpha(vector1, vector2, alpha):
    """
    Performs the BLX alpha crossing operator between two vectors.
    """
    
    alpha *= np.random.random()
    return alpha*vector1 + (1-alpha)*vector2


def sbx(vector1, vector2, strength):
    """
    Performs the SBX crossing operator between two vectors.
    """
    
    beta = np.zeros(vector1.shape)
    u = np.random.random(vector1.shape)
    for idx, val in enumerate(u):
        if val <= 0.5:
            beta[idx] = (2*val)**(1/(strength+1))
        else:
            beta[idx] = (0.5*(1-val))**(1/(strength+1))
    
    sign = random.choice([-1,1])
    return 0.5*(vector1+vector2) + sign*0.5*beta*(vector1-vector2)


def simAnnealing(solution, strength, objfunc, temp_changes, iter):
    """
    Performs a number of rounds of Simulated Annealing using a gaussian mutation
    """
    
    p0, pf = (0.1, 7)

    alpha = 0.99
    best_fit = solution.fitness

    temp_init = temp = 100
    temp_fin = alpha**temp_changes * temp_init
    vector_new = solution.vector
    while temp >= temp_fin:
        for j in range(iter):
            vector_new = gaussian(vector_new, strength)
            new_fit = objfunc.fitness(vector_new)
            y = ((pf-p0)/(temp_fin-temp_init))*(temp-temp_init) + p0 

            p = np.exp(-y)
            if new_fit > best_fit or random.random() < p:
                best_fit = new_fit
        temp = temp*alpha
    return vector_new


def harmonySearch(vector, population, strength, HMCR, PAR):
    """
    Performs a step of the Harmony search algorithm
    """
   
    new_vector = np.zeros(vector.shape)
    popul_matrix = np.vstack([i.vector for i in population])
    popul_mean = popul_matrix.mean(axis=0)
    popul_std = popul_matrix.std(axis=0)
    for i in range(vector.size):
        if random.random() < HMCR:
            new_vector[i] = random.choice(population).vector[i]
            if random.random() <= PAR:
                new_vector[i] += np.random.normal(0,strength)
        else:
            new_vector[i] = np.random.normal(popul_mean[i], popul_std[i])
    return new_vector


def DERand1(vector, population, F, CR):
    """
    Performs the differential evolution operator DE/rand/1
    """
   
    if len(population) > 3:
        r1, r2, r3 = random.sample(population, 3)

        v = r1.vector + F*(r2.vector-r3.vector)
        mask_pos = np.random.random(vector.shape) <= CR
        vector[mask_pos] = v[mask_pos]
    return vector


def DEBest1(vector, population, F, CR):
    """
    Performs the differential evolution operator DE/best/1
    """
   
    if len(population) > 3:
        fitness = [i.fitness for i in population]
        best = population[fitness.index(max(fitness))]
        r1, r2 = random.sample(population, 2)

        v = best.vector + F*(r1.vector-r2.vector)
        mask_pos = np.random.random(vector.shape) <= CR
        vector[mask_pos] = v[mask_pos]
    return vector


def DERand2(vector, population, F, CR):
    """
    Performs the differential evolution operator DE/rand/2
    """
   
    if len(population) > 5:
        r1, r2, r3, r4, r5 = random.sample(population, 5)

        v = r1.vector + F*(r2.vector-r3.vector) + F*(r4.vector-r5.vector)
        mask_pos = np.random.random(vector.shape) <= CR
        vector[mask_pos] = v[mask_pos]
    return vector


def DEBest2(vector, population, F, CR):
    """
    Performs the differential evolution operator DE/best/2
    """
   
    if len(population) > 5:
        fitness = [i.fitness for i in population]
        best = population[fitness.index(max(fitness))]
        r1, r2, r3, r4 = random.sample(population, 4)

        v = best.vector + F*(r1.vector-r2.vector) + F*(r3.vector-r4.vector)
        mask_pos = np.random.random(vector.shape) <= CR
        vector[mask_pos] = v[mask_pos]
    return vector


def DECurrentToRand1(vector, population, F, CR):
    """
    Performs the differential evolution operator DE/current-to-rand/1
    """
   
    if len(population) > 3:
        r1, r2, r3 = random.sample(population, 3)

        v = vector + np.random.random()*(r1.vector-vector) + F*(r2.vector-r3.vector)
        mask_pos = np.random.random(vector.shape) <= CR
        vector[mask_pos] = v[mask_pos]
    return vector


def DECurrentToBest1(vector, population, F, CR):
    """
    Performs the differential evolution operator DE/current-to-best/1
    """
   
    if len(population) > 3:
        fitness = [i.fitness for i in population]
        best = population[fitness.index(max(fitness))]
        r1, r2 = random.sample(population, 2)

        v = vector + F*(best.vector-vector) + F*(r1.vector-r2.vector)
        mask_pos = np.random.random(vector.shape) <= CR
        vector[mask_pos] = v[mask_pos]
    return vector


def DECurrentToPBest1(vector, population, F, CR, P):
    """
    Performs the differential evolution operator DE/current-to-pbest/1
    """
   
    if len(population) > 3:
        fitness = [i.fitness for i in population]
        upper_idx = max(1, math.ceil(len(population)*P))
        pbest_idx = random.choice(np.argsort(fitness)[:upper_idx])
        pbest = population[pbest_idx]
        r1, r2 = random.sample(population, 2)

        v = vector + F*(pbest.vector-vector) + F*(r1.vector-r2.vector)
        mask_pos = np.random.random(vector.shape) <= CR
        vector[mask_pos] = v[mask_pos]
    return vector


def pso_operator(indiv, population, global_best, w, c1, c2):
    """
    Performs a step of the Particle Swarm algorithm
    """

    c1 = c1 * np.random.random(indiv.vector.shape) 
    c2 = c2 * np.random.random(indiv.vector.shape) 

    indiv.speed = w * indiv.speed + c1 * (indiv.best - indiv.vector) + c2 * (global_best.vector - indiv.vector)
    return indiv.apply_speed()


def firefly(solution, population, objfunc, alpha_0, beta_0, delta, gamma):
    """
    Performs a step of the Firefly algorithm
    """
   
    sol_range = objfunc.sup_lim - objfunc.inf_lim
    n_dim = solution.vector.size
    new_vector = solution.vector.copy()
    for idx, ind in enumerate(population):
        if solution.fitness < ind.fitness:
            r = np.linalg.norm(solution.vector - ind.vector)
            alpha = alpha_0 * delta ** idx
            beta = beta_0 * np.exp(-gamma*(r/(sol_range*np.sqrt(n_dim)))**2)
            new_vector = new_solution + beta*(ind.vector-new_vector) + alpha * sol_range * random.random()-0.5
            new_vector = objfunc.check_bounds(new_vector)
    
    return new_solution


def dummyOp(vector, scale=1000):
    """
    Replaces the vector with one consisting of all the same value
    
    Only for testing, not useful for real applications
    """

    return np.ones(vector.shape)*scale