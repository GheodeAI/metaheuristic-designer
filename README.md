# PyEvolAlg
Python implementation of several metaheuristics/optimization algorithms

Each algorithm is located in its package directory.

To configure the hyperparameters a dictionary will have to be given to the class of each algorithm.

## General parameters
- Stopping conditions:
    - stop_cond: stopping condition, there are various options
        - "neval": stop after a given number of evaluations of the fitness function
        - "ngen": stop after a given number of generations
        - "time": stop after a fixed amount of execution time (real time, not CPU time)
        - "fit_target": stop after reaching a desired fitness, accounting for whether we have maximization and minimization
    - Neval: number of evaluations of the fitness function
    - Ngen: number of generations
    - time_limit: execution time limit given in seconds
    - fit_target: value of the fitness function we want to reach
- Display options:
    - verbose: shows a report of the state of the algorithm periodicaly
    - v_timer: amount of time between each report

## Algorithms available
- Coral Reef optimization with subtrate layers (CRO_SL)
- Genetic algorithm (GA)
- Evolution Strategy (ES)
- Harmony search (HS)
- LSHADE
- Particle Swarm Optimization (PSO)
- Simulated annealing (SimAnn)

## Operators available
- 1 point cross (1point)
- 2 point cross (2point)
- multipoint cross (Multipoint)
- BLXalpha cross (BLXalpha)
- SBX cross (SBX)
- Multi-individual cross (Multicross)
- Permutation of vector components (Perm)
- Xor with random vector (Xor)
- Cross two vectors with Xor (XorCross)
- Random mutation of vector components (MutRand)
- mutation adding Gaussian noise (Gauss)
- mutation adding Cauchy noise (Cauchy)
- mutation adding Laplace noise (Laplace)
- mutation adding Uniform noise (Uniform)
- Differential evolution operators (DE/best/1, DE/best/2, DE/rand/1, DE/rand/2, DE/current-to-rand/1, DE/current-to-best/1, DE/current-to-pbest/1)
- LSHADE operator (LSHADE)
- reduced Simulated annealing (SA)
- Harmony search operator (HS)
- Approximate population with a prob. distribution and sample (RandSample)
- Approximate population with a prob. distribution and sample on some vector components (MutSample)
- Placing fixed vector into the population (Dummy) [only intended for debugging]

## Parent selection methods available
- Tournament (Tournament)
- Taking the n best parents (Best)
- Take all parents (Nothing)

## Survivor selection methods avaliable
- Elitism (Elitism)
- Conditional Elitism (CondElitism)
- Generational (Generational)
- (λ+μ) like method ( (m+n) )
- (λ,μ) like method ( (m,n) )




