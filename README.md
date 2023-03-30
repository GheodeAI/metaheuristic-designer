# PyEvolComp
Python implementation of a general framework for the design and execution of metaheuristic algorithms.

Created with the purpose of creating the necesary tools for the analysis and design of metaheuristics
in a more granular way, choosing algorithm components and adding them to a general search framework.

Inspired by this article: 
    Swan, Jerry, et al. "Metaheuristics “in the large”." European Journal of Operational Research 297.2 (2022): 393-406.
   
Mostly following the book: 
    Eiben, Agoston E., and James E. Smith. Introduction to evolutionary computing. Springer-Verlag Berlin Heidelberg, 2015.


To configure the hyperparameters a dictionary will have to be given to the class of each algorithm.

## General parameters
- Stopping conditions:
    - stop_cond: stopping condition, there are various options
        - "neval": stop after a given number of evaluations of the fitness function
        - "ngen": stop after a given number of generations
        - "time": stop after a fixed amount of execution time (real time, not CPU time)
        - "fit_target": stop after reaching a desired fitness, accounting for whether we have maximization and minimization
        - All of the above can be combined with the logical operators 'or' or 'and' to make more complex stopping conditions,
        an example can be "ngen or time" which will stop when the number of generations or the time limit is reached.
    - Neval: number of evaluations of the fitness function
    - Ngen: number of generations
    - time_limit: execution time limit given in seconds
    - fit_target: value of the fitness function we want to reach
- Display options:
    - verbose: shows a report of the state of the algorithm periodicaly
    - v_timer: amount of time between each report

## Operators available
- 1 point cross (1point)
- 2 point cross (2point)
- Multipoint cross (Multipoint)
- Weighted average cross (WeightedAvg)
- BLXalpha cross (BLXalpha)
- SBX cross (SBX)
- Multi-individual cross (Multicross)
- Xor with random vector (Xor)
- Cross two vectors with Xor (XorCross)
- Permutation of vector components (Perm)
- Random mutation of vector components (MutRand/MutNoise)
- Add noise following a prob. distribution (RandNoise)
- Approximate population with a prob. distribution and sample on some vector components (MutSample)
- Approximate population with a prob. distribution and sample (RandSample)
- Mutation adding Gaussian noise (Gauss)
- Mutation adding Cauchy noise (Cauchy)
- Mutation adding Laplace noise (Laplace)
- Mutation adding Uniform noise (Uniform)
- Differential evolution operators (DE/best/1, DE/best/2, DE/rand/1, DE/rand/2, DE/current-to-rand/1, DE/current-to-best/1, DE/current-to-pbest/1)
- Particle swarm algorithm step (PSO)
- Firefly algorithm step (Firefly)
- Placing fixed vector into the population (Dummy) [only intended for debugging]
- Return a copy of the individual (Nothing) 

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




