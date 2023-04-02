# PyEvolComp
Python implementation of a general framework for the design and execution of metaheuristic algorithms.

Created with the purpose of creating the necesary tools for the analysis and design of metaheuristics
in a more granular way, choosing algorithm components and adding them to a general search framework.

Inspired by this article: 
    Swan, Jerry, et al. "Metaheuristics “in the large”." European Journal of Operational Research 297.2 (2022): 393-406.
   
Mostly following the book: 
    Eiben, Agoston E., and James E. Smith. Introduction to evolutionary computing. Springer-Verlag Berlin Heidelberg, 2015.

## Project structure
This project uses poetry as a package manager and nox for unit testing.


## Examples
- There are 2 scripts to test this repository:
    - "examples/exec_basic.py": Optimize a simple function, in this case, the "sphere" function that calculates the squared norm of a vector, we want a vector that minmizes this function. There are two possible flags that can be added:
        - "-a \[Alg\]" use one of the available algorithms, the choices are:
            - HillClimb: simple hill climbing algorithm.
            - LocalSearch: take the best of 20 randomly chosen neighbours.
            - ES: (100+150)-ES, basic evolutionary strategy.
            - HS: Harmony search algorithm.
            - GA: genetic algorithm.
            - SA: simulated annealing algorithm.
            - DE: DE/best/1, differential evolution algorithm.
            - PSO: simple particle swarm algorithm.
            - NoSearch: no search is done.
        - "-m" use a memetic search like structure, do local search after mutation.
    - "examples/exec_basic.py": Evolve an image so that it matches the one given as an input. Recieves mostly the same parameters except for one for indicating the input image:
        - "-i \[Image path\]" read the image and evolve a random image into this one.

To execute the scripts with the correct dependencies first run

```bash
poetry install
```

Then you can run one the scripts as:

```bash
poetry run python examples/image_evolution.py -a SA -i images/saturn.png
```

## General parameters
- Stopping conditions:
    - stop_cond: stopping condition, there are various options
        - "neval": stop after a given number of evaluations of the fitness function
        - "ngen": stop after a given number of generations
        - "time": stop after a fixed amount of execution time (real time, not CPU time)
        - "fit_target": stop after reaching a desired fitness, accounting for whether we have maximization and minimization
        - All of the above can be combined with the logical 'or' or 'and' to make more complex stopping conditions,
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




