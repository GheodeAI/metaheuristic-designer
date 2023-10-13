# Metaheuristic-designer
This is an object-oriented framework for the development, testing and analysis of metaheuristic optimization algorithms.

It defines the components of a general evolutionary algorithm and offers some implementations of algorithms along with components
that can be used directly. Those components will be explained below.

It was inspired by the article [Metaheuristics “in the large”](https://doi.org/10.1016/j.ejor.2021.05.042) that 
discusses some of the issues in the research on metaheuristic optimization, sugesting the development of libraries for the standarization
of metaheuristic algorithms.

Most of the design decisions are based on the book [Introduction to evolutionary computing by Eiben, Agoston E.,
and James E. Smith](https://doi.org/10.1007/978-3-662-44874-8) which is very well expained and is highly recomended to anyone willing to learn about the topic.

This framework doesn't claim to have a high performance, specially since the chosen language is Python and the code has not been 
designed for speed. This shouldn't really be an issue since the highest amount of time spent in these kind of algorithms
tends to be in the evaluation of the objective function. If you want to compare an algorithm made with this tool with another
one that is available by other means, it is recomended to use the number of evaluations of the objective function as a metric instead of execution time.

## Instalation

The package is available in the PyPi repository (https://pypi.org/project/metaheuristic-designer/).

To install it, use the pip command as follows:

```bash
pip install metaheuristic-designer
```

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

It is recomended that you create a virtual environment to test the examples.

This is done with the following commands:
```bash
python -m venv venv
source venv/bin/activate
pip install .[examples]
```

Once you have activate the virtual environment, you can execute one of the examples like this:
```
python examples/example_basic.py
``` 

or

```
python examples/image_evolution.py
``` 

To run the tests you need to install nox, to execute the tests use the command

```bash
nox test
```

## Implemented components
This package comes with some already made components that can be used in any algorithm in this framework

### Algorithms
The algorithms implemented are:
| Class name | Algorithm |Other info|
|------------|-----------|----------|
|RandomSearch| Random Search||
|HillClimb|Hill climb ||
|LocalSearch|Local seach ||
|SA|Simulated annealing||
|GA|Genetic algorithm||
|ES|Evolution strategy||
|HS|Harmony search||
|PSO|Particle Swarm optimization||
|DE|Differential evolution||
|CRO|Coral Reef Optimization||
|CRO_SL|Coral Reef Optimization with substrate layers||
|PCRO_SL|probabilistic Coral Reef Optimization with substrate layers|
|DPCRO_SL|Dynamic probabilistic Coral Reef Optimization with substrate layers|
|VND| Variable neighborhood descent||
|RVNS| Restricted variable neighborhood search|In progress|
|VNS| Variable neighborhood search|In progress|
|CMA_ES| Covariance matrix adaptation - Evolution strategy| Not implemented|

### Survivor selection methods
These are methods of selecting the individuals to use in future generations.

The methods implemented are:
| Method name | Algorithm |Other info|
|-------------|-----------|----------|
|"Elitism"|Elitism|
|"CondElitism"|Conditional Elitism|
|"nothing" or "generational"|Replace all the parents with their children| Needs the offspring size to be equal to the population size|
|"One-to-one" or "HillClimb"|One to one (compare each parent with its child)|Needs the offspring size to be equal to the population size|
|"Prob-one-to-one" or "ProbHillClimb"|Probabilitisc One to one (with a chance to always choose the child)|Needs the offspring size to be equal to the population size|
|"(m+n)" or "keepbest"|(λ+μ), or choosing the λ best individuals taking parents and children||
|"(m,n)" or "keepoffspring"|(λ,μ), or taking the best λ children|λ must be smaller than μ|
|"CRO"|A 2 step survivor selection method used in the CRO algorithm. Each individual attempts to enter the population K times and then a percentage of the worse individuals will be eliminated from the population|Can return a population with a variable number of individuals|

### Parent selection methods
These are methods of selecting the individuals that will be mutated/perturbed in each generation

The methods implemented are:
| Method name | Algorithm |Other info|
|-------------|-----------|----------|
|"Torunament"||
|"Best"||
|"Random"||
|"Roulette"||
|"SUS"||
|"Nothing"||

### Operators
| Class name | Domain |
|------------|--------|
|OperatorReal|Real valued vectors|
|OperatorInt|Integer valued vectors|
|OperatorBinary|Binary vectors|
|OperatorPerm|Permutations|
|OperatorList|Variable length lists|
|OperatorMeta|Other operators|

Additionaly there is a OperatorLambda that applies a user-defined function as the operator.


| Method name | Algorithm | Domains |
|-------------|-----------|---------|
|"1point"| 1 point crossover|Real, Int, Bin|
|"nothing"|||
|"branch"||Operators(Meta)|
|"sequence"||Operators(Meta)|
|"split"||Operators(Meta)|
|"pick"||Operators(Meta)|

### Initializers

| Class name | Algorithm | Other info |
|------------|-----------|------------|

### Encodings
| Class name | Algorithm | Other info |
|------------|-----------|------------|

### Benchmark functions

| Class name | Function | Domain | Other info |
|------------|----------|--------|------------|

