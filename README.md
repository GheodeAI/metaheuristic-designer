# Metaheuristic-designer
[![Documentation Status](https://readthedocs.org/projects/metaheuristic-designer/badge/?version=latest)](https://metaheuristic-designer.readthedocs.io/en/latest/?badge=latest)

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
    - "examples/exec_basic.py": Evolve an image so that it matches the one given as an input. 
        - The same parameters as the previous script.
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

### Search strategies
The algorithms implemented are:
| Class name | Strategy | Params | Other info |
|------------|-----------|--------|------------|
|NoSearch|Do nothing||For debugging purposes||
|RandomSearch|Random Search|||
|HillClimb|Hill climb |||
|LocalSearch|Local seach |**iters** (number of neighbors to test each time)||
|SA|Simulated annealing|**iter** (iterations per temperature change), **temp_init** (initial temperature), **alpha** (exponent of the temperature change) ||
|GA|Genetic algorithm|**pmut** (probability of mutation), **pcross** (probability of crossover)||
|ES|Evolution strategy|**offspringSize** (number of indiviuals to generate each generation)||
|HS|Harmony search|**HSM**, **HMCR**, **BW**, **PAR**||
|PSO|Particle Swarm optimization|**w**,**c1**,**c2**||
|DE|Differential evolution|||
|CRO|Coral Reef Optimization|**rho**,**Fb**,**Fd**,**Pd**,**attempts**||
|CRO_SL|Coral Reef Optimization with substrate layers|**rho**,**Fb**,**Fd**,**Pd**,**attempts**||
|PCRO_SL|probabilistic Coral Reef Optimization with substrate layers|**rho**,**Fb**,**Fd**,**Pd**,**attempts**||
|DPCRO_SL|Dynamic probabilistic Coral Reef Optimization with substrate layers|**rho**,**Fb**,**Fd**,**Pd**,**attempts**,**group_subs**,**dyn_method**,**dyn_steps**,**prob_amp**||
|VND| Variable neighborhood descent|||
|RVNS| Restricted variable neighborhood search||In progress|
|VNS| Variable neighborhood search||In progress|
|CMA_ES| Covariance matrix adaptation - Evolution strategy|| Not implemented yet|

### Survivor selection methods
These are methods of selecting the individuals to use in future generations.

The methods implemented are:
| Method name | Algorithm | Params | Other info |
|-------------|-----------|--------|------------|
|"Elitism"|Elitism|**amount**||
|"CondElitism"|Conditional Elitism|**amount**||
|"nothing" or "generational"|Replace all the parents with their children|| Needs the offspring size to be equal to the population size|
|"One-to-one" or "HillClimb"|One to one (compare each parent with its child)||Needs the offspring size to be equal to the population size|
|"Prob-one-to-one" or "ProbHillClimb"|Probabilitisc One to one (with a chance to always choose the child)|**p**|Needs the offspring size to be equal to the population size|
|"(m+n)" or "keepbest"|(λ+μ), or choosing the λ best individuals taking parents and children|||
|"(m,n)" or "keepoffspring"|(λ,μ), or taking the best λ children||λ must be smaller than μ|
|"CRO"|A 2 step survivor selection method used in the CRO algorithm. Each individual attempts to enter the population K times and then a percentage of the worse individuals will be eliminated from the population|**Fd**,**Pd**,**attempts**,**maxPopSize**|Can return a population with a variable number of individuals|

### Parent selection methods
These are methods of selecting the individuals that will be mutated/perturbed in each generation

The methods implemented are:
| Method name | Algorithm | Params | Other info |
|-------------|-----------|--------|------------|
|"Torunament"|Choose parents by tournament|**amount**, **p**||
|"Best"| Select the n best individuals|**amount**||
|"Random"| Take n individuals at random|**amount**||
|"Roulette"| Perform a selection with the roullette method|**amount**, **method**, **F**||
|"SUS"| Stochastic universal sampling|**amount**, **method**, **F**||
|"Nothing"| Take all the individuals from the population||

### Operators

| Class name | Domain | Other info|
|------------|--------|----|
|OperatorReal|Real valued vectors||
|OperatorInt|Integer valued vectors||
|OperatorBinary|Binary vectors||
|OperatorPerm|Permutations||
|OperatorList|Variable length lists||
|OperatorMeta|Other operators||
|OperatorLambda|Any|Lets you specify a function as an operator|

The Operators functions available in the operator classes are:
| Method name | Algorithm | Params | Domains |
|-------------|-----------|--------|---------|
|"1point"|1 point crossover||Real, Int, Bin|
|"2point"|2 point crossover||Real, Int, Bin|
|"Multipoint"|multipoint crossover||Real, Int, Bin|
|"WeightedAvg"|Weighted average crossover|**F**|Real, Int|
|"BLXalpha"|BLX-alpha crossover|**Cr**|Real|
|"Multicross"|multi-individual multipoint crossover|**Nindiv**|Real, Int, Bin|
|"XOR"|Bytewise XOR with a random vector|**N**|Int|
|"XORCross"|Bytewise XOR between 2 vectors component by component||Int|
|"sbx"|SBX crossover|**Cr**|Real|
|"Perm"|Permutate vector components|**N**|Real, Int, Bin, Perm|
|"Gauss"|Add Gaussian noise|**F**|Real, Int|
|"Laplace"|Add noise following a Laplace distribution|**F**|Real, Int|
|"Cauchy"|Add noise following a Cauchy distribution|**F**|Real, Int|
|"Poisson"|Add noise following a Cauchy distribution|**F**|Int|
|"Uniform"|Add Uniform noise|**Low**, **Up**|Real, Int|
|"MutRand" or "MutNoise"|Add random noise to a number of vector components|**method**, **N**, optionaly: **Low**, **Up**, **F**|Real, Int|
|"MutSample"|Take a sample from a probability distribution and put it on a number of vector components|**method**, **N**, optionaly: **Low**, **Up**, **F**|Real, Int|
|"RandNoise"|Add random noise|**method**, optionaly: **Low**, **Up**, **F**|Real, Int|
|"RandSample"|Sample from a probability distribution|**method**, optionaly: **Low**, **Up**, **F**|Real, Int|
|"DE/Rand/1"|Sample from a probability distribution|**F**, **Cr**|Real, Int|
|"DE/Best/1"|Sample from a probability distribution|**F**, **Cr**|Real, Int|
|"DE/Rand/2"|Sample from a probability distribution|**F**, **Cr**|Real, Int|
|"DE/Best/2"|Sample from a probability distribution|**F**, **Cr**|Real, Int|
|"DE/Current-to-rand/1"|Sample from a probability distribution|**F**, **Cr**|Real, Int|
|"DE/Current-to-best/1"|Sample from a probability distribution|**F**, **Cr**|Real, Int|
|"DE/Current-to-pbest/1"|Sample from a probability distribution|**F**, **Cr**, **p**|Real, Int|
|"PSO"|Sample from a probability distribution|**w**, **c1**, **c2**|Real, Int|
|"Firefly"|Sample from a probability distribution|**a**,**b**,**c**,**g**|Real, Int|
|"Random"|Sample from a probability distribution||Real, Int, Bin, Perm|
|"RandomMask"|Randomly sample a number of vector components|**N**|Real, Int||
|"Swap"|Swap two components||Perm||
|"Insert"|Insert a component and shift to the left||Perm||
|"Scramble"|Scramble permutation order|**N**|Perm||
|"Invert"|Reverse order of components||Perm||
|"Roll"|Roll components to the right|**N**|Perm||
|"PMX"|Partially mapped crossover||Perm||
|"OrderCross"|Ordered crossover||Perm||
|"branch"|Choose one of the provided operators randomly||Operators|
|"sequence"|Apply all the provided operators in order||Operators|
|"split"|Apply each operator to a subset of vector components following the mask provided||Operators|
|"pick"|Manually pick one of the operators provided (setting the ```chosen_idx``` attribute)||Operators|
|"Dummy"|Assing the vector to a predefined value||All|
|"Custom"|Provide a lambda function to apply as an operator|**function**|All|
|"Nothing"|Do nothing||All|

### Initializers
Initializers create the initial population that will be evolved in the optimization process.

Some of the implemente Initializers are:
| Class name | Description | Other info |
|------------|-----------|------------|
|DirectInitializer|Initialize the population to a preset list of individuals||
|SeedProbInitializer|Initializes the population with another initializer and inserts user-specified individuals with a probability||
|SeedDetermInitializer|Initializes the population with another initializer and inserts a number of user-specified individuals into the population||
|GaussianVectorInitializer|Initialize individuals with normally distributed vectors||
|UniformVectorInitializer|Initialize individuals with uniformly random distributed vectors||
|PermInitializer|Initialize individuals with random permuations||
|LambdaInitializer|Initialize individuals with a user-defined function||


### Encodings
Specifying the Encoding is optional but can be very helpful for some types of problems.

An encoding will represent each solution differently in the optimization process and the evaluation of the fintess, since most algorithm work only with vectors, but we might need other types of datatypes for our optimization.

Some of the implemented Encodings are:
| Class name | Encoding | Decoding | Other info |
|------------|----------|----------|------------|
|DefaultEncoding|Makes no changes to the input|Makes no changes to the input|
|TypeCastEncoding|Changes the datatype of the vector from **T1** to **T2**|Changes the datatype of the vectorfrom **T1** to **T2**||
|MatrixEncoding|Converts a vector into a matrix of size **NxM**|Converts a matrix to a vector with the ```.flatten()``` method||
|ImageEncoding|Converts a vector into a matrix of size **NxMx1** or **NxMx3**, each component is an unsigned 8bit number|Converts a matrix to a vector with the ```.flatten()``` method|
|LambdaEncoding|Applies the user-defined ```encode``` function|Applies the user-defined ```decode``` function||


### Benchmark functions
The benchmark functions you can use to test the algorithms are:
| Class name | Domain | Other info |
|------------|--------|------------|
|MaxOnes||Integer||
|DiophantineEq||Integer||
|MaxOnesReal||Real||
|Sphere||Real||
|HighCondElliptic||Real||
|BentCigar||Real||
|Discus||Real||
|Rosenbrock||Real||
|Ackley||Real||
|Weistrass||Real||
|Griewank||Real||
|Rastrigin||Real||
|ModSchwefel||Real||
|Katsuura||Real||
|HappyCat||Real||
|HGBat||Real||
|SumPowell||Real||
|N4XinSheYang||Real||
|ThreeSAT||Real||
|BinKnapsack||Binary||
|MaxClique||Permutation||
|TSP||Permutation||
|ImgApprox||Integer||
|ImgStd||Integer||
|ImgEntropy||Integer||

