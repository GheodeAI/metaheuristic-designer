# HS
Python implementation of the Harmony Search algorithm

To configure the hyperparameters a dictionaty will have to be given to the class HS.

## Hyperparameters
The hyperparameters will be the following:
- Algorithm's hyperparameters:
    - PopSize: equivalent to HMS, the size of the Harmony. The size of solutions.
    - HMCR: Harmony Memory Considering Rate. Percentage of elements that we subtract of the HM (matrix of solutions).
    - PAR: Pitch Adjusting Rate. Proportion of elements subtracted from HM that we mutate
    - BN: standard deviation of mutation

## Operators
Also we need to declare the mutation and replace Operator. It sould have the following structure:
    mutation: Operator("[name]":{"F":[standard_deviation]})
    replace: Operator("Replace":{"F":[standard_deviation], "method": [method_name]})