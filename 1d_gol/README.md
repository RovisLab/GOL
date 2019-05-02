# One-dimensional GOL

## Description
The one-dimensional GOL showcases the Pareto optimization performed on a one-dimensional learning problem. The one-shot objects are three values, and they are accompanied by 10 regularization samples. The generalization generator is composed of a single function, which generates normal distributed synthetic samples having a mean equal to the one-shot objects and a standard deviation controlled by the generalization function. The objective is to find Pareto optimal solutions, which maximize generalization energies and classification accuracy. 

## Prerequisites
- Python 3.x
- numpy
- scipy
- platypus-opt
- tensorflow
- matplotlib
Please install the required packages using pip install requirements.txt

## Running the one-dimensional GOL algorithm
To run the algorithm, you can call one of the scripts: e.g. python gol_v15_platypus.py.
