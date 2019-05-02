# Autonomous Vision GOL

Autonomous Vision GOL is a Python based framework used for running a One-Shot Learning type of algorithm, which takes as input a so-called one-shot (template) object, plus a small set of regularization samples and aims at generating artificial training data for deep learning algorithms as Pareto optimal solutions.

## Algorithm description
- A template object, together with some regularization samples are selected;
- The Pareto optimization is performed, and output artificial samples are created automatically;
- The solution is analyzed - a classifier is trained with the generated samples and evaluated;
- The Pareto fronts are plotted inside the user interface;
- The obtained classifier model is saved.

## Prerequisites
- Python 3.x
- pyyaml
- Pillow
- opencv-python
- scikit-image
- platypus
- platypus-opt
- matplotlib
- pygmo
- keras
- tensorflow

Please install the required packages using pip install requirements.txt


## Running the Autonomous Vision GOL algorithm
To run the algorithm, you can call python environment.py
