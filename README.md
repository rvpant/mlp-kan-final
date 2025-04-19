This reop contains code for experiments conducted in [MLPs and KANs for data-driven learning in physical problems: A performance comparison](https://arxiv.org/abs/2504.11397).

### Environment Setup
Both a conda ```environment.yml``` file and a pip ```requirements.txt``` file are provided here for Python environment setup. To create the environment, it suffices to use the basic setup options: ```conda env create -f environment.yml``` or ```pip install -r requirements.txt``` once inside of a Python virtualenv.

### File Overview
DeepONet.py, darcy_1d_deeponet.py, darcy_2d_deeponet.py, and elastic_deeponet.py contain the codes used for model training and analysis. Each accepts a number of command line arguments for running: the most important of which are ```--mode``` to indicate a shallow or deep model, and ```--modeltype``` to indicate whether MLPs or KANs are being used. Please refer to each file to better understand the arguments required for each script.

The remainder of the python files are utility scripts used to load saved models and create plots, analyze prediction loss, etc.

In darcy.py, we provide a series of solver including a finite difference and FEM solver to help create a reference training data set for the 1D Darcy problem.