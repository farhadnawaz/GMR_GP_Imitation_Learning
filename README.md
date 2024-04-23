# Original project

This code base is a copy of the original GMRbasedGP work https://sites.google.com/view/gmr-based-gp?pli=1, with minor changes made for comparison with our work https://github.com/farhadnawaz/CLF-CBF-NODE.

# Instruction
```python3 -m venv env```  
```source env/bin/activate```  
```pip3 install -r requirements.txt```  

### 2D
```
python3 GMR_based_GPR01.py --letter {I|R|O|S} --n_gaussian 20 --n_test_execution 10
```

### 3D
```
python3 GMR_based_GPR3D.py --data_name {Wiping_loop|Spiral|Towel} --n_gaussian 20 --n_test_execution 10
```

# Below is the original readme
# GMR-based Gaussian Process
This folder contains the different toy examples presented in the paper "Learning from Demonstration with model-based Gaussian Processes" (N. Jaquier, D. Ginsbourger and S. Calinon), CoRL 2019.


## Installation
These examples work with Python 2 and 3. First install the following packages:

pip install numpy

pip install matplotlib

pip install gpy  (or https://github.com/SheffieldML/GPy)

The figures generated by the examples will be saved in the figures folder.

## Description of the examples

** GMR01 **

Example of GMR for 2-D outputs with time as input. Corresponds to Figure 1a of the main paper.

** GPR_coregionalization01 **

Example of multi-output GP for 2-D output with time as input. Corresponds to Figure 2b of the main paper.

** GMR_based_GPR01 **

Example of the GMR-based GP for 2-D outputs with time as input. Corresponds to Figure 2 of the main paper.

**GMR_based_GPR_uncertainty_examples01 **

Illustration of GMR-based GPR properties with 1-dimensional input and output. Corresponds to Figure 3 of the main paper.
The user can change the number of observations, the lengthscale parameter and the noise variance in the file to generate different examples.
