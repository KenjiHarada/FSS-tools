# FSS-tools
The finite-size scaling (FSS) method is a powerful tool for getting universal information of critical phenomena. It estimates universal information from observables of critical phenomena at finite-size systems.

Here, we introduce two FSS methods by using Gaussian process (GP) and a neural network (NN).

#### prerequisites
The module [PyTorch](https://pytorch.org "PyTorch Home") and [GPyTorch](https://gpytorch.ai "GPyTorch Home") are required.

#### examples
There are two documents for FSS methods by GP and NN, respectively.
 - [FSS method by GP](examples/bsa.ipynb "Jupyter notebook")
 - [FSS method by NN](examples/nsa.ipynb "Jupyter notebook")

#### related sites
 - [FSS method by GP with C++](https://kenjiharada.github.io/BSA/ "BSA Site")
 - [FSS method by NN with Python (JAX/Flex)](https://github.com/yonesuke/jaxfss "Jaxfss Site")

#### references
1. Kenji Harada: Bayesian inference in the scaling analysis of critical phenomena, Physical Review E 84 (2011) 056704. 
DOI: [10.1103/PhysRevE.84.056704](https://hdl.handle.net/10.1103/PhysRevE.84.056704 "FSS by GP")
1. Kenji Harada: Kernel method for corrections to scaling, Physical Review E 92 (2015) 012106.
DOI: [10.1103/PhysRevE.92.012106](https://hdl.handle.net/10.1103/PhysRevE.92.012106 "FSS by GP")
1. Ryosuke Yoneda and Kenji Harada : (preparation).

#### history
 - March 14, 2022: The first release (v0.1.0)
