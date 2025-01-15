# PyQuake3D
![Uploading logo.png…]()

PyQuake3D is a Python-based Boundary Element Method (BEM) code for simulating sequences of seismic and aseismic slip (SEAS) on a complex 3D fault geometry governed by rate- and state-dependent friction. This document provides an overview of how to use the script, as well as a detailed description of the input parameters.

# Contribution
2024.7.1  Dr.Rongjiang Tang developed the code framework and the Quasi-dynamic BIEM Seismic cycle model on a complex 3D fault geometry governed by regularized aging law.
2024.10.5 Dr.Rongjiang Tang and Dr.Luca Dal Zilio implemented the H-matrix Matrix-Vector Multiplication and developed the Cascadia model.


## Running the Script

To run the PyQuake3D script, use the following command:
```bash
python -g --inputgeo <input_geometry_file> -p --inputpara <input_parameter_file>
```
For example:
```
To execute benchmarks like BP5-QD, use:
```bash
python src/main.py -g examples/cascadia/cascadia35km_ele4.msh -p examples/cascadia/parameter.txt
```
Ensure you modify the input parameter (`parameter.txt`) as follows:
- `Corefunc directory`: `bp5t_core`
- `InputHetoparamter`: `True`
- `Inputparamter file`: `bp5tparam.dat`

- 
requirement of python library
python>=3.8
numpy>=1.2
ctypes==1.1
cupy=10.6.0

Please refer to doc.pdf for more details.
