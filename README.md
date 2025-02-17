# PyQuake3D


PyQuake3D is a Python-based Boundary Element Method (BEM) code for simulating sequences of seismic and aseismic slip (SEAS) on a complex 3D fault geometry governed by rate- and state-dependent friction. This document provides an overview of how to use the script, as well as a detailed description of the input parameters.

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
