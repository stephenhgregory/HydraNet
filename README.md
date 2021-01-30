# HydraNet
A multi-branch Neural Network architecture for MRI Denoising

## Implementation
The [keras_implementation](./keras_implementation) folder contains training and testing scripts for HydraNet.

## Dependencies
The [environment.yaml](environment.yaml) file contains all of the packages needed to use HydraNet.

This file is used to create a Conda environment named **MRIDenoising**. 
1. First, create the environment by executing the following command from a terminal located at *.../HydraNet/*:
```
conda env create -f dependencies/environment.yaml
```
2. Then , activate the conda environment with the following command:
```
conda activate MRIDenoisingUpgradedTF
```
