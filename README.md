# RayleighBenard-Dataset
This project includes tools for working with Rayleigh-Benard Convection data.

## Install
### conda
Install using the predefined conda environmnet files for your operating system:  
```conda env create -f conda/<OS>.yaml```
### pip
Clone project and install from local source:  
```pip install ./```

## Data Download
### Private
via dvc

### Public
Data ist publicly available [here](https://uni-bielefeld.sciebo.de/s/eh12csvif6Imvfz).

## Usage
After installing you can use following tools:

### Viewing RBC data
To view rbc data execute the `rbcview` script and give a path to a rbc episode file:

```rbcview +path=<PATH>``` 

Type ```rbcgview --help``` to learn about the hydra configuration.