# Conditions for Instability in the Coupled Climate-Carbon System

## Running the Code
To run the code, you first must install the correct conda environment.
To do this run
```
conda env create -f climate_carbon_instability.yml
conda activate climate_carbon_instability
```

The figures are then produced by running
```
make
```

The figures will will be output into a directory called figures.

## File Structure
Python scripts used to produce figures in the paper are
found in the top level directory. 

The file ```climate_carbon_instability.yml```
is used to install the conda environment. The file ```environment.yml``` is
a record of the actual environment used to produce the figures and was 
produced with the command ```conda env export > environment.yml```. 

Figures can be found in the figures directory and data (e.g. JULES output) can
be found in the data directory.

For convience a makefile is provided. Use ```make``` to produce the figures and
```make clean``` to delete them.
