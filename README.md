# Photonics Project

This repository includes the python files that were used to produce the results of the project.
For this project HFSS, Tidy3D and the online simulator were used to simulate a silicon waveguide of dimensions wxh = 400 nm x 300 nm. The structure consists of a silicon dioxide substrate and air used for cladding.

Explanation of the directories, files and code listed below:
File or directory name | Description
--- | ---
main.py | The main directory containing the implementation of the exercises in python.
config.py | The python configuration directory which contains a configuration dictionary as well as several utilization functions.
Gamma.csv | Produced by HFSS, contains the phase constant of the structure against frequency for each mode.
si.csv | Contains the effective index of silicon against wavelength produced by HFSS.
sio2.csv | Contains the effective index of silicon dioxide against wavelength produced by HFSS.
TE_E.csv | Contains the 2D profile of the electric field in V/m for each frequency for the TE mode.
TM_E.csv | Contains the 2D profile of the electric field in V/m for each frequency for the TM mode.
online_ex1 | Directory containing all the plots from the online simulator tool for exercise 1.
HFSS_ex1 | Directory containing all the plots from HFSS for exercise 1.
all_plots_online | Directory containing all the plots from the online simulator which are used to calculate the effective area of exercise 4.

