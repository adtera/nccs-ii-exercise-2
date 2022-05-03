#! usr/bin/env python
# Imports n stuff
import numpy as np
from scipy import constants, mean
import sys
import math
import matplotlib.pyplot as plt

# Read energies 
with open('energies.txt') as f:
    text = f.readlines()
energies_str = [tuple(line.split()) for line in text[1:]]
energies_float = [(float(epot),float(ekin)) for ekin,epot in energies_str]
energies_float

# Check Conversion of Ekin + Epot 
for epot,ekin in energies_float:
    print(f"EPot = {epot}, EKin = {ekin}, Total Energie = {ekin+epot}")

# Using only the second half of the trajectory, estimate the specific heat of the system
t_half = len(energies_float)//2
energies_t_half = energies_float[t_half:]
ekins_t_half = [e[1] for e in energies_t_half]

# Calculate mean kinetic energy over second half of trajectory
ekin_mean = np.mean(ekins_t_half)
squared_varianz = np.mean((ekins_t_half-ekin_mean)**2)
specific_heat = constants.k*(ekin_mean**2)/squared_varianz
print(f"Specific Heat of the system: {specific_heat}")