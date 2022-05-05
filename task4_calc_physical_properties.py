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
energies_float = [(float(epot), float(ekin)) for ekin, epot in energies_str]
energies_float

# Check Conversion of Ekin + Epot
ekins = []
epots = []
etotals = []
for epot, ekin in energies_float:
    print(f"EPot = {epot}, EKin = {ekin}, Total Energie = {ekin+epot}")
    epots.append(epot)
    ekins.append(ekin)
    etotals.append(ekin + epot)

# Plot conversion of Ekin and Epot
fig = plt.figure()
fig.suptitle("Evolution of Epot, Ekin and Total Energy")
plt.xlabel("Time/fs")
plt.ylabel("h^2/m_e*a_o^2")
plt.plot([epots, ekins, etotals])
plt.legend(['Epot', 'Ekin', 'Etotal'])
plt.show()

# Using only the second half of the trajectory, estimate the specific heat of the system
t_half = len(energies_float) // 2
energies_t_half = energies_float[t_half:]
ekins_t_half = [e[1] for e in energies_t_half]

# Calculate mean kinetic energy over second half of trajectory
ekin_mean = np.mean(ekins_t_half)
squared_varianz = np.mean((ekins_t_half - ekin_mean)**2)
specific_heat = constants.k * (ekin_mean**2) / squared_varianz
print(f"Specific Heat of the system: {specific_heat} E_H/K")

# Read ri | fractions of interatomic distances within ri averaged over time
with open('ratio_distances_within_radii.txt') as f:
    text = f.readlines()
ris = []
fractions = []
ri_fractions_within_str = [tuple(line.split()) for line in text[1:]]
for ri, frac in ri_fractions_within_str:
    ris.append(float(ri))
    fractions.append(float(frac))

# Plot fractions of interatomic distances within ri over radii
fig = plt.figure()
fig.suptitle("Fractions of interatomic distances within ri")
plt.xlabel("Radii")
plt.ylabel("Fractions of particles within treshold radius")
plt.plot(ris, fractions)
plt.show()
