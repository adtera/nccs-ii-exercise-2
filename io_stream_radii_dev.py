#! usr/bin/env python
# Imports n stuff
import numpy as np
from scipy import constants
from matplotlib import pyplot as plt
import sys
import math
import argparse
# Instantiate the parser
parser = argparse.ArgumentParser(description='Optional app description')
parser.add_argument('pos_arg', type=str, help='Required input trajectory file')
args = parser.parse_args()

# Calculate Morse Potential
#D_e = 1.6
D_e = 0.0587989  #in 0.0587989 E_H = 1.6 eV
alpha = 3.028
r_e = 1.411


def v_morse(r):
    #    pot = D_e * (np.exp(-2*alpha*(r-r_e)) -2*np.exp(alpha*(r-r_e)))
    pot = D_e * (np.exp(-2 * alpha * (r - r_e)) - 2 * np.exp(-alpha *
                                                             (r - r_e)))
    return pot


## Function definitions


# Calculate distances of atom configuration
def calc_distances(configuration):
    positions = configuration[:, :3]
    deltas = positions[:, np.newaxis] - positions
    indices = np.triu_indices(positions.shape[0], k=1)
    deltas = deltas[indices[0], indices[1], :]
    deltas = deltas - L * np.round(deltas / L)  #armin
    distances = np.sqrt((deltas * deltas).sum(axis=1))
    return distances


# Calculate Potential Energy Function
def calc_Epot(configuration):
    distances = calc_distances(configuration)
    return v_morse(distances).sum()


# Calculate Kinetic Energy
def calc_Ekin(configuration):
    #m_e = 0.2
    m_e = 18.998403  #debug ?
    velocities = configuration[:, 3:]
    velocities_squared = np.square(velocities)
    return velocities_squared.sum() * (m_e / 2)


# Calculate EKin and EPot for every timestep and write to list containing of tuples (epot,ekin)
# Helper Function to slice input lines at at every nth element. E.g. len(lines) = 15 and n = 5, then we get 3 splits
def slice_at_nth(lines, n):
    sliced_list = []
    start = 0
    end = n
    for i in range(len(lines) // n):
        sliced_list.append(lines[start:end])
        start += n
        end += n
    return sliced_list


# Import trajectory file
with open(args.pos_arg) as f:
    text = f.readlines()
# Get number of atoms s
M = int(text[0])
# Get length of box
L = float(text[2])
# Get atom positions and velocitities for every timestep seperately
configuration = slice_at_nth(text[3:], M)

# Calculate Energies
##energies = calculate_energies(configuration,L)#

#with open('./energies.txt','w') as out_file:
#    out_file.write("  ".join(["Epot","EKin"])+"\n")
#    for line in energies:
#        out_file.write("   ".join([str(line[0]),str(line[1])])+'\n')
##        out_file.write("   ".join([str(line[0]),str(line[1]), str(float(line[0]) + float(line[1]))])+'\n') # debug


# Define function calculating atomic distances bigger than threshold radius and weigh over time
def calculate_distances_within_treshold_radius(configurations, r, side_length):
    # Calculate distances bigger than threshold radius ri
    distances_within_ri_all_timesteps = 0
    for t, timestep_config in enumerate(configurations):
        print(f"timestep: {t}")
        whole_config_timestep = np.zeros(6)
        for atom in timestep_config:
            sp = atom.split()
            each_atom = [float(a) for a in sp]
            #print(each_atom)
            whole_config_timestep = np.vstack(
                (whole_config_timestep, [each_atom]))
        whole_config_timestep = whole_config_timestep[1:]
        # calculate all distances
        distances = calc_distances(whole_config_timestep)
        #print(distances)
        # count distances bigger than threshold distanc ri
        count_d_smaller_ri = sum(map(lambda x: x < r, distances))
        #print(count_d_bigger_ri)
        fraction_of_distances_within_ri = count_d_smaller_ri / len(distances)
        #print(fraction_of_distances_bigger_ri)
        distances_within_ri_all_timesteps += fraction_of_distances_within_ri

    # Weigh sum of distances within ri over timesteps
    d_within_ri_weighted_over_time = distances_within_ri_all_timesteps / len(
        configurations)
    return d_within_ri_weighted_over_time


# Radii are at equally spaced regular intervals between 0 and L/2
n_intervals = 10
rs = np.linspace(L / (2 * n_intervals), L / 2, n_intervals)

# Use only timesteps after the first 25% of them
t_25 = math.ceil(len(configuration) * 0.25)
configuration_timesteps_75 = configuration[t_25:]
n_timestpe_75 = len(configuration_timesteps_75)

print(len(configuration_timesteps_75))

# Write distances of radii smaller than treshold radius ri to outfile
with open('./ratio_distances_within_radii_dev.txt', 'w') as out_file:
    out_file.write("  ".join([
        "Threshold radii ", "Fraction of distances within radius over time"
    ]) + "\n")
    for r in rs:
        ratio = calculate_distances_within_treshold_radius(
            configuration_timesteps_75, r, L)
        out_file.write("   ".join([str(r), str(ratio)]) + '\n')
