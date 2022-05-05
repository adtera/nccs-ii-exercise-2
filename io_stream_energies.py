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
parser.add_argument('pos_arg', type=str,
                    help='Required input trajectory file')
args = parser.parse_args()


# Calculate Morse Potential
#D_e = 1.6
D_e = 0.0587989 #in 0.0587989 E_H = 1.6 eV
alpha = 3.028
r_e = 1.411
    
def v_morse(r):
#    pot = D_e * (np.exp(-2*alpha*(r-r_e)) -2*np.exp(alpha*(r-r_e)))
    pot = D_e * (np.exp(-2*alpha*(r-r_e)) -2*np.exp(-alpha*(r-r_e)))
    return pot

## Function definitions

# Calculate distances of atom configuration 
def calc_distances(configuration):
    positions = configuration[:,:3]
    deltas = positions[:,np.newaxis]- positions
    indices = np.triu_indices(positions.shape[0], k=1)
    deltas = deltas[indices[0], indices[1], :]
    deltas = deltas - L * np.round(deltas/L) #armin
    distances = np.sqrt((deltas*deltas).sum(axis=1))
    return distances

# Calculate Potential Energy Function
def calc_Epot(configuration):
    distances = calc_distances(configuration)
    #len(distances)

    # might be faster: E_pot = v_morse(distances).sum()

    E_pot = 0
    for d in distances:
        E_pot += v_morse(d)
    return E_pot

# Calculate Kinetic Energy
def calc_Ekin(configuration):
    #m_e = 0.2
    m_e = 18.998403 #debug ?
    velocities = configuration[:,3:]
    velocities_split = np.array_split(velocities,M)
    E_kin = 0
    for v in velocities_split:
        v_abs_square = np.linalg.norm(v) ** 2 
        E_kin += v_abs_square
    E_kin *= (m_e/2)
    
    return E_kin

# Calculate EKin and EPot for every timestep and write to list containing of tuples (epot,ekin)
 # Helper Function to slice input lines at at every nth element. E.g. len(lines) = 15 and n = 5, then we get 3 splits 
def slice_at_nth(lines,n):
    sliced_list = []
    start = 0
    end = n
    for i in range(len(lines)//n):
        sliced_list.append(lines[start:end])
        start += n
        end += n
    return sliced_list


def calculate_energies(xyz,side_length):
    # Define periodic BC
    def BC(position):
        for i in range(len(position)):
            for j in range(0,3) :
                if position[i,j] > side_length:
                    mod = position[i,j] // side_length
                    #position = position.at[i,j].add(-mod * side_length)
                    np.add.at(position,[i,j],-mod * side_length)
                if position[i,j] < 0:
                    mod = 1 + (abs(position[i,j]) // side_length)
                    #position = position.at[i,j].add(mod * side_length)
                    np.add.at(position,[i,j],mod * side_length)
                else:
                    pass
        return position
    # Calc Energies for each timestep
    energies = []
    for t,timestep_config in enumerate(xyz):
        whole_config_timestep = np.zeros(6)
        for atom in timestep_config:
            sp = atom.split()
            each_atom = [float(a) for a in sp]
            #print(each_atom)
            whole_config_timestep = np.vstack((whole_config_timestep,[each_atom]))
        whole_config_timestep = whole_config_timestep[1:]
        print(whole_config_timestep)
        
        #Apply BC
        whole_config_timestep = BC(whole_config_timestep)
        ekin = calc_Ekin(whole_config_timestep)
        print(f'Ekin at timestep {t} is : {ekin}')
        epot = calc_Epot(whole_config_timestep)
        print(f'Epot at timestep {t} is : {epot}')
        energies.append((epot,ekin))
    return energies



# Import trajectory file
with open(args.pos_arg) as f:
    text = f.readlines()
# Get number of atoms s
M = int(text[0])
# Get length of box
L = float(text[2])
# Get atom positions and velocitities for every timestep seperately
configuration = slice_at_nth(text[3:],M)

# Calculate Energies
energies = calculate_energies(configuration,L)


with open('./energies.txt','w') as out_file:
    out_file.write("  ".join(["Epot","EKin"])+"\n")
    for line in energies:
        out_file.write("   ".join([str(line[0]),str(line[1])])+'\n')
#        out_file.write("   ".join([str(line[0]),str(line[1]), str(float(line[0]) + float(line[1]))])+'\n') # debug




# Plot EKin over time and discuss
kB = 3.167 * 10 **(-6)
time = []
ekin_over_time = []
epot_over_time = []
e_total = []
for t,energy in enumerate(energies):
    print(energy)
    ekin = float(energy[1])
    epot = float(energy[0])
    ekin_over_time.append(ekin/(M*kB*3/2))
    epot_over_time.append(epot)
    e_total.append(ekin + epot)
    time.append(t)
ekin_over_time
fig = plt.figure()
plt.plot(ekin_over_time)
fig.suptitle("Evolution of Kinetic Temperature")
plt.xlabel("Time/fs")
plt.ylabel("Ekin/(M*kB*3/2)")
plt.show()


fig = plt.figure()
plt.plot(epot_over_time)
fig.suptitle("Evolution of Potential Energy")
plt.xlabel("Time/fs")
plt.ylabel("Epot")
plt.show()


fig = plt.figure()
plt.plot(e_total)
fig.suptitle("Evolution of Total Energy")
plt.xlabel("Time/fs")
plt.ylabel("E_total")
plt.show()