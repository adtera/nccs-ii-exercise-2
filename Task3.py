#!/usr/bin/env python
#%%
import numpy as np
import jax
import jax.numpy as npj
import argparse

##########################################################################################

## Input arguments

parser = argparse.ArgumentParser()
parser.add_argument("input", help="name of input txt file")
parser.add_argument("timestep", help="length of a single timestep")
parser.add_argument("steps", help="numer of time steps")
args = parser.parse_args()

# Given values
m = 18.998403  #atomic mass units
delta_t = float(
    args.timestep)  #Time step length in 'atomic time units' or 0.0241885 fs
N = int(args.steps)  #Number of time steps

## Read input file

# Predefine variables
init_coord = []
init_vel = []
with open(args.input) as f:
    contents = f.readlines()
    N_of_Atoms = int(contents[0])
    side_length = float(contents[2])
    for i in range(N_of_Atoms):
        Inp = contents[3 + i].split()
        coord = [float(e) for e in Inp[:3]]
        vel = [float(e) for e in Inp[3:]]
        init_coord.append(coord)
        init_vel.append(vel)
coordinates = npj.array(init_coord)
velocities = npj.array(init_vel)

#print(coordinates)
##########################################################################################
##Define functions


# Morse potential
def E_potential(position):
    # print(position)
    delta = position[:, npj.newaxis, :] - position
    indices = npj.triu_indices(position.shape[0], k=1)
    delta = delta[indices[0], indices[1], :]
    delta = delta - side_length * npj.round(delta / side_length)
    r2 = (delta * delta).sum(axis=1)
    r = npj.sqrt(r2)
    D_e = 0.0587989  #in 0.0587989 E_H = 1.6 eV
    Alpha = 3.028
    r_e = 1.411
    V_Morse = D_e * (npj.exp(-2 * Alpha * (r - r_e)) - 2 * npj.exp(-Alpha *
                                                                   (r - r_e)))
    E_pot = V_Morse.sum()
    return E_pot


# Acceleration
def get_acceleration(position):
    morse_gradient = jax.jit(jax.grad(E_potential))
    forces = -morse_gradient(position)
    accel = forces / m
    return accel


# Velocity
def get_velocity(acceleration, acceleration_old, time_step, velocity_old):
    velocity = velocity_old + (
        (acceleration_old + acceleration) / 2) * time_step
    return velocity


# Position
def get_positions(acceleration, velocity, time_step, position):
    new_position = position + velocity * time_step + (
        1 / 2) * acceleration * time_step * time_step
    return new_position


# Periodic boundary conditions
def BC(position):
    for i in range(len(position)):
        for j in range(0, 3):
            if position[i, j] > side_length:
                mod = position[i, j] // side_length
                position = position.at[i, j].add(-mod * side_length)
            if position[i, j] < 0:
                mod = 1 + (abs(position[i, j]) // side_length)
                position = position.at[i, j].add(mod * side_length)
            else:
                pass
    return position


# Generate output
def Out(position, velocity):
    output = np.empty([len(position), 6])
    trjct = ""
    for i in range(len(position)):
        for j in range(0, 3):
            output[i, j] = position[i, j]
            output[i, j + 3] = velocity[i, j]
    for line in output:
        for elem in line:
            trjct += str(elem) + " "
        trjct += "\n"
    return trjct


######################################################################################

## Iteration

# Clear output txt file
file = open("trajectory.txt", "w")
file.write(
    str(N_of_Atoms) + "\n" + "Task3 output" + "\n" + str(side_length) + "\n")
file.close()

# First step
acceleration = get_acceleration(coordinates)  #needed only for the first step

get_positions_jit = jax.jit(get_positions)
get_acceleration_jit = jax.jit(get_acceleration)
get_velocity_jit = jax.jit(get_velocity)

#Loop over steps
for i in range(0, N):
    print(f'TIMESTEP: {i}')
    #    new_coordinates = BC(get_positions_jit(acceleration,velocities,delta_t,coordinates))
    new_coordinates = get_positions(acceleration, velocities, delta_t,
                                    coordinates)
    new_acceleration = get_acceleration_jit(new_coordinates)
    new_velocity = get_velocity_jit(new_acceleration, acceleration, delta_t,
                                    velocities)

    # Write some outputs into txt file                                                                  #to be adapted for bigger N
    if (i + 1) % 1000 == 0 or i == 0:
        file = open("trajectory.txt", "a")
        content = Out(new_coordinates, new_velocity)
        file.write(content)

    coordinates = new_coordinates
    velocities = new_velocity
    acceleration = new_acceleration

    # Close output txt file
    file.close()

##End
