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
m = 18.998403 #atomic mass units
N = int(args.steps) #Number of time steps
delta_t = float(args.timestep) #Time step length

## Read input file

# Predefine variables
init_coord = []
init_vel = []
with open(args.input) as f:
    contents = f.readlines()
    N_of_Atoms = int(contents[0])
    side_length = float(contents[2])
    for i in range(N_of_Atoms):
        Inp = contents[3+i].split()
        coord = [float(e) for e in Inp[:3]]
        vel = [float(e) for e in Inp[3:]]
        init_coord.append(coord)
        init_vel.append(vel)
coordinates = npj.array(init_coord)
velocities = npj.array(init_vel)

print(coordinates)
##########################################################################################
#%%      
##Define functions

# Morse potential
def E_potential(position):
    print(position)
    delta = position[:, npj.newaxis, :] - position
    indices = npj.triu_indices(position.shape[0], k=1)
    delta = delta[indices[0], indices[1], :]
    r2 = (delta * delta).sum(axis=1)
    r = npj.sqrt(r2)
    D_e = 1.6
    Alpha = 3.028
    r_e = 1.411
    V_Morse = D_e * (npj.exp(-2*Alpha*(r-r_e))-2*npj.exp(-Alpha*(r-r_e)))
    E_pot = sum(V_Morse)
    return E_pot

# Acceleration
def acceleration (position):
    morse_gradient = jax.jit(jax.grad(E_potential))
    forces = - morse_gradient(position)
    accel = forces/m
    return accel

# Velocity
def new_velocity(acceleration, acceleration_old,time_step, velocity_old):
    velocity = velocities + acceleration_old + (acceleration_old+acceleration)/2*delta_t
    return velocity

# Position
def new_positions(acceleration, velocity, time_step, position):
    new_position = position + velocity * time_step + 1/2*acceleration*time_step*time_step
    return new_position

# Periodic boundary conditions
def BC(position):
    for i in range(len(position)):
        for j in range(0,3) :
            if position[i,j] > side_length:
                mod = position[i,j] // side_length
                position = position.at[i,j].add(-mod * side_length)
            if position[i,j] < 0:
                mod = 1 + (abs(position[i,j]) // side_length)
                position = position.at[i,j].add(mod * side_length)
            else:
                pass
    return position

# Generate output
def Out(position,velocity):
    output = np.empty([len(position), 6])
    trjct = ""
    for i in range(len(position)):
        for j in range(0,3) :
            output[i,j] = position[i,j]
            output[i,j+3] = velocity[i,j]
    for line in output:
        for elem in line:
            trjct += str(elem) + " "
        trjct += "\n"
    return trjct

######################################################################################

## Iteration

# Clear output txt file
file = open("trajectory.txt","w")
file.close()

# First step
oldacceleration = acceleration (coordinates)                                                            #needed only for the first step

#Loop over steps
for i in range(0,N):    
    newcoordinates = BC(new_positions(oldacceleration,velocities,delta_t,coordinates))
    newacceleration = acceleration (newcoordinates)
    newvelocity = new_velocity(newacceleration, oldacceleration,delta_t, velocities)

    # Write some outputs into txt file                                                                  #to be adapted for bigger N
    if i%2 == 0:
        file = open("trajectory.txt", "a")
        content = str(N_of_Atoms) + "\n" + "Iteration Number: " + str(i) + "\n" + str(side_length) + "\n" + Out(newcoordinates,newvelocity)
        file.write(content)
        

    coordinates = newcoordinates
    velocities = newvelocity
    oldacceleration = newacceleration

    # Close output txt file
    file.close()


##End


