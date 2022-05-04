import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import scipy.optimize as minimize

import jax
import jax.numpy as npj
import argparse

class particle_cloud:
    def __init__(self, L, M, T):
        self._length_of_box = L
        self._number_of_particles = M
        self._temperature = T

        self._perfect_particles_1d = self.perfect_particles_1d(self._number_of_particles)
        self._perfect_number_of_particles = self.perfect_cube(self._perfect_particles_1d)
        self._ghost_particles = self._perfect_number_of_particles - self._number_of_particles
        self._particle_box_length = (self._length_of_box / (self._perfect_particles_1d + 1)) / 2
        self._axis_1d = self.linspace_axis_1d()

    def perfect_particles_1d(self, N):
        return int(np.floor(N ** (1/3)) + 1)

    def perfect_cube(self, perfect_particles_1d): 
        return int(perfect_particles_1d ** 3)

    def linspace_axis_1d(self):
        start = self._particle_box_length
        stop = self._length_of_box - start
        num = self._perfect_particles_1d
        return np.linspace(start, stop, num)

    def get_array(self):
        # array of perfect cube with cords of linspace_axis_1d
        arr = np.array(list(product(self._axis_1d,self._axis_1d,self._axis_1d)))
        # remove number of ghost particles randomly
        arr = arr.tolist()
        for _ in range(self._ghost_particles):
            random_index = np.random.randint(0, len(arr))
            arr.pop(random_index)
        # shift each particle coord with random distribution with distance of value start as sigma * 5
        for i in range(len(arr)):
            for j in range(len(arr[i])):
                shift_border = self._particle_box_length * 0.3
                shift = np.random.uniform((-1)*shift_border, shift_border) # A single value
                arr[i][j] += shift
        return np.array(arr)


def E_potential(position, *args):
    #not working
    print(args)
    L=args[0]
    M=args[1]
    position = np.reshape(position,(M,3))
    delta = position[:, npj.newaxis, :] - position
    indices = npj.triu_indices(position.shape[0], k=1)
    delta = delta[indices[0], indices[1], :]
    delta = delta - L * npj.round(delta/L)
    r2 = (delta * delta).sum(axis=1)
    r = npj.sqrt(r2)
    D_e = 1.6
    Alpha = 3.028
    r_e = 1.411
    V_Morse = D_e * (npj.exp(-2*Alpha*(r-r_e))-2*npj.exp(-Alpha*(r-r_e)))
    E_pot = sum(V_Morse)
    return E_pot

def energy_gradient(position, L, M):
    #not working
    morse_gradient = jax.jit(jax.grad(E_potential))
    args = (L,M)
    forces = morse_gradient(position, args)
    return np.array(forces)

def energy_gradient(position):
    #not workin. defining function with position doesnt make sense
    print('--- Start function: energy_gradient')
    morse_gradient = jax.jit(jax.grad(E_potential))
    print('--- Exit function: energy_gradient')
    return morse_gradient(position)


def out_string(position,velocity):
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

def save_to_file(position,velocity):
    with open("input.txt","w") as file:
        file.write(str(M[0]) + '\n')
        file.write("-" + '\n')
        file.write(str(L) + '\n')
        file.write(out_string(position,velocity))     