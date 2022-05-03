#%%
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
from utils import particle_cloud
# E_potential, acceleration
import scipy
import jax
import jax.numpy as npj
import argparse

def E_potential(position):
#    position = position.reshape((M, 3))
    position = np.reshape(position,(M[0],3))
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
def acceleration(position):
    morse_gradient = jax.jit(jax.grad(E_potential))
    forces = morse_gradient(position)
    #accel = forces/m
    #return accel
    return np.array(forces)

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
        
np.random.seed(800)
L = 2 #int(input())
M = (15,) #int(input())
T = 300 #int(input())
particles = particle_cloud(M[0], L, T)
coords = particles.get_array()

#k = scipy.constants.physical_constants["Boltzmann constant in eV/K"][0]
k = 3.166811429 * (10 ** -6)
varr = np.sqrt((k*T)/18.998403)

mu = np.array([0.0, 0.0, 0.0])

sigma = np.array([varr, varr, varr])
covariance = np.diag(sigma**2)

vels = np.random.multivariate_normal(mean= mu, cov=covariance, size=(M[0],1))
#vels = vels.reshape(np.array(coords).shape)
velssss = vels - vels.mean(axis=0, keepdims=True)

#print([ sum(row[i] for row in vels) for i in range(len(vels[0])) ])
#print([ sum(row[i] for row in velssss) for i in range(len(velssss[0])) ])

velssss = np.array(velssss)

#coordinates = npj.array(coords)#
coordinates = np.array(coords).flatten()

options = {
    'gtol': 1e-7,
    'disp': True,
    'return_all': True
    }
res = minimize(
    E_potential,
    coordinates,
    method = "CG",
    jac = acceleration,
#    jac = acceleration,
    options = options
    )

new_coords = np.array(res.x).reshape(M[0], 3)

vel_output = np.array([vel for [vel] in velssss])

save_to_file(new_coords, vel_output)
#%%
print("DONE")
#%%
print(res)
#%%
print(E_potential(coords))
print(E_potential(new_coords))
#print(np.array(new_coords))

#%%
x,y,z = zip(*coords)
fig = plt.figure(figsize = (15,15))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z)
ax.set_xlim(0,L)
ax.set_ylim(0,L)
ax.set_zlim(0,L)
plt.show()

#%%
x,y,z = zip(*new_coords.tolist())
fig = plt.figure(figsize = (15,15))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z)
ax.set_xlim(0,L)
ax.set_ylim(0,L)
ax.set_zlim(0,L)
plt.show()

#%%

