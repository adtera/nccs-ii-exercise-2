#%%
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
from utils import particle_cloud, E_potential, acceleration

import jax
import jax.numpy as npj
import argparse

np.random.seed(800)
L = 3 #int(input())
M = (10,) #int(input())
T = 300 #int(input())
particles = particle_cloud(M[0], L, T)
coords = particles.get_array()
coordinates = npj.array(coords)#

options = {
    'gtol': 1e-30,
    'disp': True,
    'return_all': True
    }


e_pot_gradient = jax.jit(jax.grad(E_potential))
#%%
res = minimize(
    E_potential,
    coords,
    args = M,
    method = "CG",
    jac = e_pot_gradient,
#    jac = acceleration,
    options = options)

new_coords = np.array(res.x).reshape(M[0], 3)
#%%
res
#%%
print(coords)
print(new_coords)
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

x = particles.linspace_axis_1d

fig = plt.figure(figsize = (15,15))

ax = fig.add_subplot(111)
ax.scatter(x, x)
ax.set_xlim(0,L)
ax.set_ylim(0,L)
plt.show()


#%%
pos = np.random.random((M, 3))



#%%


theta = np.random.random(M) * 2 * np.pi
pos_true = L * pos

E_pot 
#%%
print(pos_true)
# %%
fig = plt.figure(figsize = (15,15))

ax = fig.add_subplot(111, projection='3d')
x,y,z = zip(*pos_true)
ax.scatter(x, y, z)
ax.set_xlim(0,L)
ax.set_ylim(0,L)
ax.set_zlim(0,L)
plt.show()

#%%
for particle in range(M):
    pass

   
def v_morse(r):
    D_e = 5
    alpha = 2
    r_e = 1

    pot = D_e * (np.exp(-2*alpha*(r-r_e)) -2*np.exp(alpha*(r-r_e)))
    return pot

trajectories = np.random.rand(10,6)
trajectories

positions = trajectories[:,:3]
velocities = trajectories[:,3:]
deltas = positions[:,np.newaxis]- positions
indices = np.triu_indices(positions.shape[0], k=1)
deltas = deltas[indices[0], indices[1], :]
deltas
distances = np.sqrt((deltas*deltas).sum(axis=1))
#len(distances)
E_pot = 0
for d in distances:
    E_pot += v_morse(d) 
E_pot