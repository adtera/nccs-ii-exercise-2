#%%
import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns

from utils import particle_cloud
#M = int(input()) # number of particles
#L = int(input()) # side length of sim box in Angstrom
#T = int(input()) # temperatute in kelvin

M = 1000
L = 100
T = 300


particles = particle_cloud(M, L, T)

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