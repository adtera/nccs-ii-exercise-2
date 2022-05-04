#%%
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from utils import particle_cloud, save_to_file
import argparse
import jax
import jax.numpy as npj


print('=== DEFINING FUNCTIONS')
count = 0
def E_potential(position):
    print('--- Start function: E_potential')
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
    print('calculating V_Morse')
    ##
#    E_pot = 0
#    for _r in r:
#        E_pot += D_e * (npj.exp(-2*Alpha*(_r-r_e))-2*npj.exp(-Alpha*(_r-r_e)))
    ##
    V_Morse = D_e * (npj.exp(-2*Alpha*(r-r_e))-2*npj.exp(-Alpha*(r-r_e)))
    E_pot = sum(V_Morse)
    count += 1
    print(f'iter: {count}')
    print('--- Exit function: E_potential')
    return E_pot

def energy_gradient2(position):
    print(position)
    return jax.jit(jax.grad(E_potential))


def energy_gradient(position):
    print('--- Start function: energy_gradient')
    morse_gradient = jax.jit(jax.grad(E_potential))
    print('--- Exit function: energy_gradient')
    return morse_gradient(position)
    
## Input arguments
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("length", help="name of input txt file")
    parser.add_argument("number", help="length of a single timestep")
    parser.add_argument("temperature", help="numer of time steps")
    return parser.parse_args()

np.random.seed(800)

# args = parse_arguments()
args = False

if args:
    L = float(args.length)
    M = int(args.number)
    T = float(args.temperature)
else:
    L = 2 #int(input())
    M = 100 #int(input())
    T = 300 #int(input())


config = (L,M)
print(f"values: L = {L}, M = {M}, T = {T}")

particles = particle_cloud(L, M, T)
position = particles.get_array().flatten()

#k = scipy.constants.physical_constants["Boltzmann constant in eV/K"][0]
#coordinates = npj.array(coords)#

options = {
    'gtol': 1e-2,
    'disp': True,
#    'maxiter': 3,
    'return_all': True
    }

print('=== START SCIPY.OPTIMIZE.MINIMIZE')
res = minimize(
    E_potential,
    position,
#    args = config,
    method = "CG",
    jac = energy_gradient,
    options = options
    )
print('=== EXIT SCIPY.OPTIMIZE.MINIMIZE')


pos_input = np.array(res.x).reshape(M[0], 3)

##### velocity
k = 3.166811429 * (10 ** -6)
varr = np.sqrt((k*T)/18.998403)
mu = np.array([0.0, 0.0, 0.0])
sigma = np.array([varr, varr, varr])
covariance = np.diag(sigma**2)

vels = np.random.multivariate_normal(mean= mu, cov=covariance, size=(M[0],1))
#vels = vels.reshape(np.array(coords).shape)
vels_0 = vels - vels.mean(axis=0, keepdims=True)

#print([ sum(row[i] for row in vels) for i in range(len(vels[0])) ])
#print([ sum(row[i] for row in velssss) for i in range(len(velssss[0])) ])

vels_0 = np.array(vels_0)
vel_input = np.array([vel for [vel] in vels_0])

print('start writing input.txt')
save_to_file(pos_input, vel_input)
print("DONE")
#%%


print(pos_input)
print(vel_input)
#%%
print(res)
#%%
print(E_potential(coords))
print(E_potential(vel_input))
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

