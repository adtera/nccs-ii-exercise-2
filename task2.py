#%%
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from utils import particle_cloud, save_to_file, BC, plot_3d
import argparse
import jax
import jax.numpy as npj

print('=== DEFINING FUNCTIONS')


def E_potential(position):
    position = np.reshape(position, (M, 3))
    delta = position[:, npj.newaxis, :] - position
    indices = npj.triu_indices(position.shape[0], k=1)
    delta = delta[indices[0], indices[1], :]
    delta = delta - L * npj.round(delta / L)
    r2 = (delta * delta).sum(axis=1)
    r = npj.sqrt(r2)
    D_e = 0.0587989  #in 0.0587989 E_H = 1.6 eV
    Alpha = 3.028
    r_e = 1.411
    V_Morse = D_e * (npj.exp(-2 * Alpha * (r - r_e)) - 2 * npj.exp(-Alpha *
                                                                   (r - r_e)))
    E_pot = V_Morse.sum()
    return E_pot


## Input arguments
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("length", help="length of box")
    parser.add_argument("number", help="number of particles")
    parser.add_argument("temperature", help="temperature in kelvin")
    return parser.parse_args()


np.random.seed(800)

args = parse_arguments()
#args = False

if args:
    L = float(args.length)
    M = int(args.number)
    T = float(args.temperature)
else:
    L = 7  #int(input())
    M = 100  #int(input())
    T = 300  #int(input())

config = (L, M)

particles = particle_cloud(L, M, T)
position = particles.get_array().flatten()

print(f"values: L = {L}, M = {M}, T = {T}, E_pot = {E_potential(position)}")
#%%
options = {
    'gtol': 1.19e-7,
    'disp': True,
    #    'maxiter': 30,
    #    'norm': inf,
    'return_all': True
}

print('=== START SCIPY.OPTIMIZE.MINIMIZE')

energy_gradient = jax.jit(jax.grad(E_potential))
res = minimize(
    E_potential,
    position,
    #    args = config,
    method="CG",
    jac=energy_gradient,  #jax.jit()
    options=options)
print('=== EXIT SCIPY.OPTIMIZE.MINIMIZE')

pos_input = np.array(res.x).reshape(M, 3)

##### velocity, subscript H denotes Hartree
#k = scipy.constants.physical_constants["Boltzmann constant in eV/K"][0]
k = 3.166811429 * (10**-6)  # in E_H * K^{-1}
varr = np.sqrt((k * T) / 18.998403)  # v_H
mu = np.array([0.0, 0.0, 0.0])
sigma = np.array([varr, varr, varr])
covariance = np.diag(sigma**2)

vels = np.random.multivariate_normal(mean=mu, cov=covariance, size=(M, 1))
vels_0 = vels - vels.mean(axis=0, keepdims=True)

#print([ sum(row[i] for row in vels) for i in range(len(vels[0])) ])
#print([ sum(row[i] for row in velssss) for i in range(len(velssss[0])) ])

vels_0 = np.array(vels_0)
vel_input = np.array([vel for [vel] in vels_0])

print('start writing input.txt')
save_to_file(BC(pos_input, L), vel_input, M, L)
print("DONE")
#%%
# print(pos_input)
# print(vel_input)
# print(res)
# print(E_potential(position))
# print(E_potential(pos_input))
#%%
# plot_3d(particles.get_array(), L)
# plot_3d(pos_input.tolist(), L)
plot_3d(BC(pos_input, L).tolist(), L)
