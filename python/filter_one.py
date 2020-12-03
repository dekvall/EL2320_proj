#!/usr/bin/env python3

from propagate_state import propagate_state
import numpy as np
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
from numpy.random import multivariate_normal

# Set up initial state and measurement with noise
x0 = np.array([0, 2, 1.5, 0])
R = 0.15**2 * np.eye(2)
z0 = multivariate_normal(x0[:2], R)


# Initial state estimate
xh0 = np.array([*z0, 1, 0])
P0 = block_diag(R, 2**2 * np.eye(2))
nX = 100 # Number of particles
X = multivariate_normal(xh0, P0, size=nX)
w = 1/nX * np.ones((1, nX))

fig = plt.figure()
plt.scatter(X[:,0], X[:,1], marker=".", c='r', label="Particle")
plt.xlabel("X [m]")
plt.ylabel("Y [m]")
ax = plt.gca()
ax.set_xlim(-1, 11)
ax.set_ylim(0, 5)
plt.scatter(z0[0], z0[1], c='y', label="Measurement")


# Ground truth
tk = 1
xk, *_ = propagate_state(0, tk, x0)
zk = multivariate_normal(xk[:2], R)

# Get for each ball
X = np.apply_along_axis(lambda x: propagate_state(0, tk, x)[0], axis=1, arr=X)

plt.scatter(X[:,0], X[:,1], marker=".", c='b', label="Propagated Particle")

plt.legend()
plt.grid()
plt.show()


def prop_f(t_start, t_end, state, u=0, proc_noise=None):
	x, *_ = propagate_state(t_start, t_end, state)
	return x

def measurement_f(t, x, u=0):
	return x[:2]

# particle filter
def filter_for_one(t_before, t_k, X_before, w_before, u_before, z_k, f, h):
	"""
	*_before: values for * at k-1
	t_k: time for current measurement
	z_k: measurement at k
	f: propagation function
	h: measurement function

	return
	x_hat: mean state estimate for k
	X_k: particle set for k
	w_k: particle weights for k
	"""

	# Maybe just use a normal for-loop
	X_new = np.apply_along_axis(lambda r: f(t_before, t_k, r), axis=1, arr=X_before)
	Z_new = np.apply_along_axis(lambda r: h(t_k, r), axis=1, arr=X_new)
	w_new = np.apply_along_axis(lambda r: r[0] * 0.3, arr=np.concatenate((X_new, Z_new))) # Actually observation error


