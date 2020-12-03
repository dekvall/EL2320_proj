#!/usr/bin/env python3

from propagate_state import propagate_state
import numpy as np
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
from numpy.random import multivariate_normal, rand

# Set up initial state and measurement with noise
x0 = np.array([0, 2, 1.5, 0])
R = 0.15**2 * np.eye(2)
z0 = multivariate_normal(x0[:2], R)

# For the filter
invR = np.linalg.inv(R)


# Initial state estimate
xh0 = np.array([*z0, 1, 0])
P0 = block_diag(R, 2**2 * np.eye(2))
nX = 100 # Number of particles
X = multivariate_normal(xh0, P0, size=nX)
w = 1/nX * np.ones((nX,))


# Get for each ball
#X = np.apply_along_axis(lambda x: propagate_state(0, tk, x)[0], axis=1, arr=X)

#plt.scatter(X[:,0], X[:,1], marker=".", c='b', label="Propagated Particle")
#plt.show()

def prop_f(t_start, t_end, state):
	x, *_ = propagate_state(t_start, t_end, state)
	return x

def measurement_f(x):
	return x[:2]

def obs_error_p(dz):
	# Gaussian error which will be normalized later
	return np.exp(-.5 * dz.T @ invR  @ dz)

# particle filter
def filter_for_one(t_before, t_k, X_before, w_before, u_before, z_k, f, h, obs_error):
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

	N, nx = X_before.shape
	nz, = z_k.shape


	X_new = np.zeros((N, nx))
	Z_new = np.zeros((N, nz))
	w_new = np.zeros((N,))

	for i in range(N):
		X_new[i, :] = f(t_before, t_k, X_before[i, :])
		Z_new[i, :] = h(X_new[i, :])
		w_new[i] = w_before[i] * obs_error(z_k - Z_new[i, :])
	w_new /= w_new.sum()

	x_hat = np.average(X_new, axis=0, weights=w_new)

	# Systematic resampling, because superior
	cdf = np.cumsum(w_new)
	n_draws = N
	ind = np.zeros(n_draws, dtype=int)
	draws = np.sort(rand(n_draws))

	ci = 0
	for di in range(n_draws):
		while cdf[ci] < draws[di] and ci < N:
			ci += 1
		ind[di] = ci

	X_new = X_new[ind]
	w_new = np.repeat(1/N, N)

	# Perturb the new particles a little, i should probably not do this with P0 but rather with something correlated with the measurement
	X_new = np.apply_along_axis(lambda r: multivariate_normal(mean=r, cov=P0), axis=1, arr=X_new)
	

	return x_hat, X_new, w_new

	# Maybe just use a normal for-loop, yes
	# X_new = np.apply_along_axis(lambda r: f(t_before, t_k, r), axis=1, arr=X_before)
	# Z_new = np.apply_along_axis(lambda r: h(t_k, r), axis=1, arr=X_new)
	# w_new = np.apply_along_axis(lambda r: r[0] * 0.3, arr=np.concatenate((X_new, Z_new))) # Actually observation error


# Ground truth
tk = 1
xk, *_ = propagate_state(0, tk, x0)
zk = multivariate_normal(xk[:2], R)

xk = x0
zk = z0

dt = .2
# Loop for 10 secs
for t in np.arange(0, 10, dt):
	plt.xlabel("X [m]")
	plt.ylabel("Y [m]")
	ax = plt.gca()
	ax.set_xlim(-1, 11)
	ax.set_ylim(0, 5)

	x_hat, X, w = filter_for_one(t, t+dt, X, w, 0, zk, prop_f, measurement_f, obs_error_p)
	
	# A posteriori
	plt.scatter(X[:,0], X[:,1], marker=".", c='r', label="Particle")
	plt.scatter(zk[0], zk[1], c='y', label="Measurement")
	plt.scatter(x_hat[0], x_hat[1], c='m', label="Approximation")
	plt.scatter(xk[0], xk[1], c='g', label="Ground truth")
	plt.legend()
	plt.grid()
	plt.pause(dt)
	plt.clf()
	# Ground truth
	xk, *_ = propagate_state(t, t+dt, xk)
	zk = multivariate_normal(xk[:2], R)

