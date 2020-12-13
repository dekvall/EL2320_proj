#!/usr/bin/env python3

from propagate_state import propagate_state
import numpy as np
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
from numpy.random import multivariate_normal, rand

np.random.seed(69)

def prop_f(t_start, t_end, state):
	x, *_ = propagate_state(t_start, t_end, state)
	return x

def measurement_f(x):
	return x[:2]

def obs_error_p(dz, invR):
	# Gaussian error which will be normalized away with the particles later
	return np.exp(-.5 * dz.T @ invR  @ dz)

def systematic_resampling(weights, n_draws):
	cdf = np.cumsum(weights)
	particle_ind = np.zeros(n_draws, dtype=int)
	draws = np.sort(rand(n_draws))

	ci = 0
	for di in range(n_draws):
		while cdf[ci] < draws[di] and ci < n_draws:
			ci += 1
		particle_ind[di] = ci
	return particle_ind

def filter_for_one(t_before, t_k, X_before, w_before, z_k, invR, P):
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

	# Not sure how well this works,
	# but it's an optimal gaussian tuning parameter
	# The particles spread out less
	tune = (4/(N * (nx + 2)))**(1/(nx + 4));

	X_new = np.zeros((N, nx))
	Z_new = np.zeros((N, nz))
	w_new = np.zeros((N,))
	for i in range(N):
		X_new[i, :] = prop_f(t_before, t_k, X_before[i, :])
		Z_new[i, :] = measurement_f(X_new[i, :])
		X_new[i, :] += multivariate_normal(np.array([0, 0, 0, 0]), P)
		w_new[i] = w_before[i] * obs_error_p(z_k - Z_new[i, :], invR)
	w_new /= w_new.sum()

	ind = systematic_resampling(w_new, N)
	X_new = X_new[ind]
	w_new = np.repeat(1/N, N)

	# Use sample covariance to perturb the particles
	P = np.cov(X_new.T) #rows are variables and cols are observations in np.cov

#   This should be the same as the propagation in the loop above, except for the addition of tune
#	X_new = np.apply_along_axis(lambda r: multivariate_normal(mean=r, cov=tune * P), axis=1, arr=X_new)

	x_hat = np.average(X_new, axis=0, weights=w_new)
	
	return x_hat, X_new, w_new


if __name__ == "__main__":
	# Set up initial state and measurement with noise
	x0 = np.array([0, 3, 2, -6])
	R = 0.15**2 * np.eye(2)
	P = 0.3**2 * np.eye(4)
	z0 = multivariate_normal(x0[:2], R)

	# For the filter
	invR = np.linalg.inv(R)

	# Initial state estimate
	xh0 = np.array([*z0, 1, 0])
	P0 = block_diag(R, 2**2 * np.eye(2))
	nX = 1000 # Number of particles
	X = multivariate_normal(xh0, P0, size=nX)
	w = 1/nX * np.ones((nX,))

	xk = x0
	zk = z0
	errs = []
	dt = .1

	# Loop for 10 secs
	for t in np.arange(0, 10, dt):
		plt.xlabel("X [m]")
		plt.ylabel("Y [m]")
		ax = plt.gca()
		ax.set_xlim(-1, 11)
		ax.set_ylim(0, 5)

		x_hat, X, w = filter_for_one(t, t+dt, X, w, zk, invR, P)
		errs.append(xk - x_hat)
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
		xk, state_traj, *_ = propagate_state(t, t+dt, xk)
		zk = multivariate_normal(xk[:2], R)
		plt.plot(state_traj[:,0], state_traj[:,1], c='g')

	err = np.array(errs)
	plt.plot(err[:,0], label="X err")
	plt.plot(err[:,1], label="Y err")
	plt.plot(err[:,2], label="vx err")
	plt.plot(err[:,3], label="vy err")
	plt.grid()
	plt.legend()
	plt.show()

