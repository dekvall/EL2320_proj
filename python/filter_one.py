#!/usr/bin/env python3

from propagate_state import propagate_state
import numpy as np
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
from numpy.random import multivariate_normal, rand, randint

np.random.seed(69)

def prop_f(t_start, t_end, state):
	x, *_ = propagate_state(t_start, t_end, state)
	return x

def measurement_f(x):
	return x[:2]

def obs_error_p(dz, invR):
	# Gaussian error which will be normalized away with the particles later
	return np.exp(-.5 * dz.T @ invR  @ dz)

def normalizer(d, detR):
	return 1 / (np.power(2*np.pi, d/2) * np.sqrt(detR))

def uniform_init(x_low, x_high, y_low, y_high, v_x, v_y, n):
	return 	np.hstack((
		np.random.uniform(x_low, x_high, (n, 1)),
		np.random.uniform(y_low, y_high, (n, 1)),
		np.random.uniform(-v_x, v_x, (n, 1)),
		np.random.uniform(-v_y, v_y, (n, 1))
		))

def systematic_resampling(weights, n_draws):
	cdf = np.cumsum(weights)
	particle_ind = np.zeros(n_draws, dtype=int)
	draws = np.sort(rand(n_draws))

	ci = 0
	for di in range(n_draws):
		while ci < n_draws and cdf[ci] < draws[di]:
			ci += 1
		particle_ind[di] = ci
	return particle_ind

def add_random_sample(X, n_particles, sample_prob, x_low, x_high, y_low, y_high):
	"""
	Randomly resamples particles from an array of particles

	Input:
		X [np.array]: Particles
		n_particles [int]: Number of particles to be randomly sampled
		sample_prob [float 0-1]: Probability of resample
		x_low [float]: Lower boundary of x
		x_high [float]: Upper boundary of x
		y_low [float]: Lower boundary of y
		y_high [float]: Upper boundary of y
	
	Return:
		X [np.array]: New Particles
	"""
	N, nx = X.shape
	if sample_prob >= np.random.random():

		X[np.random.randint(0, N-1)]


def apriori(t_before, t_k, state_before, P):
	state = prop_f(t_before, t_k, state_before)
	return multivariate_normal(state, P)


def aprori_all_particles(t_before, t_k, X, P):
	return np.apply_along_axis(lambda r: apriori(t_before, t_k, r, P), axis=1, arr=X)


def posteriori(X, w, z_k, R, beta=None):
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
	if len(z_k) != 0:
		eps = np.spacing(1)

		N, nx = X.shape
		nz, = z_k.shape
		Z = np.zeros((N, nz))
		invR = np.linalg.inv(R)
		detR = np.linalg.det(R)
		norm = normalizer(len(z_k), detR)
		for i in range(N):
			Z[i, :] = measurement_f(X[i, :])
			w[i] *= norm*obs_error_p(z_k - Z[i, :], invR)
			if beta is not None:
				assert type(beta) is np.ndarray, "Beta must be an array"
				# print(beta)
				w[i] = np.sum(beta * w[i])

		if w.sum() < eps: # Avoid /0 errors when particles are too far away.
			w = np.repeat(1/N, N)
		else:
			w /= w.sum()

		ind = systematic_resampling(w, N)
		X = X[ind]
		w = np.repeat(1/N, N)

		# Use sample covariance to perturb the particles
		P = np.cov(X.T) #rows are variables and cols are observations in np.cov

	#   This should be the same as the propagation in the loop above, except for the addition of tune
	#	X = np.apply_along_axis(lambda r: multivariate_normal(mean=r, cov=tune * P), axis=1, arr=X)

	x_hat = np.average(X, axis=0, weights=w)
	
	return x_hat, X, w

def jpda_posteriori(X, w, z_k, ball_idx, beta, obs_error_dict):
	"""
	*_before: values for * at k-1
	t_k: time for current measurement
	z_k: list of measurements at k
	f: propagation function
	h: measurement function

	return
	x_hat: mean state estimate for k
	X_k: particle set for k
	w_k: particle weights for k
	"""
	if len(z_k) != 0:
		eps = np.spacing(1)

		N, nx = X.shape
		nz, = z_k[0].shape

		if beta is not None:
			assert type(beta) is np.ndarray, "Beta must be an array"
		
		Z = np.zeros((N, nz))
		
		# Only iterate over the non-zero elements of beta to reduce computational complexity
		z_a = np.array(z_k)
		meas_idx = np.where(beta[1:] != 0)[0]

		for i in range(N):
			Z[i, :] = measurement_f(X[i, :])
			w_temp = beta[0]
			for m_idx in meas_idx:
				w_temp += beta[m_idx+1]*obs_error_dict[(m_idx, ball_idx)][1][i]
			w[i] *= w_temp

		if w.sum() < eps: # Avoid /0 errors when particles are too far away.
			w = np.repeat(1/N, N)
			print("no")
		else:
			w /= w.sum()

		ind = systematic_resampling(w, N)
		X = X[ind]
		w = np.repeat(1/N, N)


	x_hat = np.average(X, axis=0, weights=w)
	
	return x_hat, X, w



if __name__ == "__main__":
	# Set up initial state and measurement with noise
	x0 = np.array([0, 3, 2, -6])
	R = 0.4**2 * np.eye(2)
	P = 0.1**2 * np.eye(4)
	z0 = multivariate_normal(x0[:2], R)
	R_meas = 0.3**2 * np.eye(2)
	# For the filter
	# invR = np.linalg.inv(R)

	# Initial state estimate
	xh0 = np.array([*z0, 1, 0])
	nX = 1000 # Number of particles

	# Track around initial measurement
	# P0 = block_diag(R, 2**2 * np.eye(2))
	# X = multivariate_normal(xh0, P0, size=nX)

	# Start tracking uniformly
	X = uniform_init(x_low=-1, x_high=11, y_low=0, y_high=5, v_x=4, v_y=4, n=nX)

	w = np.repeat(1/nX, nX)
	X1 = X
	xk = x0
	zk = z0
	errs = []
	dt = .05

	# add_random_sample(X, 3, 0.8, x_low, x_high, y_low, y_high)

	# Loop for 10 secs
	for t in np.arange(0, 7, dt):
		plt.xlabel("X [m]")
		plt.ylabel("Y [m]")
		ax = plt.gca()
		ax.set_xlim(-1, 11)
		ax.set_ylim(0, 5)
		x_hat, X, w = posteriori(X, w, zk, R)
		errs.append(xk - x_hat)
		# A posteriori
		plt.scatter(X[:,0], X[:,1], marker=".", c='r', alpha=.05, label="Particle")
		plt.scatter(zk[0], zk[1], c='b', label="Measurement")
		plt.scatter(x_hat[0], x_hat[1], c='m', label="Approximation")
		plt.scatter(xk[0], xk[1], c='g', label="Ground truth")
		plt.legend()
		plt.grid()
		plt.pause(dt/100)
		plt.clf()

		X = aprori_all_particles(t, t+dt, X, P)
		# Ground truth
		xk, state_traj, *_ = propagate_state(t, t+dt, xk)
		zk = multivariate_normal(xk[:2], R_meas)
		plt.plot(state_traj[:,0], state_traj[:,1], c='g')

	err = np.array(errs)
	plt.plot(err[:,0], label="X err")
	plt.plot(err[:,1], label="Y err")
	plt.plot(err[:,2], label="vx err")
	plt.plot(err[:,3], label="vy err")
	plt.grid()
	plt.legend()
	plt.show()

