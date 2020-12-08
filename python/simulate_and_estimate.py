#!/usr/bin/env python3

from filter_one import filter_for_one
from propagate_state import propagate_state
import numpy as np
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
from numpy.random import multivariate_normal, rand


# Main loop for all balls
# Specify init position and velocities for all balls
# Calculate final error, with verbose options


N_PARTICLES = 1000

class Ball:
	def __init__(self, init_state: list, init_time: float, R: list):
		self.state = init_state
		self.t = init_time
		self.state_traj = [init_state]
		self.state_time = [init_time]
		self.R = R
		self.invR = np.linalg.inv(R)


	def propagate_simulation(self, next_t: float):
		self.state, *_ = propagate_state(self.t, next_t, self.state)
		self.state_traj.append(self.state)
		self.t = next_t
		self.state_time.append(next_t)


	def propagate_estimation(self, next_t: float):
		x_hat, X, w = filter_for_one(self.t, next_t, X, w, zk, self.invR)

# TODO
def init_estimate(x, R):
	pass

def main():
	R = 0.15**2 * np.eye(2)
	
	x1_init = np.array([0, 3, 2, -6])
	x2_init = np.array([10, 5, -1, -1])
	
	x1_init_estimate = 
	b1 = Ball(x1_init, 0, R)
	b2 = Ball(x2_init, 0, R)
	dt = .1
	# Loop for 10 secs

	plt.xlabel("X [m]")
	plt.ylabel("Y [m]")
	ax = plt.gca()
	ax.set_xlim(-1, 11)
	ax.set_ylim(0, 5)
	plt.legend()
	plt.grid()
	
	
	ball_list = [b1, b2]
	for t in np.arange(0, 10, dt):
		for ball in ball_list:
			# x_hat, X, w = filter_for_one(t, t+dt, X, w, zk, prop_f, measurement_f, obs_error_p)
			# errs.append(xk - x_hat)
			# # A posteriori
			# plt.scatter(X[:,0], X[:,1], marker=".", c='r', label="Particle")
			# plt.scatter(zk[0], zk[1], c='y', label="Measurement")
			# plt.scatter(x_hat[0], x_hat[1], c='m', label="Approximation")
			plt.scatter(ball.state[0], ball.state[1], c='g', label="Ground truth")
			plt.pause(0.01)
			# Ground truth
			ball.propagate_simulation(t+dt)
			# zk = multivariate_normal(xk[:2], R)
			# plt.plot(state_traj[:,0], state_traj[:,1], c='g')

	# err = np.array(errs)
	# plt.plot(err[:,0], label="X err")
	# plt.plot(err[:,1], label="Y err")
	# plt.plot(err[:,2], label="vx err")
	# plt.plot(err[:,3], label="vy err")
	# plt.grid()
	# plt.legend()
	# plt.show()
	B = np.array(ball.state_traj)
	plt.scatter(B[:,0], B[:,1])
	plt.show()


if __name__ == "__main__":
	main()