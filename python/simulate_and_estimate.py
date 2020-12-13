#!/usr/bin/env python3

from filter_one import filter_for_one
from propagate_state import propagate_state
import numpy as np
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
from numpy.random import multivariate_normal, rand
import sys
import math

# Main loop for all balls
# Specify init position and velocities for all balls
# Calculate final error, with verbose options

N_PARTICLES = 100

class Ball:
	def __init__(self, state: list, R: list, P: list, color: str):
		self.state = state
		# This is the ground truth simulation for a ball
		self.init_simulation()
		# I suppose every ball gets a filter, luxurious!
		# Note: I don't think this is what they do in the paper, but it's too late to figure out atm
		self.init_filter(R)

		#Process noise of each simulation
		self.P = P
		self.color = color
		self.errs = None

	def init_simulation(self):
		self.state_traj = None
		self.time_traj = None
		self.t = 0

	def init_filter(self, R):
		self.invR = np.linalg.inv(R)
		z0 = multivariate_normal(self.state[:2], R)
		xh0 = np.array([*z0, 1, 0])
		P0 = block_diag(R, 2**2 * np.eye(2))
		self.particles = multivariate_normal(xh0, P0, size=N_PARTICLES)
		self.weights = 1/N_PARTICLES * np.ones((N_PARTICLES,))
		self.state_estimate = np.average(self.particles, axis=0, weights=self.weights)

	def propagate_simulation(self, next_t: float):
		self.state, st, tt = propagate_state(self.t, next_t, self.state)
		self.state_traj = np.vstack((self.state_traj, st[1:,:])) if self.state_traj is not None else st
		self.time_traj = np.concatenate((self.time_traj, tt[1:])) if self.time_traj is not None else tt
		self.old_t = self.t
		self.t = next_t

	def apply_filter(self, zk):
		self.state_estimate, self.particles, self.weights = filter_for_one(self.old_t,\
																			self.t,\
																			self.particles,\
																			self.weights,\
																			zk,\
																			self.invR, \
																			self.P)
		err = self.state - self.state_estimate
		self.errs = np.vstack((self.errs, err)) if self.errs is not None else err




def init_ball(x, R, P, color):
	x = np.array(x)
	return Ball(x, R, P, color)

def plot_error(ball):
	plt.plot(ball.errs[:,0], label="X err")
	plt.plot(ball.errs[:,1], label="Y err")
	plt.plot(ball.errs[:,2], label="vx err")
	plt.plot(ball.errs[:,3], label="vy err")
	plt.grid()
	plt.legend()

def main(verbose):
	R = 0.1**2 * np.eye(2)
	P = 0.1 **2 * np.eye(4) #Try with different covariances for velocities?
	# Base colors available: gbcmyk
	balls = [init_ball([0, 3, 2, -6], R, P, 'g')]
#			init_ball([10, 5, -1, -1], R, P, 'b')]
	
	dt = .05
	simulation_time = 5
	if verbose > 1:
		plt.xlabel("X [m]")
		plt.ylabel("Y [m]")
		ax = plt.gca()
		ax.set_xlim(-1, 11)
		ax.set_ylim(0, 5)
		plt.grid()
		
	# Loop for 10 secs
	for t in np.arange(0, simulation_time, dt):
		for i, ball in enumerate(balls):
			ball_no = i + 1
			# Ground truth
			ball.propagate_simulation(t+dt)

			# Filter
			zk = multivariate_normal(ball.state[:2], R)
			ball.apply_filter(zk)
			if verbose > 1:
				plt.scatter(ball.state[0], ball.state[1], marker=".", c=ball.color, label=f"G.T. Ball #{ball_no}")
				plt.scatter(ball.state_estimate[0], ball.state_estimate[1], c='k', marker="*", label=f"Approx. Ball #{ball_no}")
				plt.pause(dt/10)

				
	if verbose > 0:
		for ball in balls:
			traj = np.array(ball.state_traj)
			plt.plot(traj[:,0], traj[:,1], c=ball.color)

		for i, ball in enumerate(balls):
			plt.figure(i + 2)
			plot_error(ball)
			plt.title(f"Error for ball #{i + 1}")
		plt.show()

	for i, ball in enumerate(balls):
		x_err, y_err, vx_err, vy_err = np.mean(np.absolute(ball.errs),axis=0)
		x_var, y_var, vx_var, vy_var = np.var(ball.errs, axis=0)
		print("x: mean error: ", x_err, ", variance: ", x_var)
		print("y mean error: ", y_err, ", variance: ", y_var)
		print("vx mean error: ", vx_err, ", variance: ", vx_var)
		print("vy mean error", vy_err, ", variance: ", vy_var, "\n")

if __name__ == "__main__":
	if len(sys.argv) == 2:
		verbose_option = int(sys.argv[1])
	else: 
		verbose_option = 3

	main(verbose_option)