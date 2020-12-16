#!/usr/bin/env python3

from filter_one import posteriori, uniform_init, obs_error_p, aprori_all_particles
from propagate_state import propagate_state
import numpy as np
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
from numpy.random import multivariate_normal, rand
import sys
import math
import argparse
from rich.console import Console
from rich.table import Table

console = Console()


# Main loop for all balls
# Specify init position and velocities for all balls
# Calculate final error, with verbose options

N_PARTICLES = 1000

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
		self.old_t = 0

	def init_filter(self, R):
		self.R = R
		self.particles = uniform_init(x_low=-1, x_high=11, y_low=0, y_high=5, v_x=4, v_y=4, n=N_PARTICLES)
		self.weights = np.repeat(1/N_PARTICLES, N_PARTICLES)
		self.state_estimate = np.average(self.particles, axis=0, weights=self.weights)
		self.z_hat = self.state_estimate[:2]

	def propagate_simulation(self, next_t: float):
		self.state, st, tt = propagate_state(self.t, next_t, self.state)
		self.state_traj = np.vstack((self.state_traj, st[1:,:])) if self.state_traj is not None else st
		self.time_traj = np.concatenate((self.time_traj, tt[1:])) if self.time_traj is not None else tt
		self.old_t = self.t
		self.t = next_t


	def estimate(self, zk, beta=None):
		self.state_estimate, self.particles, self.weights = posteriori(self.particles,
																		self.weights,
																		zk,
																		self.R,
																		beta)
		err = self.state - self.state_estimate
		self.errs = np.vstack((self.errs, err)) if self.errs is not None else err

	def predict(self, dt):
		self.particles = aprori_all_particles(self.t, self.t+dt, self.particles, self.P)
		self.z_hat = np.average(self.particles, axis=0, weights=self.weights)[:2]

def calc_beta(measurements, balls):
	"""
	beta: sizeof(targets) x sizeof(measurements) 
	"""
	M = len(measurements)
	T = len(balls)
	P_D = .9
	P_FA = .05

	# z0 assignments, i.e. false measurements,
	# should differ for each ball but we don't care atm 
	# since we have no gates to care about.
	gate_radius = 5

	# theta_t^j is the particular event which assigns the measurement j to the target t
	individual_events = np.zeros((T, M))
	joint_probs = np.zeros_like(individual_events)

	for t, ball in enumerate(balls):
		for j, m in enumerate(measurements):
			dist = np.linalg.norm(m - ball.z_hat)
			if dist > gate_radius:
				continue
			individual_events[t, j] = obs_error_p(m - ball.z_hat, ball.R)
	for t, target in enumerate(individual_events):
		valid_measurements = target != 0
		n_z0 = M - valid_measurements.sum()
		p_theta = P_D**(T-n_z0)*(1 - P_D)**n_z0*P_FA**(M-(T-n_z0))*np.prod(target, where=valid_measurements)
		joint_probs[t][valid_measurements] = p_theta

	"""
	Figure out wtf,
	... as the θ_j^t summation over all the joint events in
	which the marginal event θ^j_t, of interest occurs.

	Why is this dimension so high?
	"""
	# There's  < 0% chance that this is right..
	# I have no idea.. this maybe?
	beta = np.tile(joint_probs.sum(axis=1), (T,1))
	return beta


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

def display_table(balls):
	table = Table(show_header=True, header_style="bold")
	table.add_column("Param")
	table.add_column("Mean error")
	table.add_column("Variance")
	for i, ball in enumerate(balls, start=1):
		x_err, y_err, vx_err, vy_err = np.mean(np.absolute(ball.errs),axis=0)
		x_var, y_var, vx_var, vy_var = np.var(ball.errs, axis=0)
		table.add_row(f"Ball {i}", end_section=True)
		table.add_row("x", f"{x_err:.3f}", f"{x_var:.3f}")
		table.add_row("y", f"{y_err:.3f}", f"{y_var:.3f}")
		table.add_row("vx", f"{vx_err:.3f}", f"{vx_var:.3f}")
		table.add_row("vy", f"{vy_err:.3f}", f"{vy_var:.3f}", end_section=True)
	console.print(table)

def main(args):
	noplot = args["noplot"]
	R = 0.1**2 * np.eye(2)
	P = 0.1 **2 * np.eye(4) #Try with different covariances for velocities?
	# Base colors available: gbcmyk
	balls = [init_ball([0, 3, 2, -6], R, P, 'g'),
			init_ball([10, 5, -1, -1], R, P, 'b')]
	
	dt = .05
	simulation_time = 5
	if not noplot:
		plt.xlabel("X [m]")
		plt.ylabel("Y [m]")
		ax = plt.gca()
		ax.set_xlim(-1, 11)
		ax.set_ylim(0, 5)
		plt.grid()

	# Loop for 10 secs
	for t in np.arange(0, simulation_time, dt):
		measurements = [multivariate_normal(b.state[:2], b.R) for b in balls]
		beta = calc_beta(measurements, balls)
		for i, ball in enumerate(balls, start=1):
			# Filter
			zk = multivariate_normal(ball.state[:2], R)
			ball.estimate(zk, beta)
			ball.predict(dt)

			if not noplot:
				plt.scatter(ball.state[0], ball.state[1], marker=".", c=ball.color, label=f"G.T. Ball #{i}")
				plt.scatter(ball.state_estimate[0], ball.state_estimate[1], c='k', marker="*", label=f"Approx. Ball #{i}")
				plt.pause(dt/10)
			# Ground truth
			ball.propagate_simulation(t+dt)

	if not noplot:
		for ball in balls:
			traj = np.array(ball.state_traj)
			plt.plot(traj[:,0], traj[:,1], c=ball.color)

		for i, ball in enumerate(balls):
			plt.figure(i + 2)
			plot_error(ball)
			plt.title(f"Error for ball #{i + 1}")
		plt.show()

	display_table(balls)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="An implementaion of MC-JPDAF")
	parser.add_argument('--noplot', dest='noplot', default=False, action='store_true')
	args = vars(parser.parse_args())
	main(args)