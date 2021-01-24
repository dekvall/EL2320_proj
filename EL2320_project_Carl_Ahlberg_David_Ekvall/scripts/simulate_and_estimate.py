#!/usr/bin/env python3

from particle_filter import jpda_posteriori, posteriori, uniform_init, obs_error_p, aprori_all_particles, normalizer 
from propagate_state import propagate_state
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import multivariate_normal, rand
import sys
import math
from itertools import permutations
from parameters import N_PARTICLES, P_D, P_FA
np.random.seed(60)

class Ball:
	def __init__(self, state: list, R: list, P: list, color: str, bound: list, bounce_uncertainty: bool, t_no_meas_start: float, t_no_meas_end: float, known_init_pos: bool=False):
		self.state = state
		self.known_init_pos = known_init_pos
		self.gate_size = 10
		self.bound = bound
		self.particle_exclusion_radius = 2
		self.init_simulation()
		self.init_filter(R, P)

		self.color = color
		self.label = None
		self.errs = None

		# Time interval when there will be no measurements of the ball
		self.t_no_meas_start = t_no_meas_start
		self.t_no_meas_end = t_no_meas_end

		# If true, ball will randomly change velocities simulation when bouncing 
		self.bounce_uncertainty = bounce_uncertainty

	def init_simulation(self):
		self.state_traj = None
		self.time_traj = None
		self.estimate_traj = None
		self.t = 0
		self.old_t = 0
		
	def init_filter(self, R, P):
		self.R = R
		self.P = P
		if self.known_init_pos:
			self.particles = multivariate_normal(self.state, P, N_PARTICLES)
			self.particles[:,2] = np.random.uniform(-self.bound[4], self.bound[4], N_PARTICLES)
			self.particles[:,3] = np.random.uniform(-self.bound[5], self.bound[5], N_PARTICLES)

		else:
			self.particles = uniform_init(x_low=self.bound[0], 
											x_high=self.bound[1], 
											y_low=self.bound[2], 
											y_high=self.bound[3], 
											v_x=self.bound[4], 
											v_y=self.bound[5], 
											n=N_PARTICLES)

		self.weights = np.repeat(1/N_PARTICLES, N_PARTICLES)
		self.state_estimate = np.average(self.particles, axis=0, weights=self.weights)
		self.z_hat = self.state_estimate[:2]

	def propagate_simulation(self, next_t: float):
		self.state, st, tt = propagate_state(self.t, next_t, self.state, self.bounce_uncertainty)
		self.state_traj = np.vstack((self.state_traj, self.state)) if self.state_traj is not None else st
		self.time_traj = np.concatenate((self.time_traj, tt[1:])) if self.time_traj is not None else tt
		self.old_t = self.t
		self.t = next_t


	def estimate(self, zk, ball_idx, beta, obs_error_dict):
		self.state_estimate, self.particles, self.weights = jpda_posteriori(self.particles,
																		self.weights,
																		zk,
																		ball_idx,
																		beta,
																		obs_error_dict)

		err = self.state - self.state_estimate
		self.errs = np.vstack((self.errs, err)) if self.errs is not None else err
		self.estimate_traj = np.vstack((self.estimate_traj, self.state_estimate)) if self.estimate_traj is not None else self.state_estimate



	def predict(self, delta_t):
		self.particles = aprori_all_particles(self.t, self.t+delta_t, self.particles, self.P)
		self.z_hat = np.average(self.particles, axis=0, weights=self.weights)[:2]

	def has_valid_measurement(self):
		if (self.t_no_meas_start is not None and self.t >= self.t_no_meas_start) and (self.t_no_meas_end is None or self.t < self.t_no_meas_end):
			return False

		return True
		
	def has_converged(self, threshold):
		return (not self.known_init_pos) and (np.cov(self.particles[:,:2].T) < threshold).all()

	def remove_particles(self, balls):
		pos_estimations = np.empty([0,2])

		# Remove particles from areas cointaining existing targets
		for b in balls:
			dist = (self.particles[:,0]-b.z_hat[0])**2+(self.particles[:,1]-b.z_hat[1])**2
			self.particles = np.delete(self.particles, dist < self.particle_exclusion_radius**2, 0)
			pos_estimations = np.vstack((pos_estimations, np.array([b.z_hat[0], b.z_hat[1]])))

		# Resample particles to original number
		while self.particles.shape[0] < N_PARTICLES:
			x = np.random.uniform(self.bound[0], self.bound[1], 1)
			y = np.random.uniform(self.bound[2], self.bound[3], 1)
			vx = np.random.uniform(-self.bound[4], self.bound[4], 1)
			vy = np.random.uniform(-self.bound[5], self.bound[5], 1)
			dist = (pos_estimations[:,0]-x[0])**2+(pos_estimations[:,1]-y[0])**2
			if (dist > self.particle_exclusion_radius).all():
				self.particles = np.vstack((self.particles, np.array([x[0], y[0], vx[0], vy[0]])))



class MeasurementObject:

	def __init__(self, covariance, boundaries, p_measurement=1., p_clutter=0.):
		self.covariance = covariance
		self.boundaries = boundaries
		self.p_measurement = p_measurement
		self.p_clutter = p_clutter

	def get_clutter(self):
		if rand() < self.p_clutter:
			x_clutter = np.random.uniform(self.boundaries[0], self.boundaries[1])
			y_clutter = np.random.uniform(self.boundaries[2], self.boundaries[3])
			return [np.array([x_clutter, y_clutter])]
		return []

	def get_measurement(self, balls):
		return [multivariate_normal(b.state[:2], self.covariance) for b in balls if rand() < self.p_measurement and b.has_valid_measurement()] + self.get_clutter()



def dist(meas, ball):
	return (meas[0]-ball.z_hat[0])**2 + (meas[1]-ball.z_hat[1])**2

def feasible_association_events(measurements, targets, gating=True):
	'''
	Finds all feasible association event, that is
	the combinations of all targets and measurements.
	value "0" indicates invalid detection.
	One target can only be associated to a single measurement. 
	'''
	meas_linspace = np.arange(len(measurements)+1)
	padding = np.zeros(max(len(measurements), len(targets)), dtype=np.int16)
	meas_linspace = np.append(meas_linspace, padding)
	perm_obj = set(permutations(meas_linspace, len(targets)))
	perm = []
	for p in perm_obj:
		check = True
		if gating:
			for t, m in enumerate(p):
				if m != 0 and dist(measurements[m-1], targets[t]) > targets[t].gate_size**2:
					check = False
					break
		if check:
			perm.append(p)

	# No viable measurements if all measurements outside of gates
	if len(perm) == 0:
		print("No viable measurements")
		measurements.clear()
	
	return np.array(perm)

def predictive_likelihood(measurement, target):	
	invR = np.linalg.inv(target.R)
	detR = np.linalg.det(target.R)
	likelihood = np.zeros(N_PARTICLES)
	norm = normalizer(len(measurement), detR)
	for p in range(N_PARTICLES):
		likelihood[p] = norm*obs_error_p(measurement-target.particles[p][:2], invR)

	return np.sum(likelihood)*1/N_PARTICLES, likelihood

def joint_event_posterior(measurements, targets, event, obs_error_dict):
	'''
	Calculates the posterior probability of an event.
	Invalid detections (value 0) are omitted.
	'''
	target_associations = np.count_nonzero(event!=0)
	false_alarms = len(measurements) - target_associations
	weight = P_FA ** false_alarms * (1 - P_D) ** (len(targets) - target_associations) * P_D ** target_associations
	prod = 1
	for t, m_idx  in enumerate(event):
		if m_idx != 0:
			if (m_idx-1, t) in obs_error_dict:
				likelihood_sum = obs_error_dict[(m_idx-1, t)][0]
			else:
				likelihood_sum, likelihood_array = predictive_likelihood(measurements[m_idx-1], targets[t])
				obs_error_dict[(m_idx-1, t)] = (likelihood_sum, likelihood_array)

			prod *= likelihood_sum
	return weight*prod


def calc_beta(measurements, target_idx, targets, feasible_events):
	'''
	Calculates the probability of a target beeing associated to each measurement
	measurements: List of all measurements
	target_idx: Index of target in targets list 
	targets: List of targets
	feasible_events: np array of all feasible events
	'''
	obs_error_dict = {} # Dictionary used to remove duplicate computations of a target to a measurement
	beta = np.zeros(len(measurements)+1)
	if len(measurements) != 0:
		for m_idx in range(len(measurements)+1):
			events_rows = np.where(feasible_events[:,target_idx]==m_idx)[0]
			events = feasible_events[events_rows,:]
			for event in events:
				beta[m_idx]+=joint_event_posterior(measurements, targets, event, obs_error_dict)

	return beta, obs_error_dict


def init_ball(x, R, P, color, bound, bounce_uncertainty=False, no_meas_start=None, no_meas_end=None):
	x = np.array(x)
	return Ball(x, R, P, color, bound, bounce_uncertainty, no_meas_start, no_meas_end)

def plot_error(ball):
	plt.xlabel("Time step")
	plt.ylabel("Error")
	plt.plot(ball.errs[:,0], label="X err", c='r')
	plt.plot(ball.errs[:,1], label="Y err", c='g')
	plt.plot(ball.errs[:,2], label="vx err", c='b')
	plt.plot(ball.errs[:,3], label="vy err", c='y')
	plt.grid()
	plt.legend()

def display_results(balls):

	for i, ball in enumerate(balls, start=1):
		x_err, y_err, vx_err, vy_err = np.mean(np.absolute(ball.errs),axis=0)
		x_var, y_var, vx_var, vy_var = np.var(ball.errs, axis=0)
		print(f"""Ball {i}:
==================================================================
x MAE: {x_err}		x Var: {x_var}
y MAE: {y_err}		y Var: {y_var}
vx MAE: {vx_err}		vx Var: {vx_var}
vy MAE: {vy_err}		vy Var: {vy_var}
==================================================================
""")


def set_axis_limit(balls, ax):
	init_states = np.array([ball.state[0:2] for ball in balls])
	ax.set_xlim(np.min(init_states[:,0])-1, np.max(init_states[:,0])+1)
	ax.set_ylim(0, np.max(init_states[:,1])+1)


def monte_carlo_jpdaf_simulation(sim_time, delta_t, all_balls, meas_obj, plot_boundaries, noplot, target_map = None):

	covariance_threshold = 0.5

	# Determine if tracking or global localization
	ball_idx = 0
	if not all_balls[0].known_init_pos:
		balls = [all_balls[0]]

	else:
		balls = all_balls

	# Loop through specified simulation time
	for t in np.arange(0, sim_time, delta_t):
		# Initialization of new balls
		if balls[-1].has_converged(covariance_threshold):
			balls[-1].gate_size = 2
			if ball_idx + 2 <= len(all_balls):
				print("Ball nr ", ball_idx + 1, "Has converged, Initializing next ball")
				all_balls[ball_idx+1].remove_particles(balls)
				balls.append(all_balls[ball_idx+1])
				ball_idx += 1
		measurements = meas_obj.get_measurement(all_balls)
		plt.clf()

		if not noplot:
			plt.xlabel("X [m]")
			plt.ylabel("Y [m]")
			ax = plt.gca()
			ax.set_xlim(plot_boundaries[0], plot_boundaries[1])
			ax.set_ylim(plot_boundaries[2], plot_boundaries[3])
			plt.grid()

		feasible_events = feasible_association_events(measurements, balls)
		for i, ball in enumerate(balls, start=1):
			beta, obs_error_dict = calc_beta(measurements, i-1, balls, feasible_events)
			ball.estimate(measurements, i-1, beta, obs_error_dict)
			# Plot particles for the last initialized filter
			if (i == len(balls)) and not noplot:
				plt.scatter(balls[-1].particles[:,0], balls[-1].particles[:,1], marker=".", c='y')
			
			ball.predict(delta_t)
		if not noplot:
			# Plot active filter estimates
			for i, ball in enumerate(balls, start=1):
				plt.scatter(ball.state_estimate[0], ball.state_estimate[1], c=ball.color, marker="x", label=f"Approx. Ball #{i}")
			# Plot real state of all balls
			for i, ball in enumerate(all_balls, start=1):
				plt.scatter(ball.state[0], ball.state[1], marker="o", facecolors='none', edgecolors=ball.color, label=f"G.T. Ball #{i}")
			# Plot measurements
			for m in measurements:
				plt.scatter(m[0], m[1], marker="*", c='r')

			plt.pause(delta_t/100)
		
		for ball in all_balls:
			ball.propagate_simulation(t+delta_t)
		
	if target_map is not None:
		for est, sim in target_map.items():
			re, ce = balls[est].estimate_traj.shape
			rs, cs = balls[sim].state_traj.shape
			balls[est].errs = np.abs(balls[est].estimate_traj - balls[sim].state_traj[rs-re:,])

	if not noplot:
		plt.clf()
		plt.grid()
		plt.xlabel("X [m]")
		plt.ylabel("Y [m]")
		for i, ball in enumerate(balls, start=1):
			traj = np.array(ball.state_traj)
			plt.scatter(traj[:,0], traj[:,1], marker="o", c=ball.color, label=f"Ball #{i}", s=12)
		plt.legend()

		for i, ball in enumerate(balls):
			plt.figure(i + 2)
			plot_error(ball)
			plt.title(f"Error for ball #{i + 1}")
		plt.show()

	display_results(balls)

