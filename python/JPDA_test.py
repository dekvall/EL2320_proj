from simulate_and_estimate import init_ball, MeasurementObject, monte_carlo_jpdaf_simulation
from itertools import permutations
import numpy as np

def TEST1():
	R = 0.2**2 * np.eye(2)
	P = 0.1 **2 * np.eye(4)
	# Base colors available: gbcmyk
	balls = [init_ball([0, 3, 2, -6], R, P, 'g'),
			init_ball([10, 5, -1, -1], R, P, 'b'),
			init_ball([0, 7, 4, -2], R, P, 'y')]

	meas_obj = MeasurementObject(R, np.array([0, 10, 0, 10]))
	delta_t = .1
	time = 3
	monte_carlo_jpdaf_simulation(sim_time=time, \
								delta_t=delta_t, \
								balls=balls, \
								meas_obj=meas_obj, \
								plot_boundaries=meas_obj.boundaries, \
								noplot=False)


def TEST2():
	R = 0.3**2 * np.eye(2)
	P = 0.2 **2 * np.eye(4)
	# Base colors available: gbcmyk
	balls = [init_ball([0, 10, 2, -7], R, P, 'g', 1.5, 3)]
	R = 0.2**2 * np.eye(2)
	meas_obj = MeasurementObject(R, np.array([-1, 10, 0, 10]))
	delta_t = .1
	time = 5
	monte_carlo_jpdaf_simulation(sim_time=time, \
								delta_t=delta_t, \
								balls=balls, \
								meas_obj=meas_obj, \
								plot_boundaries=meas_obj.boundaries, \
								noplot=False)


def TEST3():
	R = 0.3**2 * np.eye(2)
	P = 0.2**2 * np.eye(4)
	# Base colors available: gbcmyk
	balls = [init_ball([0, 10, 2, -7], R, P, 'g', 1.5, 3),
			init_ball([0, 5, 1, -3], R, P, 'b')]
	
	R = 0.1**2 * np.eye(2)
	meas_obj = MeasurementObject(R, np.array([-1, 10, 0, 10]))
	delta_t = .1
	time = 5
	monte_carlo_jpdaf_simulation(sim_time=time, \
								delta_t=delta_t, \
								balls=balls, \
								meas_obj=meas_obj, \
								plot_boundaries=meas_obj.boundaries, \
								noplot=False)


if __name__ == "__main__":
	TEST3()