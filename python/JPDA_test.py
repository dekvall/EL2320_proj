from simulate_and_estimate import init_ball, MeasurementObject, monte_carlo_jpdaf_simulation
from itertools import permutations
import numpy as np
import matplotlib.pyplot as plt


def TEST1():
	R = 0.2**2 * np.eye(2)
	P = 0.2 **2 * np.eye(4)
	boundaries = [-1, 10, 0, 10, 10, 10] # [x_min, x_max, y_min, y_max, vx, vy]
	balls = [init_ball([0, 3, 2, -6], R, P, 'g', boundaries),
			init_ball([8, 5, -1, -1], R, P, 'b', boundaries),
			init_ball([0, 7, 4, -2], R, P, 'y', boundaries)]

	R = 0.3**2 * np.eye(2)
	meas_obj = MeasurementObject(R, boundaries[:4],1, 0.2)
	delta_t = .05
	time = 5
	monte_carlo_jpdaf_simulation(sim_time=time, \
								delta_t=delta_t, \
								all_balls=balls, \
								meas_obj=meas_obj, \
								plot_boundaries=meas_obj.boundaries, \
								noplot=False)


def TEST2():
	R = 0.2**2 * np.eye(2)
	P = 0.2 **2 * np.eye(4)
	boundaries = [-1, 10, 0, 15, 10, 10]	
	balls = [init_ball([0, 10, 2, -3], R, P, 'g', boundaries, False),
			init_ball([0, 10, 2, -3], R, P, 'b', boundaries, True)]
	R = 0.1**2 * np.eye(2)
	meas_obj = MeasurementObject(R, boundaries[:4])
	delta_t = .05
	time = 5
	monte_carlo_jpdaf_simulation(sim_time=time, \
								delta_t=delta_t, \
								all_balls=balls, \
								meas_obj=meas_obj, \
								plot_boundaries=meas_obj.boundaries, \
								noplot=False)


def TEST3():
	R = 0.2**2 * np.eye(2)
	P = 0.2**2 * np.eye(4)
	boundaries = [-1, 10, 0, 10, 10, 10]	
	balls = [init_ball([0, 10, 2, -7], R, P, 'g', boundaries),
			init_ball([0, 5, 1, -3], R, P, 'b', boundaries)]
	
	R = 0.2**2 * np.eye(2)
	meas_obj = MeasurementObject(R, boundaries[:4])
	delta_t = .1
	time = 5
	monte_carlo_jpdaf_simulation(sim_time=time, \
								delta_t=delta_t, \
								all_balls=balls, \
								meas_obj=meas_obj, \
								plot_boundaries=meas_obj.boundaries, \
								noplot=False)


def TEST4():
	R = 0.2**2 * np.eye(2)
	P = 0.15**2 * np.eye(4)
	boundaries = [-1, 4, 0, 4, 10, 12]	
	balls = [init_ball([0, 1, 0.5, -8], R, P, 'g', boundaries, 0.5, 1.5),
			init_ball([0, 1.5, 0.7, -6], R, P, 'm', boundaries, 0, 1)]	
	R = 0.1**2 * np.eye(2)
	meas_obj = MeasurementObject(R, boundaries[:4], 0.9)
	delta_t = 1/40.
	time = 7
	monte_carlo_jpdaf_simulation(sim_time=time, \
								delta_t=delta_t, \
								all_balls=balls, \
								meas_obj=meas_obj, \
								plot_boundaries=meas_obj.boundaries, \
								noplot=False)


def TEST5():
	R = 0.2**2 * np.eye(2)
	P = 0.2**2 * np.eye(4)
	boundaries = [-1, 10, 0, 10, 10, 10]	
	balls = [init_ball([0, 10, 2, -7], R, P, 'g', boundaries),
			init_ball([8, 5, -1, -3], R, P, 'b', boundaries)]
	
	R = 0.2**2 * np.eye(2)
	meas_obj = MeasurementObject(R, boundaries[:4])
	delta_t = .1
	time = 5
	monte_carlo_jpdaf_simulation(sim_time=time, \
								delta_t=delta_t, \
								all_balls=balls, \
								meas_obj=meas_obj, \
								plot_boundaries=meas_obj.boundaries, \
								noplot=False)

def TEST6():
	R = 0.1**2 * np.eye(2)
	P = 0.2**2 * np.eye(4)
	boundaries = [-1, 10, 0, 10, 10, 10]	
	balls = [init_ball([0, 10, 2, -1], R, P, 'g', boundaries),
			init_ball([5, 10, -2, -1], R, P, 'b', boundaries),
			init_ball([7, 0, -2, 10], R, P, 'm', boundaries)]
	
	R = 0.1**2 * np.eye(2)
	meas_obj = MeasurementObject(R, boundaries[:4], 0.9, 0.1)
	delta_t = .05
	time = 5
	monte_carlo_jpdaf_simulation(sim_time=time, \
								delta_t=delta_t, \
								all_balls=balls, \
								meas_obj=meas_obj, \
								plot_boundaries=meas_obj.boundaries, \
								noplot=False)


def TEST7():
	R = 0.2**2 * np.eye(2)
	P = 0.25**2 * np.eye(4)
	boundaries = [-1, 4, 0, 4, 10, 10]	
	balls = [init_ball([4, 2, -3, -8], R, P, 'g', boundaries, 0.25),
			init_ball([3.2, 2, -0.2, -8], R, P, 'b', boundaries, 0, 0.3)]	

	R = 0.03**2 * np.eye(2)
	meas_obj = MeasurementObject(R, boundaries[:4], 0.95)
	delta_t = 1/60
	time = 5
	monte_carlo_jpdaf_simulation(sim_time=time, \
								delta_t=delta_t, \
								all_balls=balls, \
								meas_obj=meas_obj, \
								plot_boundaries=meas_obj.boundaries, \
								noplot=False)




if __name__ == "__main__":
	# remove_particles()
	TEST2()