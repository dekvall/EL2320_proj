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
	R = 0.1**2 * np.eye(2)
	P = 0.1 **2 * np.eye(4)
	boundaries = [-1, 10, 0, 15, 10, 10]	
	balls = [init_ball([0, 10, 2, -3], R, P, 'g', boundaries)]
	R = 0.1**2 * np.eye(2)
	meas_obj = MeasurementObject(R, boundaries[:4], 0.7, 0.2)
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


def remove_particles():
	n = 20000
	rad = 2
	x = np.random.uniform(-10, 10, n)
	y = np.random.uniform(-10, 10, n)
	tot = np.array([x, y]).T
	print(tot.shape)
	x_p = 3
	y_p = 0

	bounds = np.array([[5, 3], [0, 0], [-2, -5]])
	for r in bounds:
		dist = (tot[:,0]-r[0])**2+(tot[:,1]-r[1])**2
		tot = np.delete(tot, dist < rad, 0)
	print(tot.shape[0])
	
	while tot.shape[0] < n:
		x = np.random.uniform(-10, 10, 1)
		y = np.random.uniform(-10, 10, 1)

		dist = (bounds[:,0]-x[0])**2+(bounds[:,1]-y[0])**2
		if (dist > rad).all():
			tot = np.vstack((tot, np.array([x[0], y[0]])))

	a = np.empty([0,2])
	a = np.vstack((a, np.array([1,2])))
	print(a)
	plt.scatter(tot[:,0],tot[:,1])
	plt.show()




if __name__ == "__main__":
	# remove_particles()
	TEST6()