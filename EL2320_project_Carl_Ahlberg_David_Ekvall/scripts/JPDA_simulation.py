from simulate_and_estimate import init_ball, MeasurementObject, monte_carlo_jpdaf_simulation
from itertools import permutations
import numpy as np


def TEST1():
	R = 0.3**2 * np.eye(2) # Measurement noise
	P = 0.2 **2 * np.eye(4) # Process noise
	boundaries = [-1, 15, 0, 10, 10, 10] # [x_min, x_max, y_min, y_max, vx, vy]
	balls = [init_ball([0, 3, 2, -6], R, P, 'g', boundaries, True), # Initialization of all balls
			init_ball([8, 5, -1, -1], R, P, 'b', boundaries, True),
			init_ball([0, 7, 4, -2], R, P, 'r', boundaries, True)]

	R_sim = 0.3**2 * np.eye(2) # Simulation measurement noise
	P_D_SIM = 0.9 # Probability producing a ball measurement
	P_FP_SIM = 0.2 # Probability of producing a outlier
	meas_obj = MeasurementObject(R_sim, boundaries[:4], P_D_SIM, P_FP_SIM) 
	delta_t = .05 # Simulation time step
	time = 5  # Total simulation time
	target_map = {0:2, 1:0, 2:1} # Optional map to pair estimated balls (keys) to simulated balls (values)
	monte_carlo_jpdaf_simulation(sim_time=time, \
								delta_t=delta_t, \
								all_balls=balls, \
								meas_obj=meas_obj, \
								plot_boundaries=meas_obj.boundaries, \
								noplot=False, \
								target_map=target_map)


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
	balls = [init_ball([0, 10, 2, -7], R, P, 'g', boundaries)]
	
	R = 0.2**2 * np.eye(2)
	meas_obj = MeasurementObject(R, boundaries[:4])
	delta_t = .05
	time = 0.2
	monte_carlo_jpdaf_simulation(sim_time=time, \
								delta_t=delta_t, \
								all_balls=balls, \
								meas_obj=meas_obj, \
								plot_boundaries=meas_obj.boundaries, \
								noplot=True)


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
	
	R = 0.1**2 * np.eye(2)
	meas_obj = MeasurementObject(R, boundaries[:4])
	delta_t = 0.05
	time = 5
	target_map = {0:1, 1:0}
	monte_carlo_jpdaf_simulation(sim_time=time, \
								delta_t=delta_t, \
								all_balls=balls, \
								meas_obj=meas_obj, \
								plot_boundaries=meas_obj.boundaries, \
								noplot=False,
								target_map=target_map)

def TEST6():
	R = 0.2**2 * np.eye(2)
	P = 0.2**2 * np.eye(4)
	boundaries = [-1, 10, 0, 10, 10, 10]	
	balls = [init_ball([0, 10, 2, -1], R, P, 'g', boundaries),
			init_ball([5, 10, -2, -1], R, P, 'b', boundaries),
			init_ball([7, 0, -2, 10], R, P, 'm', boundaries)]
	
	R = 0.1**2 * np.eye(2)
	meas_obj = MeasurementObject(R, boundaries[:4], 0.9, 0.1)
	delta_t = .05
	time = 1
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

# Choose which test too run
if __name__ == "__main__":
	TEST1()
