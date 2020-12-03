#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

def propagate_state(t_start: float, t_end: float, state: list, dt: float =.05) -> list:
	"""
	in:
	t_start: Simulation start time
	t_end: Simulation end time
	x: Starting state: [x, y, vx, vy]

	return
	propagated state at t_end
	state- and time-trajectory
	"""
	G = 9.82
	BOUNCE_COEFF = .84 # Tennis balls

	#trajectory start
	state_traj = [state]
	time_traj = [t_start]

	t = t_start

	while t < t_end:
		dt = min(dt, t_end - t)
		t += dt

		x, y, vx, vy = state


		tree = vy**2 + 2 * G * y

		if tree < 0:
			dT = 0
		else:
			dT = (vy + tree**.5) / G

		# Never go back in time
		dT = max(0, dT)

		if dT < dt:
			# Bounce
			vy = -BOUNCE_COEFF * (vy - G * dT)

			# Save bounce state
			state_traj.append([x + vx * dT, 0, vx, vy])
			time_traj.append(t - dt + dT)

			dT = dt - dT # Bounce time
			y = 0 + vy * dT - .5 * G * dT**2
			vy = vy - G * dT # update velocity for remainder f step
		else:
			# No bounce
			y = y + vy * dt - .5 * G * dt**2
			vy = vy - G * dt

		x = x + vx * dt
		vx = 0.98 * vx #Approximate drag

		state = [x, y, vx, vy]
		state_traj.append(state)
		time_traj.append(t)
	return np.array(state), np.array(state_traj), np.array(time_traj)


if __name__ == "__main__":
	x = np.array([0, 1, .4, 0])
	x, xt, tt = propagate_point(0, 6, x)

	plt.plot(xt[:,0], xt[:,1])
	plt.axhline(y=0, color='r', linestyle='-')
	plt.show()
