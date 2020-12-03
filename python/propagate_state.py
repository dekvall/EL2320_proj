#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

points = []
def propagate_point(t_start: float, t_end: float, state: list, dt: float =.05) -> list:
	"""
	in:
	t_start: Simulation start time
	t_end: Simulation end time
	x: Starting state: [x, y, vx, vy]

	return
	propagated state at t_end
	"""
	G = 9.82
	BOUNCE_COEFF = .84 # Tennis balls

	#trajectory start
	xt = state
	tt = t_start

	t = t_start

	while t < t_end - 1e-16:
		dt = min(dt, t_end - t)
		t += dt

		x, y, vx, vy = state

		root = vy**2 + 2 * G * y

		if root < 0:
			dT = 0
		else:
			dT = (vy + root**.5) / G

		# Never go back in time
		dT = max(0, dT)

		if dT < dt:
			# Bounce
			vy = -BOUNCE_COEFF * (vy - G * dT)
			dT = dt - dT # Bounce time
			y = 0 + vy * dT - .5 * G * dT**2
			vy = vy - G * dT # update velocity for remainder f step
		else:
			# No bounce
			y = y + vy * dt - .5 * G * dt**2
			vy = vy - G * dt

		x = x + vx * dt
		vx = vx

		state = np.array([x, y, vx, vy])
		points.append(state)
	return state




if __name__ == "__main__":
	points = []
	x = np.array([0,2,.4, 0])
	xn = propagate_point(0, 2, x)


	rp = np.array(points)

	plt.plot(rp[:,0], rp[:,1])
	plt.show()

