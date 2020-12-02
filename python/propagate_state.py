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
	BOUNCE_COEFF = 0.84 # Tennis balls

	#trajectory start
	xt = state
	tt = t_start

	t = t_start

	while t < t_end:
		dt = min(dt, t_end - t_start)
		t += dt

		x, y, vx, vy = state

		root = vy**2 + 2 * G * y

		if root < 0:
			dT = 0
		else:
			dT = (vy + root**.5) / G

		dT = max(0, dT)

		if dT < dt:
			# Bounce
			vyn = - BOUNCE_COEFF * (vy - G * dT)
			dT = dt - dT # Bounce time
			yn = 0 + vyn * dT - .5 * G * dT**2
			vyn = vyn - 9.81 * dT # update velocity for remainder f step


		else:
			# No bounce
			yn = y + vy + dt - .5 * G * dt**2
			vyn = vy - G * dt

		xn = x + vx * dt
		vxn = vx

		state = np.array([xn, yn, vxn, vyn])
		points.append(state)
	return state




if __name__ == "__main__":
	points = []
	x = np.array([2,2,.4,0])
	xn = propagate_point(0, 2, x)


	rp = np.array(points)

	plt.plot(rp[:,0], rp[:,1])
	plt.show()

