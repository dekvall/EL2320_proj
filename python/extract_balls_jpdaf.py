#!/usr/bin/env python3
import cv2

import numpy as np
import matplotlib.pyplot as plt
from operator import add
from simulate_and_estimate import *

#cap = cv2.VideoCapture('../resx/small/2x1-short-small.mp4')
#cap = cv2.VideoCapture('../resx/small/4-balls-low-small.mp4')
cap = cv2.VideoCapture('../resources/60fps/4_cut.mp4')

LOWER = (20, 50, 40)
UPPER = (100, 255, 255)

graph = []
snapshots = []

G = 9.8
DT = 1./cap.get(cv2.CAP_PROP_FPS)
BOUNCE_COEFF = 0.7 #Arbitrary
PIXEL_SCALE = .007 #Size of pixel in m for a resolution of 320x240 prev 0.007

def create_blob_params():
	# Setup SimpleBlobDetector parameters.
	params = cv2.SimpleBlobDetector_Params()

	# Change thresholds
	params.minThreshold = 10
	params.maxThreshold = 255

	params.filterByCircularity = False
	# params.minCircularity = 0.01
	params.filterByInertia = True

	params.filterByConvexity = False

	params.filterByArea = True
	params.minArea = 40
	return params


def detect_with_diff(frame, first):
	detector = cv2.SimpleBlobDetector_create(create_blob_params())

	hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	hsv_first = cv2.cvtColor(first, cv2.COLOR_BGR2HSV)


	mask = cv2.inRange(hsv_frame, LOWER, UPPER)
	masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
	masked_first = cv2.bitwise_and(first, first, mask=mask)


	diff = cv2.absdiff(masked_frame, masked_first)
	mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
	imask = mask > 1 # threshold is 1

	canvas = np.zeros_like(frame, np.uint8)
	canvas[imask] = frame[imask]

	fullgray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)

	# Blob detection with black background requires the image to be inverted
	inverted = cv2.bitwise_not(fullgray)
	keypoints = detector.detect(inverted)
	
	graph.append([])
	if keypoints:
		height, width = fullgray.shape
		for p in keypoints:		
			x, y = p.pt
			y = height - y

			x *= PIXEL_SCALE
			y *= PIXEL_SCALE
			graph[-1].append(np.array([x, y]))

		snapshots.append(fullgray)

	result = cv2.drawKeypoints(frame, keypoints, np.array([]), (255,0,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	return graph[-1], cv2.cvtColor(result, cv2.COLOR_BGR2RGB)


def main():
	first = None
	first_detection = False
	
	plot_boundaries = [0, 3, 0, 4]
	nr_of_balls = 1
	covariance_threshold = 0.5

	colors = ["k", "c", "y", "m", "b"]
	noplot = False
	while cap.isOpened():
		if not noplot:
			plt.figure(1)
			plt.xlabel("X [m]")
			plt.ylabel("Y [m]")
			ax = plt.gca()
			ax.set_xlim(plot_boundaries[0], plot_boundaries[1])
			ax.set_ylim(plot_boundaries[2], plot_boundaries[3])
			plt.grid()

		ret, frame = cap.read()
		if not ret or cv2.waitKey(1) & 0xFF == ord('q'):
			break

		if first is None:
			first = frame
			continue

		detection, diff_result = detect_with_diff(frame, first)
		if not first_detection and detection is not None:
			first_detection = True
			R = 0.2**2 * np.eye(2)
			P = 0.3 **2 * np.eye(4)
			balls = [init_ball([0, 0, 0, 0], R, P, 'g', plot_boundaries+[10, 10])]

		if first_detection:
			plt.figure(1)
	
			if not noplot:
				for m in detection:
					plt.scatter(m[0], m[1], marker="*", c='r')
	
			gate_size = True
			if detection is not None:
				if balls[-1].has_converged(covariance_threshold):
					balls[-1].gate_size = 0.5
					if len(balls) < nr_of_balls:
						print("Ball nr ", len(balls) + 1, "Has converged, Initializing next ball")
						c = colors.pop()
						new_ball = init_ball([0, 0, 0, 0], R, P, c, plot_boundaries+[10, 10])						
						new_ball.remove_particles(balls)
						balls.append(new_ball)

			feasible_events = feasible_association_events(detection, balls, gate_size)
			for i, ball in enumerate(balls, start=1):
				# Filter
				if i == 1:
					print(ball.state_estimate)
				beta, obs_error_dict = calc_beta(detection, i-1, balls, feasible_events)
				ball.estimate(detection, i-1, beta, obs_error_dict)
				ball.predict(DT)
				if not noplot:
					plt.scatter(ball.state_estimate[0], ball.state_estimate[1], c=ball.color, marker="x", label=f"Approx. Ball #{i}")
					plt.scatter(ball.particles[:,0], ball.particles[:,1], marker=".", c='r', alpha=.1, label="Particle")

					# plt.pause(DT/10)
				# Ground truth

		# true_pos = zip(*graph)
		# plt.figure(1)
		# plt.ylim(0, 1.3)
		# plt.xlim(0,2)
		# plt.scatter(*true_pos, c='b')

		# plt.figure(2)
		# plt.imshow(diff_result)
		plt.pause(.0001)
		plt.clf()
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


	cap.release()
	cv2.destroyAllWindows()

	base = snapshots[0]
	for img in snapshots:
		base = cv2.bitwise_or(base, img)

	plt.imshow(base, cmap='gray')
	plt.show()

if __name__ == "__main__":
	main()