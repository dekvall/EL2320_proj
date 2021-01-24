#!/usr/bin/env python3
import cv2

import numpy as np
import matplotlib.pyplot as plt
from simulate_and_estimate import *
from parameters import R, P

cap = cv2.VideoCapture('../raw_data_4_balls.mp4')

LOWER = (30, 50, 40)
UPPER = (100, 255, 255)

graph = []

DT = 1./cap.get(cv2.CAP_PROP_FPS)
PIXEL_SCALE = .007 # Size of pixel in m for a resolution of 320x240
Y_ADJUST = 0.3 # Offset ground level

FRAME_WIDTH = int(cap.get(3))
FRAME_HEIGHT = int(cap.get(4))

COLOR_MAP = {'g': (0, 255, 0), 'b': (0, 0, 255), 'm': (255, 132, 132), 'y': (255, 255, 0)}
def create_blob_params():
	# Setup Blob detection parameters.
	params = cv2.SimpleBlobDetector_Params()

	params.minThreshold = 30
	params.maxThreshold = 255

	params.filterByCircularity = False
	params.filterByInertia = True

	params.filterByConvexity = False

	params.filterByArea = True
	params.minArea = 80
	return params


def detect_with_diff(frame, first):
	# Detect balls using a blob detector
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
			y = y * PIXEL_SCALE + Y_ADJUST
			graph[-1].append(np.array([x, y]))

	result = cv2.drawKeypoints(frame, keypoints, np.array([]), (255,0,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	return graph[-1], cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

def add_marker_to_img(img, pos, color):
	height, width, _ = img.shape
	x = int(pos[0] / PIXEL_SCALE)
	y = height - int(((pos[1] - Y_ADJUST) / PIXEL_SCALE))
	if color in COLOR_MAP:
		rgb_color = COLOR_MAP[color]
	else:
		rgb_color = (255, 0, 0)
	return cv2.drawMarker(img, (x, y), rgb_color, 1, 20, 5)

def main():
	real_plot = False # Set this to true to see the results on the image frame, Algorithm will run slower
	first_frame = None
	first_detection = False
	
	plot_boundaries = [0, 8, 0, 5]
	nr_of_balls = 4
	covariance_threshold = 0.4

	colors = ["k", "c", "y", "m", "b"]
	noplot = False

	plt.figure(1)
	plt.grid()

	meas_lable = "Measurements"
	while cap.isOpened():
		if not noplot:
			plt.figure(1)
			plt.xlabel("X [m]")
			plt.ylabel("Y [m]")
			ax = plt.gca()
			ax.set_xlim(plot_boundaries[0], plot_boundaries[1])
			ax.set_ylim(plot_boundaries[2], plot_boundaries[3])
			ax.legend()		

		ret, frame = cap.read()
		if not ret or cv2.waitKey(1) & 0xFF == ord('q'):
			break

		if first_frame is None:
			first_frame = frame
			continue

		detection, diff_result = detect_with_diff(frame, first_frame)
		# Initialize first ball
		if not first_detection and detection is not None:
			first_detection = True
			balls = [init_ball([0, 0, 0, 0], R, P, 'g', plot_boundaries+[10, 10])]
			balls[0].label = "Ball #1 estimate"

		if first_detection:
			plt.figure(1)
	
			if not noplot:
				for m in detection:
					plt.scatter(m[0], m[1], marker="x", c='r', label=meas_lable, s=10)
				meas_lable = ""
	
			gate_size = True
			# Initialize next ball when previous ball reaches covariance threshold
			if detection is not None:
				if balls[-1].has_converged(covariance_threshold):
					balls[-1].gate_size = 1
					if len(balls) < nr_of_balls:
						print("Ball nr ", len(balls), "Has converged")
						c = colors.pop()
						new_ball = init_ball([0, 0, 0, 0], R, P, c, plot_boundaries+[10, 10])
						new_ball.label = f"Ball #{len(balls)+1} estimate"
						new_ball.remove_particles(balls)
						balls.append(new_ball)

			feasible_events = feasible_association_events(detection, balls, gate_size)
			for i, ball in enumerate(balls, start=1):
				beta, obs_error_dict = calc_beta(detection, i-1, balls, feasible_events)
				ball.estimate(detection, i-1, beta, obs_error_dict)

				if ball.has_converged(covariance_threshold) and real_plot:
					diff_result = add_marker_to_img(diff_result, (ball.state_estimate[0], ball.state_estimate[1]), ball.color)
				ball.predict(DT)

				if not noplot and ball.gate_size == 1:
					plt.scatter(ball.state_estimate[0], ball.state_estimate[1], c=ball.color, marker="o", label=ball.label, alpha=.6)
					ball.label = ""

		if real_plot:
			plt.figure(2)
			plt.imshow(diff_result)
		
		plt.pause(.0001)

	cap.release() 

if __name__ == "__main__":
	main()