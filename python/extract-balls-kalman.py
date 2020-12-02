#!/usr/bin/env python3
import cv2

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from operator import add

#cap = cv2.VideoCapture('../resx/small/2x1-short-small.mp4')
cap = cv2.VideoCapture('../resx/1-long.mp4')

PIXEL_SCALE = 0.07/10 #Size of pixel in m for a resolution of 320x240
G = 9.8
DELTA_T = 1/30
BOUNCE_COEFF = 0.7 #Arbitrary

LOWER = (30, 100, 100)
UPPER = (60, 255, 255)

kalman_graph = []
graph = []
snapshots = []
first = None

def create_blob_params():
	# Setup SimpleBlobDetector parameters.
	params = cv2.SimpleBlobDetector_Params()

	# Change thresholds
	params.minThreshold = 1
	params.maxThreshold = 255


	return params

def detect_with_blob(frame, first):
	detector = cv2.SimpleBlobDetector_create()
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	mask = cv2.inRange(hsv, LOWER, UPPER)
	masked = cv2.bitwise_and(frame, frame, mask=mask)
	gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)

	# Blob detection with black backgreound requires the image to be inverted
	inverted = cv2.bitwise_not(gray)
	keypoints = detector.detect(inverted)
	success = False
	if keypoints:
		success = True
		height, width = gray.shape
		x, y = keypoints[0].pt
		y = (height - y) * PIXEL_SCALE
		x *= PIXEL_SCALE

		graph.append((x, y))
		snapshots.append(masked)

	result = cv2.drawKeypoints(frame, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	return success, result

def detect_with_hough(frame, first):
	diff = cv2.absdiff(fullgray, first)
#	cv2.imshow('diff',diff)
	cv2.waitKey(50)
	circles = cv2.HoughCircles(diff, cv2.HOUGH_GRADIENT, .3, minDist=5,
							param1=200, param2=40, minRadius=7, maxRadius=10)

	if circles is None:
		return frame

	circles = np.uint16(np.around(circles))
	for i in circles[0,:]:
		cv2.circle(frame,(i[0],i[1]),i[2],(0,255,0),2)
	return frame

def init_kalman_filter():
	kf = cv2.KalmanFilter(4, 2, 2)
	kf.transitionMatrix = np.array([[1, 0, DELTA_T, 0],
									[0, 1, 0, DELTA_T],
									[0, 0, 1, 0],
									[0, 0, 0, 1]])
	
	kf.controlMatrix = np.array([[0, 0],
								[1., 0],
								[0, 0],
								[0, 1.]])
	
	kf.measurementMatrix = np.array([[1., 0, 0, 0],
									[0, 1., 0, 0]])

	kf.processNoiseCov = 1e-2 * np.eye(4)

	kf.measurementNoiseCov = 0 * np.eye(2)

	#Belief in initial state
	kf.errorCovPost = 1. * np.ones((4,4))

	return kf


kf = init_kalman_filter()
const_mat = np.array([[-1/2*G*DELTA_T**2], [-G*DELTA_T]])

first_detection = True
while cap.isOpened():
	ret, frame = cap.read()

	if not ret or cv2.waitKey(1) & 0xFF == ord('q'):
		break

	frame = cv2.resize(frame, (320, 240))
	fullgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	if first is None:
		first = fullgray
		continue

	blob_ret, res2 = detect_with_blob(frame, first)

	if blob_ret:
		if first_detection:
			first_detection = False
			print("Initializing kalman filter")
			kf.statePost = np.array([[graph[-1][0]],
									[graph[-1][1]],
									[0.2],
									[-1]])
		else:
			print("Ball detected")
			measurements = np.array([[graph[-1][0]], [graph[-1][1]]],dtype="float64")
			kf.correct(measurements)
	
	if not first_detection:
		print("y-pos: ", kf.statePost[1][0], ", y_vel: ", kf.statePost[3][0])
		check = False
		if kf.statePost[1][0] < 0:
			print("bounce")
			check = True
			kf.transitionMatrix = np.array([[1, 0, DELTA_T, 0],
									[0, 1, 0, DELTA_T],
									[0, 0, 1, 0],
									[0, 0, 0, -1*BOUNCE_COEFF]])

		kf.predict(const_mat)
		
		if check:
			kf.transitionMatrix = np.array([[1, 0, DELTA_T, 0],
											[0, 1, 0, DELTA_T],
											[0, 0, 1, 0],
											[0, 0, 0, 1]])



	cv2.imshow('frame',fullgray)
	cv2.waitKey(50)


	cv2.imshow('frame2',res2)
	cv2.waitKey(50)

cap.release()
cv2.destroyAllWindows()

sns.scatterplot(*zip(*graph))
plt.show()

base = snapshots[0]
for img in snapshots:
	base = cv2.bitwise_or(base, img)

plt.imshow(base)
plt.show()

