#!/usr/bin/env python3
import cv2

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from operator import add

#cap = cv2.VideoCapture('../resx/small/2x1-short-small.mp4')
#cap = cv2.VideoCapture('../resx/small/4-balls-low-small.mp4')
cap = cv2.VideoCapture('../resources/small/ex6-one-small.mp4')

LOWER = (30, 50, 50)
UPPER = (60, 255, 255)

graph = []
kf_graph = []
snapshots = []
first = None

G = 9.8
DT = 1/30
BOUNCE_COEFF = 0.7 #Arbitrary
PIXEL_SCALE = .007 #Size of pixel in m for a resolution of 320x240

def create_blob_params():
	# Setup SimpleBlobDetector parameters.
	params = cv2.SimpleBlobDetector_Params()

	# Change thresholds
	params.minThreshold = 1
	params.maxThreshold = 255


	return params

def init_kalman_filter():
	kf = cv2.KalmanFilter(4, 2, 2)
	kf.transitionMatrix = np.array([[1, 0, DT, 0],
									[0, 1, 0, DT],
									[0, 0, 1, 0],
									[0, 0, 0, 1]])
	
	kf.controlMatrix = np.array([[0, 0],
								[1., 0],
								[0, 0],
								[0, 1.]])
	
	kf.measurementMatrix = np.array([[1., 0, 0, 0],
									[0, 1., 0, 0]])

	kf.processNoiseCov = 1e-2 * np.eye(4)

	kf.measurementNoiseCov = 1e-5 * np.eye(2)

	#Belief in initial state
	kf.errorCovPost = 1. * np.ones((4,4))

	return kf

def detect_with_diff(frame, first):
	detector = cv2.SimpleBlobDetector_create()

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

	if keypoints:
		height, width = fullgray.shape
		x, y = keypoints[0].pt
		y = height - y

		x *= PIXEL_SCALE
		y *= PIXEL_SCALE

		graph.append((x, y))
		snapshots.append(fullgray)

	result = cv2.drawKeypoints(frame, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	return graph[-1] if keypoints else None, result


# Kalman initzialisation
kf = init_kalman_filter()
const_mat = np.array([[-1/2*G*DT**2], [-G*DT]])

first_detection = True

while cap.isOpened():
	ret, frame = cap.read()
	if not ret or cv2.waitKey(1) & 0xFF == ord('q'):
		break

	if first is None:
		first = frame
		continue

	detection, diff_result = detect_with_diff(frame, first)

	if detection is not None:
		x, y = detection
		print(x, y)
		if first_detection:
			first_detection = False
			print("Init kalman filter")
			kf.statePost = np.array([[x],
									[y],
									[.2],
									[-1]],
									dtype='float64')
		else:
			print("Ball detected")
			measurements = np.array([[x],
									[y]],
									dtype="float64")
			kf.correct(measurements)

	x_pred, y_pred = kf.statePost.ravel()[:2]

	if not first_detection:
		if y_pred < .2:
			print("bounce")
			kf.transitionMatrix[3,3] = -1*BOUNCE_COEFF

		kf.predict(const_mat)

		# Restore to normal model
		kf.transitionMatrix[3,3] = 1

		x_pred, y_pred = kf.statePost.ravel()[:2]
		kf_graph.append((x_pred, y_pred))


	#cv2.imshow('Original',frame)
	#cv2.imshow('Blob/diff detect', diff_result)
	#cv2.waitKey(50)


cap.release()
cv2.destroyAllWindows()

true_pos = zip(*graph)
kalman_pos = zip(*kf_graph)

plt.scatter(*true_pos)
plt.plot(*kalman_pos, 'ro-')
plt.show()

base = snapshots[0]
for img in snapshots:
	base = cv2.bitwise_or(base, img)

plt.imshow(base, cmap='gray')
plt.show()