#!/usr/bin/env python3
import cv2

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from operator import add

#cap = cv2.VideoCapture('../resx/small/2x1-short-small.mp4')
cap = cv2.VideoCapture('../resx/small/4-balls-low-small.mp4')

LOWER = (30, 100, 100)
UPPER = (60, 255, 255)

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

	if keypoints:
		height, width = gray.shape
		x, y = keypoints[0].pt
		y = height - y

		graph.append((x, y))
		snapshots.append(masked)

	result = cv2.drawKeypoints(frame, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	return result

def detect_with_hough(frame, first):
	diff = cv2.absdiff(fullgray, first)
	cv2.imshow('diff',diff)
	cv2.waitKey(50)
	circles = cv2.HoughCircles(diff, cv2.HOUGH_GRADIENT, .3, minDist=5,
							param1=200, param2=40, minRadius=7, maxRadius=10)

	if circles is None:
		return frame

	circles = np.uint16(np.around(circles))
	for i in circles[0,:]:
		cv2.circle(frame,(i[0],i[1]),i[2],(0,255,0),2)
	return frame


while cap.isOpened():
	ret, frame = cap.read()
	if not ret or cv2.waitKey(1) & 0xFF == ord('q'):
		break

	fullgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	if first is None:
		first = fullgray
		continue

	result = detect_with_hough(fullgray, first)
	res2 = detect_with_blob(frame, first)


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

