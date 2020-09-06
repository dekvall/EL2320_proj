#!/usr/bin/env python3
import cv2

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from operator import add

#cap = cv2.VideoCapture('../resx/small/2x1-short-small.mp4')
#cap = cv2.VideoCapture('../resx/small/4-balls-low-small.mp4')
cap = cv2.VideoCapture('../resources/small/ex1-small.mp4')

LOWER = (30, 50, 50)
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

	# Blob detection with black background requires the image to be inverted
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

		graph.append((x, y))
		snapshots.append(fullgray)

	result = cv2.drawKeypoints(frame, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	return result

	# cv2.imshow('diff',fullgray)
	# cv2.waitKey(50)

while cap.isOpened():
	ret, frame = cap.read()
	if not ret or cv2.waitKey(1) & 0xFF == ord('q'):
		break

	if first is None:
		first = frame
		continue

	diff_result = detect_with_diff(frame, first)


	#cv2.imshow('Original',frame)
	#v2.imshow('Blob/diff detect', diff_result)
	#cv2.waitKey(50)


cap.release()
cv2.destroyAllWindows()

sns.scatterplot(*zip(*graph))
plt.show()

base = snapshots[0]
for img in snapshots:
	base = cv2.bitwise_or(base, img)

plt.imshow(base, cmap='gray')
plt.show()