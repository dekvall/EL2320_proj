#!/usr/bin/env python3
import cv2

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

cap = cv2.VideoCapture('../resx/small/2x1-short-small.mp4')

LOWER = (30, 100, 100)
UPPER = (60, 255, 255)

graph = []
snapshots = []

def create_blob_params():
	# Setup SimpleBlobDetector parameters.
	params = cv2.SimpleBlobDetector_Params()

	# Change thresholds
	params.minThreshold = 170
	params.maxThreshold = 255


	return params

detector = cv2.SimpleBlobDetector_create(create_blob_params())

while cap.isOpened():
	ret, frame = cap.read()
	if not ret or cv2.waitKey(1) & 0xFF == ord('q'):
		break

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

	result = cv2.drawKeypoints(masked, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

	cv2.imshow('frame',result)
	cv2.waitKey(50)

cap.release()
cv2.destroyAllWindows()

sns.scatterplot(*zip(*graph))
plt.show()
