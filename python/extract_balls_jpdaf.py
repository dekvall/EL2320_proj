#!/usr/bin/env python3
import cv2

import numpy as np
import matplotlib.pyplot as plt
from operator import add

#cap = cv2.VideoCapture('../resx/small/2x1-short-small.mp4')
#cap = cv2.VideoCapture('../resx/small/4-balls-low-small.mp4')
cap = cv2.VideoCapture('../resources/60fps/1_cut.mp4')

LOWER = (20, 50, 40)
UPPER = (100, 255, 255)

graph = []
kf_graph = []
kf_graph_corrected = []
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

	if keypoints:
		height, width = fullgray.shape
		x, y = keypoints[0].pt
		y = height - y

		x *= PIXEL_SCALE
		y *= PIXEL_SCALE

		graph.append((x, y))
		snapshots.append(fullgray)

	result = cv2.drawKeypoints(frame, keypoints, np.array([]), (255,0,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	return graph[-1] if keypoints else None, cv2.cvtColor(result, cv2.COLOR_BGR2RGB)



while cap.isOpened():
	ret, frame = cap.read()
	if not ret or cv2.waitKey(1) & 0xFF == ord('q'):
		break

	if first is None:
		first = frame
		continue

	detection, diff_result = detect_with_diff(frame, first)
	true_pos = zip(*graph)
	# plt.figure(1)
	# plt.ylim(0, 1.3)
	# plt.xlim(0,2)
	# plt.scatter(*true_pos, c='b')

	plt.figure(2)
	plt.imshow(diff_result)
	plt.pause(.01)
	plt.clf()


cap.release()
cv2.destroyAllWindows()

base = snapshots[0]
for img in snapshots:
	base = cv2.bitwise_or(base, img)

plt.imshow(base, cmap='gray')
plt.show()