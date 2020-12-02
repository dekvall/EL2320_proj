import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import random

G = 9.8
SIM_TIME = 10
DELTA_T = 0.1
BOUNCE_COEFF = 0.84
DETECTION_PROB = 1
MSR_COVAR = 0.05



class SimulateBall():
	def __init__(self, x0, y0, vx, vy):
		self.x = x0
		self.y = y0
		self.vx = vx
		self.vy = vy
		self.time = 0
	
	def next_step(self):
		self.x += DELTA_T * self.vx

		if self.y + self.vy * DELTA_T - 0.5 * G * (DELTA_T**2) < 0:
			pre_t =  self.vy/ G  + math.sqrt((self.vy**2) / (G**2) + (2 / G) * self.y)
			pre_vy = (self.vy - 9.8 * pre_t)*BOUNCE_COEFF
			self.y = (-pre_vy) * (DELTA_T - pre_t) - 0.5 * G * ((DELTA_T - pre_t) **2)
			
			if self.y < 0:
				self.y = 0
				self.vy = 0
			else:
				self.vy = -pre_vy - 9.8 * (DELTA_T - pre_t)
		
		else:
			self.y += DELTA_T * self.vy - 0.5 * G * (DELTA_T**2)
			self.vy -= 9.8 * DELTA_T

		self.time += DELTA_T

	def get_measurement(self, detection_prob, covariance):
		if detection_prob > random.random():
			x = self.x + random.normalvariate(0, covariance)
			y = self.y + random.normalvariate(0, covariance)
		else:
			x = None
			y = None

		return x, y


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

	kf.measurementNoiseCov = 1e-5 * np.eye(2)

	#Belief in initial state
	kf.errorCovPost = 1. * np.ones((4,4))

	return kf


if __name__ =="__main__":
	ball = SimulateBall(10., 10., 1., 0.)

	ground_truth = []
	coords = []
	kf_graph = []

	kf = init_kalman_filter()
	const_mat = np.array([[-1/2*G*DELTA_T**2], [-G*DELTA_T]])

	first_detection = True
	
	i = 0
	last_bounce = 0
	while ball.time < SIM_TIME:

		x, y = ball.get_measurement(DETECTION_PROB, MSR_COVAR)
		if x != None: 
			if first_detection:
				first_detection = False
				kf.statePost = np.array([[x],
										[y],
										[1.],
										[0.]],
										dtype='float64')
			else:
				measurements = np.array([[x],
										[y]],
										dtype="float64")
				kf.correct(measurements)
				coords.append((x, y))

		x_pred, y_pred = kf.statePost.ravel()[:2]

		if not first_detection:
			if y_pred < 0 and i > last_bounce + 10:
				print("bounce")
				last_bounce = i
				kf.transitionMatrix[3,3] = -1*BOUNCE_COEFF

			kf.predict(const_mat)

			# Restore to normal model
			kf.transitionMatrix[3,3] = 1

			x_pred, y_pred = kf.statePost.ravel()[:2]
			kf_graph.append((x_pred, y_pred))

		i += 1
		ground_truth.append((ball.x, ball.y))
		ball.next_step()


	plt.scatter(*zip(*coords))
	plt.plot(*zip(*kf_graph), 'ro-')
	plt.plot(*zip(*ground_truth), 'b-')

	plt.show()


