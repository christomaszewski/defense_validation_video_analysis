import sys
import os
import cv2
import numpy as np
from collections import deque

class Boat(object):

	def __init__(self, position, time, identifier=0, maxHist=100):
		self._maxHistory = maxHist
		self._positions = deque([position], maxlen=self._maxHistory)
		self._times = deque([time], maxlen=self._maxHistory)

		self._id = identifier

	def drawBubble(self, img, radius=5):
		if (self._id is 0):
			# Enemy
			cv2.circle(img, self._positions[-1], radius, (0, 0, 255), -1)
		elif (self._id is 1):
			# Asset
			cv2.circle(img, self._positions[-1], radius, (0, 255, 0), -1)
		elif (self._id is 2):
			# Defender
			cv2.circle(img, self._positions[-1], radius, (255, 0, 0), -1)

	def updatePosition(self, position, time):
		self._positions.append(position)
		self._times.append(time)

	def distanceTo(self, point):
		x, y = self._positions[-1]
		newX, newY = point

		return np.linalg.norm([newX - x, newY - y])

	@property
	def isEnemy(self):
		return self._id == 0

	@property
	def isAsset(self):
		return self._id == 1

	@property
	def position(self):
		return self._positions[-1]


def getCentroid(contour):
	M = cv2.moments(contour)
	area = cv2.contourArea(contour)
	if (area == 0 or area is None):
		print("Area is really small ", area)
		return tuple(contour[0])
	
	M['m01']
	M['m10']

	cx = int(M['m10'] / area)
	cy = int(M['m01'] / area)

	return (cx, cy)





if __name__ == '__main__':
	scenario = 'line_defense'

	inputDir = '../Documents/datasets/gams/'
	inputFile = inputDir + scenario +'.MOV'

	outputDir = inputDir + scenario + '/'

	if not os.path.exists(outputDir):
		os.makedirs(outputDir)


	breached  = 0
	flanked = 0

	cap = cv2.VideoCapture(inputFile)

	frameIndex = 0

	#Set to end frame to terminate after full 360
	max_frame = 12600

	cv2.namedWindow("img", cv2.WINDOW_NORMAL)

	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	out = cv2.VideoWriter(outputDir + 'output.avi',fourcc, 30.0, (3840,2160), True)


	enemy = None
	defenders = None
	asset = None
	boats = []
	maximumDetectionDistance = 50
	protection_radius = 30
	# Skip ahead some frames to the action 
	# (initial detections are hardcoded, increasing this may break stuff)
	frameSkip = 2700;#3600
	for x in range(0, frameSkip):
		cap.grab()
		frameIndex += 1


	while (frameIndex < max_frame):
		ret, img = cap.read()

		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		cv2.medianBlur(gray, 5)
		ret, binaryThresh = cv2.threshold(gray,200, 255,cv2.THRESH_BINARY)

		"""
		adaptThresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,
											cv2.THRESH_BINARY,11,2)
		adaptGaussThresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
		 							cv2.THRESH_BINARY,11,2)
		"""

		thresh = cv2.erode(binaryThresh, None, iterations=2)
		thresh = cv2.dilate(thresh, None, iterations=2)

		"""
		fileName = outputDir + '/frame_' + str(frameIndex) + '.jpg'
		cv2.imwrite(fileName, frame)
		frameIndex += 1
		"""

		im2, contours, hierarchy = cv2.findContours(binaryThresh, cv2.RETR_TREE, 
												cv2.CHAIN_APPROX_SIMPLE)

		# Filter spurious contours
		cnts = [cnt for cnt in contours if cv2.contourArea(cnt) > 50]

		boatCentroids = []
		radii = []

		if (len(boats) < 6):
			boats = []
			defenders = []


			asset = Boat(getCentroid(cnts[0]), frameIndex, 1)

			enemy = Boat(getCentroid(cnts[5]), frameIndex, 0)
			#68: 0 1 2 3 4 -1
			#59: 0 6 2 3 4 5, 2100, 12600

			defenders.append(Boat(getCentroid(cnts[1]), frameIndex, 2))
			defenders.append(Boat(getCentroid(cnts[2]), frameIndex, 2))
			defenders.append(Boat(getCentroid(cnts[3]), frameIndex, 2))
			defenders.append(Boat(getCentroid(cnts[4]), frameIndex, 2))

			boats.append(enemy)
			boats.append(asset)
			boats.extend(defenders)

		else:
			prevBoats = boats
			remainingCnts = []

			closest = None
			closestDist = 0
			for b in boats:
				for cnt in cnts:
					dist = b.distanceTo(getCentroid(cnt))
					if (dist > maximumDetectionDistance):
						#detection is too far, skip it and go on
						remainingCnts.append(cnt)
					elif (closest is None):
						closest = cnt
						closestDist = dist
					elif (dist < closestDist):
						remainingCnts.append(closest)
						closest = cnt
						closestDist = dist
					else:
						remainingCnts.append(cnt)

				if (closest is not None):
					# Managed to associate a detection to the boat instance
					b.updatePosition(getCentroid(closest), frameIndex)

				cnts = remainingCnts
				remainingCnts = []
				closest = None
				closestDist = 0

		frameIndex += 1
		print("Frame: ", frameIndex)

		# Create layer for transparent overlay
		overlay = img.copy()


		# Find the enemy and asset boat instances
		enemy = None
		asset = None
		defenders = []
		for b in boats:
			b.drawBubble(overlay, protection_radius)
			if (b.isEnemy):
				enemy = b
			elif (b.isAsset):
				asset = b
			else:
				defenders.append(b)


		#Compute equation of line
		a = enemy.position[1] - asset.position[1]
		b = asset.position[0] - enemy.position[0]
		c = enemy.position[0]*asset.position[1] - asset.position[0]*enemy.position[1]
		c2 = -asset.position[0]*b + asset.position[1]*a
		#Compute signed distance (w/o scaling) from line to each defender
		dists = []
		v_dists = []
		min_abs_d = 1e6
		i = 0
		for d in defenders:
			dist = a*d.position[0] + b*d.position[1] + c
			dists.append( (i,dist) ) 
			vdist = abs(b*d.position[0] - a*d.position[1] + c2)
			v_dists.append( (i,vdist) ) 
			i += 1
			if abs(dist) < min_abs_d:
				min_abs_d = abs(dist)

		#Sort boats in order of increasing signed distance
		dists.sort(key=lambda tup: tup[1]) 


		#Check for internal breach
		if min_abs_d > protection_radius*np.sqrt(a*a+b*b):
			breached += 1
			colour = (0, 0, 255)
			#Check for flank
			if dists[-1][1] < 0:
				flanked += 1
		else:
			colour = (255, 255, 255)


		if v_dists[dists[1][0]][1] > v_dists[dists[0][0]][1] and\
				v_dists[dists[1][0]][1] > v_dists[dists[2][0]][1]:
			cv2.line(overlay, defenders[dists[0][0]].position, defenders[dists[1][0]].position, (0,255,0), 5)
			if v_dists[dists[2][0]][1] > v_dists[dists[3][0]][1]:
				cv2.line(overlay, defenders[dists[1][0]].position, defenders[dists[2][0]].position, (0,255,0), 5)
				cv2.line(overlay, defenders[dists[2][0]].position, defenders[dists[3][0]].position, (0,255,0), 5)
			else:
				cv2.line(overlay, defenders[dists[1][0]].position, defenders[dists[3][0]].position, (0,255,0), 5)
		elif v_dists[dists[2][0]][1] > v_dists[dists[0][0]][1] and\
				v_dists[dists[2][0]][1] > v_dists[dists[3][0]][1]:
			cv2.line(overlay, defenders[dists[0][0]].position, defenders[dists[2][0]].position, (0,255,0), 5)
			cv2.line(overlay, defenders[dists[2][0]].position, defenders[dists[3][0]].position, (0,255,0), 5)
		else:
			cv2.line(overlay, defenders[dists[0][0]].position, defenders[dists[3][0]].position, (0,255,0), 5)


		#Draw lines between defenders
		#for ix in range(1, 4):
		#    cv2.line(overlay, defenders[dists[ix-1][0]].position, defenders[dists[ix][0]].position, (0,255,0), 5)

		# Draw line from enemy to asset
		cv2.line(overlay, enemy.position, asset.position, colour, 5)
		
		opacity = 0.4
		cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)

		cv2.imshow("img", img)
		out.write(img)
		cv2.waitKey(1)

	cap.release()
	out.release()
	print("Breached ", breached, " times, rate = ", float(breached)/(frameIndex-frameSkip))
	print("Flanked ", flanked, " times, rate = ", float(flanked)/(frameIndex-frameSkip))
