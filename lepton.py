#!/usr/bin/env python

import sys
import numpy as np
import cv2
import RPi.GPIO as GPIO

import vlc
#import pygame
import time
import threading

from pylepton import Lepton

fireLimit = 65
imgHeight = 60
imgWidth = 80
imgMagni = 8

class ObjInfo(object):
	def __init__(self, no, x, y, width, height, area, cx, cy, temp, state):
		self.no, self.x, self.y, self.width, self.height, self.area, self.cx, self.cy, self.temp, self.state = no, x, y, width, height, area, cx, cy, temp, state

def mean(grayImg):
	val = 0
	height, width, ch = grayImg.shape

	for y in range(0, height):
		for x in range(0, width):
			val = val + grayImg.item(y, x, 0)

	return val/(height*width)

def noiseCancelling(binImg):
	ret, _, stats, _ = cv2.connectedComponentsWithStats(binImg)

	for i in range(1, ret):
		x, y, width, height, area = stats[i]

		if area <= 3 or width <= 5 or height <= 5:
			cv2.rectangle(binImg, (x, y), (x + width, y + height), 0, -1)

	return binImg

def labeling(binImg, objList):
	ret, labels, stats, centriods = cv2.connectedComponentsWithStats(binImg)

	for i in range(1, ret):
		x, y, width, height, area = stats[i]
		cx, cy = centriods[i]

		objList.append(ObjInfo(i, x, y, width, height, area, int(cx), int(cy), 0, 0))

	return objList

def myMap(x, in_min, in_max, out_min, out_max):
	return int((x-in_min) * (out_max-out_min) / (in_max-in_min) + out_min)

def tempMeasuring(rawImg, binImg, objList):
	for i in range(len(objList)):
		cropImg = rawImg[objList[i].y:objList[i].y + objList[i].height, objList[i].x:objList[i].x + objList[i].width]
		cropBinImg = binImg[objList[i].y:objList[i].y + objList[i].height, objList[i].x:objList[i].x + objList[i].width, 0]
		height, width, ch = cropImg.shape

		maxVal = 0
		cnt = 0
		temp = 0

		for y in range(0, height):
			for x in range(0, width):
				if cropBinImg.item(y, x) != 0:
					if maxVal < cropImg.item(y, x, 0):
						maxVal = cropImg.item(y, x, 0)

		temp = myMap(maxVal, 0, 65535, -273.15, 382.2)
		objList[i].temp = temp

	return objList

def guideDrawing(bgrImg, objList):
	for i in range(len(objList)):
		if objList[i].state == 1 :
			cv2.circle(bgrImg, (objList[i].cx, objList[i].cy), 1, (255, 0, 0), -1)
			cv2.putText(bgrImg, str(objList[i].temp) + "'c", (objList[i].cx, objList[i].cy), cv2.FONT_ITALIC, 0.25, (0, 0, 255), 1)
			#cv2.rectangle(bgrImg, (objList[i].x, objList[i].y), (objList[i].x + objList[i].width, objList[i].y + objList[i].height), (255, 255, 255), 1) 
			#cv2.circle(bgrImg, (objList[i].cx, objList[i].cy), 1, (255, 0, 0), -1)
			#qcv2.putText(bgrImg, "no." + str(objList[i].no), (objList[i].x, objList[i].y), cv2.FONT_ITALIC, 0.3, (0, 0, 255), 1)
			#cv2.putText(bgrImg, str(objList[i].temp) + "'c", (objList[i].cx, objList[i].cy), cv2.FONT_ITALIC, 0.3, (0, 0, 255), 1)

	return bgrImg

def fireDetecting(objList):
	flag = False
	maxTemp = 0
	for i in range(len(objList)):
		if objList[i].temp >= fireLimit and objList[i].temp >= maxTemp:
			maxTemp = objList[i].temp

	for i in range(len(objList)):
		if maxTemp == objList[i].temp:
			objList[i].state = 1
			flag = True

	return objList, flag

def fireTracing(objList):
	priList = objList[:]

	priList.sort(key = lambda object:object.temp)

	if len(objList) == 0:
		target = [40, 30]
	else:
		target = [priList[-1].cy, priList[-1].cx]

	return target

def capture(flip_v = True, device = "/dev/spidev0.0"):
	with Lepton(device) as l:
		a,_ = l.capture()

	if flip_v:
		cv2.flip(a,1,a)

	return np.uint16(a)

if __name__ == '__main__':

	sp = vlc.MediaPlayer('./fire.wav')
	sound = False
	soundCnt = 0
	sp.stop()

	rawImg = np.zeros((imgHeight, imgWidth, 1), np.uint16) # 16bit, 1ch
	binImg = np.zeros((imgHeight, imgWidth, 1), np.uint8) # 8bit, 1ch
	finalImg = np.zeros((imgHeight*imgMagni, imgWidth*imgMagni, 3), np.uint8) # 8bit, 3ch

	objList = []

	fireX = 0
	fireY = 0
	ang = 3.0
	servoCnt = 0

	GPIO.setwarnings(False)
	GPIO.setmode(GPIO.BOARD)
	GPIO.setup(32, GPIO.OUT)
	GPIO.setup(33, GPIO.OUT)
	servoY = GPIO.PWM(32, 50)
	servoX = GPIO.PWM(33, 50)
	(servoY).start(0)
	(servoX).start(0)
	(servoY).ChangeDutyCycle(7.5)
	(servoX).ChangeDutyCycle(7.5)

	time.sleep(2)

	while True:
		# < Collecting input raw image >
		rawImg = capture()

		# < Making binary layer image >
		instImg = rawImg.copy()
		backMean = np.mean(instImg) + 10
		if backMean < 30000:
			backMean = 30000
		instImg[rawImg >= np.mean(backMean)] = 65535
		instImg[rawImg < np.mean(backMean)] = 0
		np.right_shift(instImg, 8, instImg)
		binImg = np.uint8(instImg)
		binImg = noiseCancelling(binImg)
		# ret, binImg = cv2.threshold(binImg, 128, 255, cv2.THRESH_BINARY)

		# < Labeling image >
		objList = []
		objList = labeling(binImg, objList)

		# < Detecting fire >
		objList = tempMeasuring(rawImg, binImg, objList)
		objList, flag = fireDetecting(objList)

		# < Speaker on >
		if not sound and flag :
			sp.stop()
			sp.play()
			sound = flag
			soundCnt = 0
		elif sound and not flag :
			sp.stop()
			sound = flag
		else:
			soundCnt += 1
			if soundCnt > 100:
				sound = False

		# < Tracing fire >
		if flag :
			fireY, fireX = fireTracing(objList)
			angY = float(myMap(fireY, 0, 79, 4.5, 11))
			angX = float(myMap(fireX, 0, 79, 4.5, 11))
			(servoY).ChangeDutyCycle(angY)
			(servoX).ChangeDutyCycle(angX)
			servoCnt = 0
		else :
			if servoCnt == 100 :
				(servoX).ChangeDutyCycle(7.5)
	                        (servoY).ChangeDutyCycle(7.5)
			servoCnt += 1

		# < Printing result >
		instImg = rawImg.copy()
		cv2.normalize(instImg, instImg, 0, 65535, cv2.NORM_MINMAX)
		np.right_shift(instImg, 8, instImg)
		instImg = np.uint8(instImg)
		instImg = cv2.cvtColor(instImg, cv2.COLOR_GRAY2BGR)
		instImg = guideDrawing(instImg, objList)
		finalImg = cv2.resize(instImg, (imgWidth*imgMagni,imgHeight*imgMagni), interpolation=cv2.INTER_AREA)
		cv2.namedWindow('pythermal', cv2.WND_PROP_FULLSCREEN)
		cv2.setWindowProperty('pythermal', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
		cv2.imshow('pythermal', finalImg)
		# cv2.imwrite('output.bmp', finalImg)

		key = cv2.waitKey(1)
		if (key == ord('q')):
			break
