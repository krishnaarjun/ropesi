"""Class that get the eigen hands out of a video with hands and stores them"""

import sys
import urllib
import re
import copy
import math
import random
"""for openCV ;)"""
import cv
class eigenHands:
	def __init__(self):
		self.default_width  = 640
		self.default_height = 480
		self.rescale_ratio  = 5
	"""________________________________________________________________________"""
	"""get the training set from video of hands"""		
	def getHandsVideo(self):
		cv.NamedWindow("camera", 1)
		capture    = cv.CreateCameraCapture(0)
		"""set the size in opencv2.1"""
		img_width  = int(self.default_width/self.rescale_ratio)	
		img_heigth = int(self.default_height/self.rescale_ratio)	
		#cv.SetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_WIDTH, img_width)
		#cv.SetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_HEIGHT, img_heigth)
		cv.SetCaptureProperty(capture, cv.CV_CAP_PROP_FPS, 10)
		#cv.SetCaptureProperty(capture, cv.CV_CAP_PROP_CONVERT_RGB, 1)
		index     = 0
		gray_img  = cv.CreateImage((self.default_width, self.default_height), cv.IPL_DEPTH_8U, 3)
		small_img = cv.CreateImage((img_width, img_heigth), cv.IPL_DEPTH_8U, 3)
		while True:
			img = cv.QueryFrame(capture)
			index += 1
			if(index % 50 == 0):
				cv.Resize(img,small_img)
				#"""if the color set is RGB then it can be converted to GRAY"""
				#cv.CvtColor(img, gray_img, cv.CV_RGB2GRAY)
				"""select the region of interest (ROI) 70px around the center"""
				cv.SetImageROI(small_img, ((int(img_width/2)-45), (int(img_heigth/2)-45), 70, 70));
				cv.SaveImage("train/camera"+str(index)+".jpg", small_img)		
				cv.ShowImage("camera", small_img)	
				cv.ResetImageROI(small_img)
    			if cv.WaitKey(10)==27:
       		 		break
		
		
"""________________________________________________________________________"""
hands = eigenHands()
hands.getHandsVideo()
