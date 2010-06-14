# Does preprocessing of images for the training data and for the svm
#
# Input:
# - default_width  = the width of the image from camera	 
# - default_height = the height of the image from camera	 
# - rescale_ratio  = the ratio with which the image needs to be scaled	  
# - pca            = an object of the class "eigenHands"
# - gabor          = an object of the class "gaborFilters"

#
# Output:
# - getHandsVideo()                                => get the images for the training set
# - doManyGabors(theSign,noComp,gaborComp,isPrint) => convolves the data matrix corresponding to "theSign" with a set of Gabor Wavelets
#					      	      computes the eigen-hands for matrix obtained by convolving each wavelet with the data
#					      	      concatenates the eigen-hands in a row for each image
#					      	      calls PCA again over the matrix of concatenated eigen-hands to reduce the dimensionality		

import sys
import urllib
import re
import copy
import math
import random
import cv
import numpy 
import os
import glob
import mlpy
from eigenHands import *
from gaborFilters import *
class preprocessing:
	def __init__(self):
		self.default_width  = 640
		self.default_height = 480
		self.rescale_ratio  = 5
		self.pca            = eigenHands()
		self.gabor          = gaborFilters(False)
	#________________________________________________________________________
	#get the training set from video of hands
	def getHandsVideo(self, nr):
		cv.NamedWindow("camera", 1)
		capture    = cv.CreateCameraCapture(0)
		img_width  = int(self.default_width/self.rescale_ratio)	
		img_heigth = int(self.default_height/self.rescale_ratio)	
		#cv.SetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_WIDTH, img_width)
		#cv.SetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_HEIGHT, img_heigth)
		#cv.SetCaptureProperty(capture, cv.CV_CAP_PROP_FPS, 10)
		#cv.SetCaptureProperty(capture, cv.CV_CAP_PROP_CONVERT_RGB, 1)
		index     = 0
		gray_img  = cv.CreateImage((70, 70), cv.IPL_DEPTH_8U, 1)
		small_img = cv.CreateImage((img_width, img_heigth), cv.IPL_DEPTH_8U, 3)
		while True:
			img = cv.QueryFrame(capture)
			index += 1
			if(index % 50 == 0):
				cv.Resize(img,small_img)
				cv.SetImageROI(small_img, ((int(img_width/2)-45), (int(img_heigth/2)-45), 70, 70))
				cv.CvtColor(small_img, gray_img, cv.CV_BGR2GRAY)		
				cv.EqualizeHist(gray_img, gray_img)			
				cv.ShowImage("camera", gray_img)
				cv.SaveImage("train/"+str(nr)+"camera"+str(index)+".jpg", gray_img)
				cv.ResetImageROI(small_img)
    			if cv.WaitKey(10)==27:
       		 		break
	#________________________________________________________________________
	#prepare the data with multiple Gabor filters for SVM
	def doManyGabors(self, theSign, noComp, gaborComp, isPrint):
		#1) get the initial cv.Images 
		data = cv.Load("data_train/"+theSign+"Train.dat")
		
		#2) compute a set of different gabor filters
		lambdas = [3.0, 5.0, 10.0, 3.0, 4.0, 7.0] #between 2 and 256
		gammas  = [1.0, 0.9, 1.0, 0.4, 0.8, 1.0] # between 0.2 and 1 
		psis    = [10, 30, 20, 40, 100, 70] #between 0 and 180 
		thetas  = [-45, 0, -10, 90, -90, -30] #between (0 and 180) or (-90 and +90)
		sigmas  = [8.0, 7.0, 6.0, 5.0, 8.0, 7.0] #between 3 and 68
		sizes   = [2.0, 2.0, 3.0, 1.0, 2.0, 1.0] #between 1 and 10
		convo   = numpy.empty((data.height, noComp * len(lambdas)), dtype=float)
		for i in range(0, len(lambdas)):
			self.gabor.setParameters(lambdas[i], gammas[i], psis[i], thetas[i], sigmas[i], sizes[i])
			convolved = self.gabor.convolveImg(data,isPrint)

			#3) do PCA on each convolved image
			preConv     = self.pca.cv2array(convolved,True)
			convPCA,_,_ = self.pca.doPCA(preConv, noComp, -1)	
			for j in range(0, data.height):
				for k in range(0, noComp):
					convo[j,(i*noComp)+k] = convPCA[j,k]
			
		#4) do PCA on the concatenated convolved images ?????????????????????????
		preToSVM,_,_ = self.pca.doPCA(convo, gaborComp, -1)
		toSVM        = self.pca.array2cv(preToSVM, False)
		#cv.Save("data_train/"+theSign+"GaborTrain.dat", toSVM)
		return toSVM
#____________________________________________________________________________________________







