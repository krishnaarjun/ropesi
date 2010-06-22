# Does preprocessing of images for the training data and for the svm
#
# Input:
# - size           = size of the training images	
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
# - doSmallManyGabors(theSign,noComp,gaborComp,    => convolves the data matrix corresponding to "theSign" with a set of Gabor Wavelets
#	isPrint)		          	      concatenates the convolved images with the original image in a row for each image
#					      	      calls PCA again over the matrix of concatenated features to reduce the dimensionality		

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
	def __init__(self, size, noComp):
		self.noComp         = noComp
		self.default_width  = 640
		self.default_height = 480
		self.rescale_ratio  = 5
		self.pca            = eigenHands(size)
		self.gabor          = gaborFilters(False, size)
		self.bgTotal        = cv.CreateMat(70, 70, cv.CV_8UC3)
	#________________________________________________________________________
	#get the training set from video of hands
	def getHandsVideo(self, nr):
		capture    = cv.CreateCameraCapture(0)
		img_width  = int(self.default_width/self.rescale_ratio)	
		img_height = int(self.default_height/self.rescale_ratio)	
		#cv.SetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_WIDTH, img_width)
		#cv.SetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_HEIGHT, img_heigth)
		#cv.SetCaptureProperty(capture, cv.CV_CAP_PROP_FPS, 10)
		#cv.SetCaptureProperty(capture, cv.CV_CAP_PROP_CONVERT_RGB, 1)
		index     = 0
		gray_img  = cv.CreateImage((70, 70), cv.IPL_DEPTH_8U, 1)
		small_img = cv.CreateImage((img_width, img_height), cv.IPL_DEPTH_8U, 3)
		imgHSV    = cv.CreateImage((70, 70), cv.IPL_DEPTH_8U, 3)
		h_plane   = cv.CreateImage((70, 70), cv.IPL_DEPTH_8U, 1)
		v_plane   = cv.CreateImage((70, 70), cv.IPL_DEPTH_8U, 1)
		while True:
			index += 1
			if(index%5==0):
				img = cv.QueryFrame(capture)
				cv.Resize(img, small_img)
				cv.SetImageROI(small_img, ((int(img_width/2)-45), (int(img_height/2)-45), 70, 70))

				cv.CvtColor(small_img, imgHSV, cv.CV_BGR2HSV)
				cv.Split(imgHSV, h_plane, None, v_plane, None)
				#cv.InRangeS(h_plane,2,60,h_plane)
				cv.InRangeS(v_plane,130,360,v_plane)
				for i in range(0, 70):
					for j in range(0, 70):
						if(v_plane[i,j]==0):
							small_img[i,j] = (0,0,0)
				cv.CvtColor(small_img, gray_img, cv.CV_BGR2GRAY)		
				cv.EqualizeHist(gray_img, gray_img)			
				cv.ShowImage("camera", gray_img)
				cv.SaveImage("train/"+str(nr)+"camera"+str(index)+".jpg", gray_img)
				cv.ResetImageROI(small_img)
    			if cv.WaitKey(10)==27:
       	 			break
	#________________________________________________________________________
	#prepare the data with multiple Gabor filters for SVM
	def doManyGabors(self, data, txtLabels, theSign, isPCA):
		if(isPCA != True):
			self.noComp = data.shape[1] 
		signs = {"h":["hands", "garb"], "c":["rock", "paper", "scissors"]}

		#1) compute a set of different gabor filters
		lambdas = 4.0 #between 2 and 256
		gammas  = 0.7 # between 0.2 and 1 
		psis    = 20 #between 0 and 180 
		thetas  = [0,(numpy.pi/4.0),(numpy.pi/2.0),(numpy.pi*3.0/4.0)] #between (0 and 180) or (-90 and +90)
		if(self.pca.sizeImg == 20):
			sigmas = 2.0 #between 3 and 68
			sizes  = 1.0 #between 1 and 10
		else:
			sigmas = 3.0 #between 3 and 68
			sizes  = 2.0 #between 1 and 10

		#2) loop over all gabor kernels
		convo = numpy.empty((data.shape[0], self.noComp * len(thetas)), dtype=float)		
		for i in range(0, len(thetas)):
			#3) convolve the images with the gabor filters
			self.gabor.setParameters(lambdas, gammas, psis, thetas[i], sigmas, sizes)
			convolved = self.gabor.convolveImg(self.pca.array2cv(data,True),False)

			#4) concatenate the concolved images with the original stuff on each line
			preConv = self.pca.cv2array(convolved,True)
			for j in range(0, data.shape[0]):
				for k in range(0, self.noComp):
					convo[j,(i*self.noComp)+k] = preConv[j,k]
		
		#5) do PCA on the concatenated convolved images
		finalConv = None
		if(isPCA == True): #not the test image
			print "does PCA"
			if(data.shape[0]>1):
				self.pca.doPCA(convo, self.noComp, "Gabor/")
			else:
				finalConv = self.pca.projPCA(convo, False, "Gabor/", "")

		#6) split the set corresponding to labels and store it
		if(data.shape[0]>1): #not the test image			
			for aSign in signs[theSign]:
				signPart = txtLabels[aSign]
				signSet  = convo[signPart,:]
				if(isPCA == True):
					finalConv = self.pca.projPCA(signSet, False, "Gabor/", "")
					cv.Save("data_train/Gabor/"+aSign+"Train"+str(self.pca.sizeImg)+".dat", self.pca.array2cv(finalConv,False))
				else:
					cv.Save("data_train/Gabor/"+aSign+"Train"+str(self.pca.sizeImg)+".dat", self.pca.array2cv(signSet,False))
		return finalConv
	#________________________________________________________________________
	#prepare the data with multiple Gabor filters for SVM
	def doSmallManyGabors(self, data, txtLabels, theSign, isPCA):
		signs = {"h":["hands", "garb"], "c":["rock", "paper", "scissors"]}
		
		#1) compute a set of different gabor filters
		lambdas = 4.0 #between 2 and 256
		gammas  = 0.8 # between 0.2 and 1 
		psis    = 20 #between 0 and 180 
		thetas  = [0,(numpy.pi/4.0),(numpy.pi/2.0),(numpy.pi*3.0/4.0)] #between (0 and 180) or (-90 and +90)
		if(self.pca.sizeImg == 20):
			sigmas = 2.0 #between 3 and 68
			sizes  = 1.0 #between 1 and 10
		else:
			sigmas = 3.0 #between 3 and 68
			sizes  = 2.0 #between 1 and 10

		#2) store each image as a line at each begining of the row
		convo = numpy.empty((data.shape[0], data.shape[1]*(len(thetas)+1)), dtype=float)
		for j in range(0, data.shape[0]):
			for k in range(0, data.shape[1]):
				convo[j,k] = data[j,k]

		#3) loop over all kernels and convolve them with the images
		for i in range(0, len(thetas)):
			#4) convolve the images with the gabor filters
			self.gabor.setParameters(lambdas, gammas, psis, thetas[i], sigmas, sizes)
			convolved = self.gabor.convolveImg(self.pca.array2cv(data,True),False)

			#5) concatenate the concolved images with the original image on each line
			convNumpy = self.pca.cv2array(convolved,True)
			for j in range(0, data.shape[0]):
				for k in range(0, data.shape[1]):
					convo[j,((i+1)*data.shape[1])+k] = convNumpy[j,k]
			
		#5) do PCA on the concatenated (convolved+original) images 
		finalConv = None
		if(isPCA == True): #not the test image
			if(data.shape[0]>1):					
				self.pca.doPCA(convo, self.noComp, "GaborImg/")
			else:
				finalConv = self.pca.projPCA(convo, False, "GaborImg/", "")

		#6) split the set corresponding to labels and store it
		if(data.shape[0]>1): #not the test image
			for aSign in signs[theSign]:
				signPart = txtLabels[aSign]
				signSet  = convo[signPart,:]

				if(isPCA == True):
					#project only what i need out of convo
					finalConv = self.pca.projPCA(signSet, False, "GaborImg/", "")
					cv.Save("data_train/GaborImg/"+aSign+"Train"+str(self.pca.sizeImg)+".dat", self.pca.array2cv(finalConv,False))
				else:
					cv.Save("data_train/GaborImg/"+aSign+"Train"+str(self.pca.sizeImg)+".dat", self.pca.array2cv(signSet,False))
		return finalConv
#____________________________________________________________________________________________







