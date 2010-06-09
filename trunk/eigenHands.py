# Class that get the eigen hands out of a video with hands and stores them
#
# Input:
# - default_width  = the width of the image from camera	 
# - default_height = the height of the image from camera	 
# - rescale_ratio  = thr ratio with which the image needs to be scaled	  
#
# Output:
# - getHandsVideo()        => get the images for the training set
# - makeMatrix(dir)        => creates a matrix out of a set of images from a directory "dir"
# - cv2array(img, depth)   => converts a cv matrix to a numpy array (if "depth" is 0 the cv matrix is an image)
# - array2cv(array, isImg) => converts a numpy array to a cv matrix (if "isImg" is 1 it converrts to a cv image)
# - doPCA(dataMat, nrComp) => returns the projection matrix, the eigenvalues and the mean of the data from the cv matrix "dataMat"

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
from PIL import Image
class eigenHands:
	def __init__(self):
		self.default_width  = 640
		self.default_height = 480
		self.rescale_ratio  = 5
	#________________________________________________________________________
	#get the training set from video of hands
	def getHandsVideo(self):
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
				cv.SaveImage("train/17camera"+str(index)+".jpg", gray_img)
				cv.ResetImageROI(small_img)
    			if cv.WaitKey(10)==27:
       		 		break
	#________________________________________________________________________
	#store the images from the input file/files into a huge matrix	
	def makeMatrix(self, fold):
		files     = os.listdir('train/'+fold)
		imgMatrix = cv.CreateMat(len(files), 4900, cv.CV_8UC1)
		anIndex   = 0
		matches   = glob.glob(os.path.join("train/"+fold, '*.jpg'))
		matches.sort()
		for aImage in matches:
			hands    = cv.LoadImageM(aImage, cv.CV_LOAD_IMAGE_GRAYSCALE)
			handsMat = cv.Reshape(hands, 0, 1)
			for j in range(0,4900):
				imgMatrix[anIndex,j] = handsMat[0,j]
			anIndex += 1

		#cv.NamedWindow("show1",1)
		#cv.ShowImage("show1", cv.Reshape(imgMatrix[0,:], 0, 70))
		#cv.WaitKey()
		cv.Save("data_train/"+fold+"Train.dat", imgMatrix)	
	#________________________________________________________________________
	#perform PCA on set of all images
	def doPCA(self, matName, noComp, showIm):
		#0) convert the cvMatrix to a numpy array
		mat    = cv.Load("data_train/"+matName+"Train.dat")
		preX   = numpy.asfarray(self.cv2array(mat,0))
		X      = preX[:,:,0]
		nr,dim = X.shape

		#1) extract the mean of the images out of each image
		meanX = X.mean(axis=0)
		for i in range(0, nr):
			X[i,:] -= meanX
		
		#2) compute the projection matrix: (X * X.T)/N * vi = li * vi
		#   ui = (X.T * vi)/sqrt(N * li)
		otherS = numpy.dot(X, X.T) #compute: (X * X.T)/N
		otherS = numpy.divide(otherS, float(nr))
		li,vi  = numpy.linalg.eigh(otherS) #eigenvalues and eigenvectors of (X * X.T)/N
		ui     = numpy.dot(X.T, vi) #the formula for the highdim data
		for i in range(0, nr):
			ui[:,i] = numpy.divide(ui[:,i], numpy.sqrt(float(nr) * li[i])) 	
		ui     = ui[:,::-1] #reverse since last eigenvectors are the ones we want
		finLi  = li[::-1] #reverse since eigenvalues are in increasing order
		
		#3) do projection/back-projection on first N components to check
		if(showIm >= 0 and showIm < X.shape[0]):
			projX = numpy.dot(ui[0:noComp,:], X)
			backX = numpy.dot(ui[0:noComp,:].T, projX)
			for i in range(0, nr):
				backX[i] += meanX
			eigenHand = self.array2cv(backX, 1)
			cv.NamedWindow("PCA", 1)
			cv.ShowImage("PCA", cv.Reshape(eigenHand[showIm], 0, 70))
			cv.WaitKey()       

		#4) return the projection-matrix, eigen-values and the mean
		return ui[0:noComp,:],X,finLi,meanX
	#________________________________________________________________________
	#covert an cvMatrix (image) to a numpy array 	
	def cv2array(self,im, depth):
		if(depth == 0):
			arrdtype = 'uint8'
			channels = 1
		else:
			depth2dtype = {
				cv.IPL_DEPTH_8U: 'uint8',
				cv.IPL_DEPTH_8S: 'int8',
				cv.IPL_DEPTH_16U: 'uint16',
				cv.IPL_DEPTH_16S: 'int16',
				cv.IPL_DEPTH_32S: 'int32',
				cv.IPL_DEPTH_32F: 'float32',
				cv.IPL_DEPTH_64F: 'float64',
			}
			arrdtype = depth2dtype[im.depth]
			channels = im.nChannels
		a = numpy.fromstring(im.tostring(),dtype=arrdtype,count=im.width*im.height*channels)
		a.shape = (im.height,im.width,channels)
		return a
	#________________________________________________________________________
	#covert a numpy array to a cvMatrix (image) 		    
	def array2cv(self, arr, isImg):
		if(isImg == 1):
			arr = numpy.array(arr, dtype=numpy.uint8)
			prevType = numpy.dtype(numpy.uint8)
		else:
			prevType = arr.dtype
		dtype2depth = {
			'uint8':   cv.IPL_DEPTH_8U,
			'int8':    cv.IPL_DEPTH_8S,
			'uint16':  cv.IPL_DEPTH_16U,
			'int16':   cv.IPL_DEPTH_16S,
			'int32':   cv.IPL_DEPTH_32S,
			'float32': cv.IPL_DEPTH_32F,
			'float64': cv.IPL_DEPTH_64F,
		}
		try:
			nChannels = arr.shape[2]
		except:
			nChannels = 1
		cv_im = cv.CreateImageHeader((arr.shape[1],arr.shape[0]), dtype2depth[str(prevType)],nChannels)
		cv.SetData(cv_im, arr.tostring(), prevType.itemsize*nChannels*arr.shape[1])
		return cv_im			
#________________________________________________________________________
#hands = eigenHands()
#hands.getHandsVideo()
#crate the matrixes for all train sets: garb, rock, paper, scissors
#hands.makeMatrix("test")
#hands.makeMatrix("rock")
#hands.makeMatrix("paper")
#hands.makeMatrix("scissors")
#hands.doPCA("paper", noComp, -1)



