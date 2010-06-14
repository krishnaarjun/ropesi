# Class that get the eigen hands out of a video with hands and stores them
#
# Input:
# - -- 
# Output:
# - makeMatrix(dir)        => creates an matrix-image out of a set of images from a directory "dir"
# - justGetDataMat(what)   => just returns the stored data as a numpy.array instead of a cv.Mat corresponding to "what" (hands,rock,paper,scissors)
# - cv2array(img, isImg)   => converts a cv.Mat("isImg" is FALSE)/cv.Image("isImg" is TRUE) to a numpy array 
# - array2cv(array, isImg) => converts a numpy.array to a cv.Mat("isImg" is FALSE)/cv.Image("isImg" is TRUE)
# - doPCA(X,nrComp,showIm) => returns the projection matrix, the data and the mean of the data 
#			      X      - input data (a matrix with 1 line for each image)
#			      nrComp - number of components to be returned
#			      showIm - show an example backprojection						

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
		"""Nothing ;)"""
	#________________________________________________________________________
	#store the images from the input file/files into a huge cv.Image-matrix	
	def makeMatrix(self, fold):
		files     = os.listdir('train/'+fold)
		imgMatrix = cv.CreateMat(len(files), 4900, cv.CV_8UC1)		
		anIndex   = 0
		matches   = glob.glob(os.path.join("train/"+fold, '*.jpg'))
		matches.sort()
		for aImage in matches:
			hands    = cv.LoadImageM(aImage, cv.CV_LOAD_IMAGE_GRAYSCALE)
			handsMat = cv.Reshape(hands, 0, 1)
			for j in range(0,handsMat.width):
				imgMatrix[anIndex,j] = float(handsMat[0,j])
			anIndex += 1			
		#cv.NamedWindow("img", 1)
		#cv.ShowImage("img",cv.Reshape(imgMatrix[20], 0, 70))
		#print "press any key.."
		#cv.WaitKey()       
		cv.Save("data_train/"+fold+"Train.dat", imgMatrix)	
	#________________________________________________________________________
	#just read the cvMat of data and transforms it to a numpy matrix 
	def justGetDataMat(self, what):
		mat = cv.Load("data_train/"+what+"Train.dat")
		return self.cv2array(mat,True)
	#________________________________________________________________________
	#perform PCA on set of all images from matName
	def doPCA(self, X, noComp, showIm):
		#1) extract the mean of the images out of each image
		nr,dim = X.shape
		meanX  = X.mean(axis=0)
		for i in range(0, nr):
			X[i,:] -= meanX
		
		#2) compute the projection matrix: (X * X.T) * vi = li * vi
		#   ui[:,i] = (X.T * vi)/norm(ui[:,i])
		otherS = numpy.dot(X, X.T) #compute: (X * X.T)
		li,vi  = numpy.linalg.eigh(otherS) #eigenvalues and eigenvectors of (X * X.T)/N
		vi     = vi[:,::-1] #reverse since last eigenvectors are the ones we want
		li     = li[::-1] #reverse since eigenvalues are in increasing order	
		ui     = numpy.dot(X.T, vi) #the formula for the highdim data	
		for i in range(0, nr): #normalize the final eigenvectors
			ui[:,i] = numpy.divide(ui[:,i], numpy.linalg.norm(ui[:,i]))
	
		#3) do projection/back-projection on first N components to check
		if(showIm == True):
			projX = numpy.dot(ui[0:noComp,:], X)
			backX = numpy.dot(ui[0:noComp,:].T, projX)
			for i in range(0, nr):
				backX[i] += meanX
			eigenHand = self.array2cv(backX,True)
			cv.NamedWindow("PCA", 1)
			cv.ShowImage("PCA", cv.Reshape(eigenHand[0], 0, 70))
			print "press any key.."
			cv.WaitKey()       

		#4) return the projection-matrix, eigen-values and the mean
		return ui[0:noComp,:],X,meanX
	#________________________________________________________________________
	#covert an cvMatrix / cvImage to a numpy.array 	
	def cv2array(self, im, isImg):
		if(isImg == True):
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
		a = numpy.asfarray(a)
		return a[:,:,0]
	#________________________________________________________________________
	#covert a numpy.array to a cvMatrix / cvImage 		    
	def array2cv(self, arr, isImg):
		if(isImg == True):
			arr = numpy.asarray(arr, dtype=numpy.uint8)
		else:
			arr = numpy.asarray(arr, dtype=numpy.float32)
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
		if(isImg == True):
			return cv_im
		else:
			return cv.GetMat(cv_im, 1)	
#________________________________________________________________________
#hands = eigenHands()
#hands.makeMatrix("hands")
#X = hands.justGetDataMat("rock")
#hands.doPCA(X, 1000, True)



