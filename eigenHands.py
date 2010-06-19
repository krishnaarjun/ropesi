# Class that get the eigen hands out of a video with hands and stores them
#
# Input:
# - sizeImg                => the size of the images to be used 
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
#from PIL import Image
class eigenHands:
	def __init__(self, size):
		self.sizeImg = size
	#________________________________________________________________________
	#store the images from the input file/files into a huge cv.Image-matrix	
	def makeMatrix(self, fold):
		files     = os.listdir('train/'+fold)
		anIndex   = 0
		matches   = glob.glob(os.path.join("train/"+fold, '*.jpg'))

		print (self.sizeImg*self.sizeImg)		
		print len(matches)

		imgMatrix = cv.CreateMat(len(matches), (self.sizeImg*self.sizeImg), cv.CV_8UC1)		
		oneLine   = cv.CreateMat(1, (self.sizeImg*self.sizeImg), cv.CV_8UC1)	
		resizeImg = cv.CreateMat(self.sizeImg, self.sizeImg, cv.CV_8UC1)
				
		matches.sort()
		for aImage in matches:
			hands    = cv.LoadImageM(aImage, cv.CV_LOAD_IMAGE_GRAYSCALE)
			cv.Resize(hands, resizeImg, interpolation = cv.CV_INTER_AREA)	
			handsMat = cv.Reshape(resizeImg, 0, 1)
			for j in range(0,handsMat.width):
				imgMatrix[anIndex,j] = handsMat[0,j]
				
			"""cv.NamedWindow("img", 1)
			for j in range(0, (self.sizeImg*self.sizeImg)):
				oneLine[0,j] = imgMatrix[anIndex,j]
			cv.ShowImage("img",cv.Reshape(oneLine, 0, self.sizeImg))
			print "press any key.."+str(anIndex)
			cv.WaitKey()"""

			anIndex += 1		
		cv.Save("data_train/"+fold+"Train"+str(self.sizeImg)+".dat", imgMatrix)	
	#________________________________________________________________________
	#just read the cvMat of data and transforms it to a numpy matrix 
	def justGetDataMat(self, what):
		mat = cv.Load("data_train/"+what+"Train"+str(self.sizeImg)+".dat")
		return self.cv2array(mat,True)
	#________________________________________________________________________
	#perform PCA on set of all images from matName
	def doPCA(self, X, noComp, showIm, sign):
		#1) extract the mean of the images out of each image
		nr,dim = X.shape
		meanX  = X.mean(axis=0)
		for i in range(0, nr):
			X[i,:] -= meanX
		
		#2) compute the projection matrix: (X * X.T) * vi = li * vi
		#compute: ui[:,i] = (X.T * vi)/norm(ui[:,i])
		otherS = numpy.dot(X, X.T) #compute: (X * X.T)
		li,vi  = numpy.linalg.eigh(otherS) #eigenvalues and eigenvectors of (X * X.T)/N
		indxs  = numpy.argsort(li) 
		ui     = numpy.dot(X.T, vi) #the formula for the highdim data
		for i in range(0, nr): #normalize the final eigenvectors
			ui[:,i] = numpy.divide(ui[:,i], numpy.linalg.norm(ui[:,i]))
		ui = ui[:,indxs] #sort the eigenvectors of (X * X.T)/N by the eigenvalues (ui and vi have the same eigenvalues => li)

		#3) do projection on first "?" components: [N,4900]x[4900,?] => [N,?] 
		projX  = numpy.dot(X, ui[:,0:noComp])
		print projX.shape
		if(showIm == True):
			#4) do back-projection on first "?" components to check: [N,?]x[?,4900] => [N,4900] 
			backX = numpy.dot(projX, ui[:,0:noComp].T)
			for i in range(0, nr):
				backX[i,:] += meanX
			eigenHand = self.array2cv(backX,True)
			cv.NamedWindow("PCA", 1)
			cv.ShowImage("PCA", cv.Reshape(eigenHand[0], 0, self.sizeImg))
			print "press any key.."
			cv.WaitKey()       

		#4) return the projection-matrix, eigen-values and the mean
		if(len(sign)>0):
			cvData = self.array2cv(projX, False)
			cv.Save("data_train/"+str(sign)+"PcaTrain"+str(self.sizeImg)+".dat", cvData)	
		return projX, X, meanX
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
