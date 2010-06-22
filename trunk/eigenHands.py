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
			anIndex += 1		
		cv.Save("data_train/"+fold+"Train"+str(self.sizeImg)+".dat", imgMatrix)	
	#________________________________________________________________________
	#just read the cvMat of data and transforms it to a numpy matrix 
	def justGetDataMat(self, what, folder, justData):
		data = self.cv2array(cv.Load("data_train/"+str(folder)+str(what)+"Train"+str(self.sizeImg)+".dat"),True)
		if(justData == True):
			return data
		else:
			signs = {"rock":["rock","paper","scissors"], "paper":["rock","paper","scissors"], "scissors":["rock","paper","scissors"], "hands":["hands","garb"], "garb":["hands","garb"]}
			mat   = []
			rows  = 0
			cols  = 0 
			for sign in signs[what]:
				premat = self.cv2array(cv.Load("data_train/"+str(folder)+str(sign)+"Train"+str(self.sizeImg)+".dat"),True)
				mat.append(premat)
				rows += premat.shape[0]
				cols  = premat.shape[1]
			finalMat  = numpy.zeros((rows, cols), dtype=float)
			txtLabels = {}
			sizeSoFar = 0
			for i in range(0,len(mat)):
				for j in range(0,mat[i].shape[0]):
					k             = (j+sizeSoFar)
					if(signs[what][i] in txtLabels):					
						txtLabels[signs[what][i]].append(k)
					else:
						txtLabels[signs[what][i]] = []
					finalMat[k,:] = mat[i][j,:]
				sizeSoFar += mat[i].shape[0]
			return data,finalMat,txtLabels 
	#________________________________________________________________________
	#perform PCA on set of all images from matName
	def doPCA(self, X, noComp, folder):
		#1) extract the mean of the images out of each image
		nr,dim   = X.shape
		meanX    = numpy.empty((1,dim), dtype=float)	
		meanX[0] = X.mean(axis=0)
		for i in range(0, nr):
			X[i,:] -= meanX[0,:]
		
		#2) compute the projection matrix: (X * X.T) * vi = li * vi
		#compute: ui[:,i] = (X.T * vi)/norm(ui[:,i])
		otherS = numpy.dot(X, X.T) #compute: (X * X.T)
		li,vi  = numpy.linalg.eigh(otherS) #eigenvalues and eigenvectors of (X * X.T)/N
		indxs  = numpy.argsort(li) 
		ui     = numpy.dot(X.T, vi) #the formula for the highdim data

		#3) normalize the eigenvectors and sort them
		for i in range(0, nr): #normalize the final eigenvectors
			ui[:,i] = numpy.divide(ui[:,i], numpy.linalg.norm(ui[:,i]))
		ui = ui[:,indxs] #sort the eigenvectors of (X * X.T)/N by the eigenvalues (ui and vi have the same eigenvalues => li)
		
		#4) store the eigenvectors and the mean
		cvEigen = self.array2cv(ui[:,0:noComp], False)
		cv.Save("data_train/"+str(folder)+"PcaEigen"+str(self.sizeImg)+".dat", cvEigen)
		cvMean = self.array2cv(meanX, False)
		cv.Save("data_train/"+str(folder)+"PcaMean"+str(self.sizeImg)+".dat", cvMean)
	#________________________________________________________________________
	#project and verify PCA
	def projPCA(self, X, showIm, folder, sign):
		#1) Load the eigenVector and the Mean of the data:
		eigen = self.cv2array(cv.Load("data_train/"+str(folder)+"PcaEigen"+str(self.sizeImg)+".dat"), False)
		meanX = self.cv2array(cv.Load("data_train/"+str(folder)+"PcaMean"+str(self.sizeImg)+".dat"), False)
	
		#2) Substract the mean of the data
		for i in range(0, X.shape[0]):
			X[i,:] -= meanX[0,:]
		
		#3) do projection on first "?" components: [N,4900]x[4900,?] => [N,?] 
		projX = numpy.dot(X, eigen)
	
		#5) do back-projection on first "?" components to check: [N,?]x[?,4900] => [N,4900] 
		if(showIm == True):
			backX = numpy.dot(projX, eigen.T)
			for i in range(0, X.shape[0]):
				backX[i,:] += meanX[0,:] #add the mean back
			#6) normalize the backprojection
			mini   = numpy.min(backX)
			backX += abs(mini)
			maxi   = numpy.max(backX)
			backX  = numpy.multiply(backX, float(255)/float(maxi))
			
			#7) Shot the backprojected image
			eigenHand = self.array2cv(backX[0:1,:],True)
			cv.ShowImage("PCA", cv.Reshape(eigenHand, 0, self.sizeImg))
			print "press any key.."
			cv.WaitKey()       

		#8) store the projection of the data 
		if(len(sign)>1):
			cvData = self.array2cv(projX, False)
			cv.Save("data_train/"+str(folder)+str(sign)+"Train"+str(self.sizeImg)+".dat", cvData)
		return projX
	#________________________________________________________________________
	#covert an cvMatrix / cvImage to a numpy.array 	
	def cv2array(self, im, isImg):
		if(isImg == True):
			arrdtype = 'uint8'
			channels = 1
		else:
			arrdtype = 'float32'
			channels = 1
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
