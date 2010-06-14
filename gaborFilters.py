# Convolve an image with a gabor filter and return the resulted marix for all images to be classified
#
# Input:
# - lambda => wavelength between 2 and 256
# - theta  => the orientation specified in degreees; valid values between 0 and 180
# - psi    => phase offset of the cosine factor; valid values between -180  and 180
#	      for symmetric functions: valid values between 0 and 180
#	      for antisymmetric functions: between -90 and 90 	 		
# - gamma  => ellipticity of the Gaussian factor; valid values between 0.2 and 1  
# - sigma  => valid values between 3 and 68
#
# Output:
# - 

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
class gaborFilters:
	def __init__(self, makeData):
		self.pca = eigenHands()
		if(makeData == True):
			self.pca.makeMatrix("garb")
			self.pca.makeMatrix("hands")
			self.pca.makeMatrix("rock")
			self.pca.makeMatrix("paper")
			self.pca.makeMatrix("scissors")
	#________________________________________________________________________
	#create the gabor filter with the parameters and return the wavelet
	def setParameters(self, lambd, gamma, psi, theta, sigma):
		self.lambd = lambd
		self.gamma = gamma
		self.psi   = psi
		self.theta = theta
		self.sigma = sigma
	#________________________________________________________________________
	#create the gabor filter with the parameters and return the wavelet
	def createGabor(self,dimension,isPrint):
		sigmaX = float(self.sigma)
		sigmaY = float(self.sigma)/float(self.gamma)
		xMax   = numpy.maximum(numpy.abs(dimension*sigmaX*numpy.cos(self.theta)),numpy.abs(dimension*sigmaY*numpy.sin(self.theta)))		
		xMax   = numpy.ceil(numpy.maximum(1.0,xMax))
		yMax   = numpy.maximum(numpy.abs(dimension*sigmaX*numpy.cos(self.theta)),numpy.abs(dimension*sigmaY*numpy.sin(self.theta)))
		yMax   = numpy.ceil(numpy.maximum(1.0,xMax))
		xMin   = -xMax
		yMin   = -yMax
		gabor  = numpy.empty((int(xMax)-int(xMin), int(yMax)-int(yMin)), dtype=float)
		for x in range(int(xMin), int(xMax)):
			for y in range(int(yMin), int(yMax)):
				xPrime = x*numpy.cos(self.theta)+y*numpy.sin(self.theta)
				yPrime = -x*numpy.sin(self.theta)+y*numpy.cos(self.theta) 			
				gabor[x+int(xMin),y+int(yMin)] = numpy.exp(-0.5*((xPrime*xPrime)/(sigmaX*sigmaX)+(yPrime*yPrime)/(sigmaY*sigmaY)))*numpy.cos(2.0*numpy.pi/self.lambd*xPrime+self.psi)
		if(isPrint == True):
			mini   = numpy.min(gabor)
			showG  = gabor+numpy.abs(mini)
			maxi   = numpy.max(showG)
			showG  = numpy.divide(showG, maxi)
			showG  = numpy.dot(showG, 255)
			imageG = self.pca.array2cv(showG,True)
			cv.NamedWindow("gabor", 1)
			cv.ShowImage("gabor", imageG)
			print "press any key .."
			cv.WaitKey()
		wavelet = self.pca.array2cv(gabor, False)
		print gabor.shape      		
		return gabor,wavelet
	#________________________________________________________________________
	#covolve an the images with gabor a given filter
	def convolveImg(self, data, isPrint):
		_,wavelet = self.createGabor(2, isPrint)
		finalData = cv.CreateMat(data.height, data.width, cv.CV_8UC1)
		for i in range(0,data.height):
			reshData = cv.Reshape(data[i], 0, 70)
			if(isPrint == True and i < 2):
				cv.NamedWindow("img"+str(i), 1)
				cv.ShowImage("img"+str(i), reshData)
			cv.Filter2D(reshData,reshData,wavelet)	
			temp = cv.Reshape(reshData, 0, 1)		
			for j in range(0,temp.width):			
				finalData[i,j] = temp[0,j]
			if(isPrint == True and i < 2):
				cv.NamedWindow("response"+str(i), 1)
				cv.ShowImage("response"+str(i), reshData)
				print "press any key.."
				cv.WaitKey()     
		return finalData
#________________________________________________________________________
#lambda=[2,256], gamma=[0.2,1], psi=[0,180], theta=[0,180], sigma=[3,68]
gabor = gaborFilters(True)
gabor.setParameters(3.0, 1.0, 10.0, -45.0, 8.0)
#gabor.createGabor(5,True)
data  = cv.Load("data_train/rockTrain.dat")
gabor.convolveImg(data,True)



		