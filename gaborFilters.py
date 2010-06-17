# Convolve an image with a Gabor filter and return the resulted marix for all images to be classified
#
# Input:
# - lambda => wavelength between 2 and 256
# - theta  => the orientation specified in degreees; valid values between 0 and 180
# - psi    => phase offset of the cosine factor; valid values between -180  and 180
#         for symmetric functions: valid values between 0 and 180
#         for antisymmetric functions: between -90 and 90           
# - gamma  => ellipticity of the Gaussian factor; valid values between 0.2 and 1  
# - sigma  => valid values between 3 and 68
# - dim    => the dimension od the Gabor wavelet is proportional with "dim"
#
# Output:
# - setParameters(lambda,gamma,psi,theta,sigma,dim) => set the parameter of the Gabor filter
# - createGabor(isPrint)      => creates a Gabor wavelet with the given parameters
# - convolveImg(data,isPrint) => reshapes each row in "data" to the size of the original images (DxD) and convolves them with 
#                the computed Gabor wavelet

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
		self.lambd = None
		self.gamma = None
		self.psi   = None
		self.theta = None
		self.sigma = None
		self.dim   = None
		if(makeData == True):
			self.pca.makeMatrix("garb")
			self.pca.makeMatrix("hands")
			self.pca.makeMatrix("rock")
			self.pca.makeMatrix("paper")
			self.pca.makeMatrix("scissors")
	#________________________________________________________________________
	#create the gabor filter with the parameters and return the wavelet
	def setParameters(self, lambd, gamma, psi, theta, sigma, dim):
		self.lambd = lambd
		self.gamma = gamma
		self.psi   = psi
		self.theta = theta
		self.sigma = sigma
		self.dim   = dim
	#________________________________________________________________________
	#create the gabor filter with the parameters and return the wavelet
	def createGabor(self, isPrint):
		sigmaX = float(self.sigma)
		sigmaY = float(self.sigma)/float(self.gamma)
		xMax   = numpy.maximum(numpy.abs(self.dim*sigmaX*numpy.cos(self.theta)),numpy.abs(self.dim*sigmaY*numpy.sin(self.theta)))		
		xMax   = numpy.ceil(numpy.maximum(1.0,xMax))
		yMax   = numpy.maximum(numpy.abs(self.dim*sigmaX*numpy.cos(self.theta)),numpy.abs(self.dim*sigmaY*numpy.sin(self.theta)))
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
		_,wavelet = self.createGabor(isPrint)
		finalData = cv.CreateMat(data.height, data.width, cv.CV_8UC1)
		dataVect  = cv.CreateMat(1, data.width, cv.CV_8UC1)
		for i in range(0,data.height):
			for j in range(0, data.width):
				dataVect[0,j] = data[i,j] 
			reshData = cv.Reshape(dataVect, 0, self.pca.sizeImg)
			if(isPrint == True and i==3):
				cv.NamedWindow("img", 1)
				cv.ShowImage("img", reshData)
			cv.Filter2D(reshData,reshData,wavelet)	
			temp = cv.Reshape(reshData, 0, 1)		
			for j in range(0,temp.width):			
				finalData[i,j] = temp[0,j]
			if(isPrint == True and i==3):
				cv.NamedWindow("response", 1)
				cv.ShowImage("response", reshData)
				print "press any key.."+str(i)
				cv.WaitKey()
		return finalData
#________________________________________________________________________

    
