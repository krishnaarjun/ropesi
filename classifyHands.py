# Classifies an image as containing a "rock,paper,scissors" sign or not
# Classifies between the signs in order to distinguish between the 3 signs
#
# Input:
# - pca = an object of the class "eigenHands" 
#
# Output:
# - merge(bigList)                      => merges 2 lists where "bigList" = [list1, list2]
# - classifyHands(noComp,onImg,theSign) => classifies hands from non-hands using the eigenHands defined by the "noComp" for the data given by "theSign" 
# - getDataLabels(noComp,onImg,theSign) => returns data-set and labels (if onImg is true then returns the images, else the eigenhands); 
#					   "theSign" -- is "hands" for hands vs no-hands
#					             -- is "rock" for rock vs paper&scissors
#						     -- is "paper" for papaer vs scissors								

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
from PIL import Image
from eigenHands import *
from numpy.random import shuffle
class classifyHands:
	def __init__(self, makeData):
		self.pca = eigenHands()
		#create the data matrix if they are not there
		if(makeData == True):
			self.pca.makeMatrix("garb")
			self.pca.makeMatrix("hands")
			self.pca.makeMatrix("rock")
			self.pca.makeMatrix("paper")
			self.pca.makeMatrix("scissors")
	#________________________________________________________________________
	#merges two list that are stored like: bigList = [list1, list2]
	def merge(bigList):
	    merged = []
	    for aList in bigList:
		for x in aList:
		    merged.append(x)
	    return merged 

	#________________________________________________________________________
	#get the training set and the labels 
	def getDataLabels(self, noComp, onImg, theSign):
		signs = {"hands":["garb"], "rock":["paper", "scissors"], "paper":["garb"]}
		if(onImg == True):
			good = (self.pca.justGetDataMat(theSign)).T
		else:
			good,_,_ = self.pca.doPCA(theSign, noComp, -1)	 
		bad     = []
		badSize = 0
 		for aSign in signs[theSign]:
			if(onImg == True):
				preBad = (self.pca.justGetDataMat(aSign)).T
			else:
				preBad,_,_ = self.pca.doPCA(aSign, noComp, -1)			 	
			badSize += preBad.shape[1]
			bad.append(preBad)
		labels    = numpy.empty(good.shape[1]+badSize, dtype=int)
		train     = numpy.empty((good.shape[1]+badSize, good.shape[0]), dtype=float)
		indexs    = numpy.empty(good.shape[1]+badSize, dtype=int)
		sizeSoFar = 0
		j         = 0	
		for i in range(0, good.shape[1]+badSize):
			indexs[i] = i
			if(i<good.shape[1]):
				labels[i]  = 1
				train[i,:] = good[:,i]
			else:
				labels[i] = -1
				ind       = (i-good.shape[1])-sizeSoFar
				if((ind-bad[j].shape[1]) == 0):
					sizeSoFar += bad[j].shape[1]
					j         += 1
					ind        = (i-good.shape[1])-sizeSoFar
				train[i,:] = bad[j][:,ind]
		return indexs,labels,train

	#________________________________________________________________________
	#classify images using svm
	def classifyHands(self, noComp, onImg, theSign):
		#0) get training data data and the labels
		indexs,labels,train = self.getDataLabels(noComp, onImg, theSign)

		#2) initialize the svm and compute the model
		if(theSign == "hands"): #the model for hands/no-hands 		
			problem = mlpy.Svm(kernel='gaussian', C=1.0, kp=0.3, tol=0.001, eps=0.001, maxloops=1000, opt_offset=True)
		else: #the model for signs
			problem = mlpy.Svm(kernel='gaussian', C=1.0, kp=0.1, tol=0.001, eps=0.001, maxloops=1000, opt_offset=True)

		#2) shuffle input data to do the 10-fold split 
		shuffle(indexs)
		labels = labels[indexs]
		train  = train[indexs,:] 

		#3) define the folds, train and test
		pred_err = 0.0
		folds    = mlpy.kfoldS(cl = labels, sets = 5)
		for trainI, testI in folds:
			trainSet, testSet = train[trainI], train[testI]
        		trainLab, testLab = labels[trainI], labels[testI]
			learned           = problem.compute(trainSet, trainLab)
			print "it learned >>> "+str(learned)
        		prediction        = problem.predict(testSet)
			pred_err         += mlpy.err(testLab, prediction)
			print pred_err
		avg_err = float(pred_err)/float(len(folds))
		print avg_err
		return problem
#________________________________________________________________________
classi = classifyHands(True)
classi.classifyHands(1500, False, "rock")


