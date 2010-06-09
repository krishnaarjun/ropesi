# Classifies an image as containing a "rock,paper,scissors" sign or not
# Classifies between the signs in order to distinguish between the 3 signs
#
# Input:
# - pca = an object of the class "eigenHands" 
#
# Output:
# - merge(bigList)        => merges 2 lists where "bigList" = [list1, list2]
# - classifyHands(noComp) => classifies hands from non-hands using the eigenHands defined by the "noComp" 
# - classifySigns()       => classifies new images as rocks, papers or scissors

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
	#get the training set from video of hands
	def classifyHands(self, noComp):
		#0) get training data data and the labels 	
		garb,dataG,liG,meanG  = self.pca.doPCA("garb", noComp, -1)
		hands,dataH,liH,meanH = self.pca.doPCA("hands", noComp, -1)

		garb = garb[:,0:92]
		hands = hands[:,0:225]

		labels = numpy.empty(hands.shape[1]+garb.shape[1], dtype=int)
		train  = numpy.empty((hands.shape[1]+garb.shape[1],garb.shape[0]), dtype=float)
		indexs = numpy.empty(hands.shape[1]+garb.shape[1], dtype=int)
		for i in range(0, hands.shape[1]+garb.shape[1]):
			indexs[i] = i
			if(i<hands.shape[1]):
				labels[i]  = 1
				train[i,:] = hands[:,i]
			else:
				labels[i]  = -1
				train[i,:] = garb[:,(i-hands.shape[1])]

		#2) initialize the svm and compute the model 		
		problem = mlpy.Svm(kernel='gaussian', kp=0.5, tol=0.000001, eps=0.0000001, maxloops=10000)

		#2) shuffle input data to do the 10-fold split 
		shuffle(indexs)
		labels = labels[indexs]
		train  = train[indexs,:] 

		print labels
		print train.shape

		#3) define the folds, train and test
		pred_err  = 0.0
		folds     = mlpy.kfoldS(cl = labels, sets = 5)
		
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
	#get the training set from video of hands
	def classifySigns(self):
		""".."""
		#pred_label, pred_probability = model.predict_probability(test[i, 0:noComp].tolist())
		#if(pred_label == 1):
		#	labelName = "hands"
		#else:
		#	labelName = "garb"
		#print "label:"+str(labelName)+"["+str(pred_label)+"] >>> prob:"+str(pred_probability)

#________________________________________________________________________
classi = classifyHands(False)
classi.classifyHands(1500)

