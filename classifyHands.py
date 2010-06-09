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
class classifyHands:
	def __init__(self, makeData):
		self.pca = eigenHands()
		#create the data matrix if they are not there
		self.pca.makeMatrix("hands")
		self.pca.makeMatrix("garb")

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
		#0) get training data data and the samples 	
		garb,dataG,liG,meanG  = self.pca.doPCA("garb", noComp, -1)
		hands,dataH,liH,meanH = self.pca.doPCA("hands", noComp, -1)

		test,dataT,liT,meanT  = self.pca.doPCA("test", noComp, -1)
		#labelsTrain           = numpy.ones(hands.shape, dtype=int)

		labels = []
		train  = numpy.empty((garb.shape[0], hands.shape[1]+garb.shape[1]), dtype=int)
		print hands.shape
		print garb.shape
		print train.shape

		for i in range(0, hands.shape[1]+garb.shape[1]):
			if(i<hands.shape[1]):
				labels.append(1)
				train[:,i] = hands[:,i]
			else:
				labels.append(-1)
				train[:,i] = garb[:,(i-hands.shape[1])]
		labelsTrain = numpy.asarray(labels)
		print labelsTrain.shape
		print train.shape
		
			
		#2) initialize the svm and compute the model 		
		problem      = mlpy.Svm(kernel='linear', kp=0.1, C=1.0, eps=0.001)
		problem.compute(train.T, labelsTrain)
		predictTrain = problem.predict(train.T)	
		trainErr     = mlpy.err(labelsTrain, predictTrain)
		print trainErr

		#4) test the classification
		labelsTest  = [1,1,1,1,1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
		predictTest = problem.predict(test.T)	
		testErr     = mlpy.err(labelsTest, predictTest)

		print testErr			
		print noComp
		return problem

	#________________________________________________________________________
	#get the training set from video of hands
	def oldclassifyHands(self, noComp):
		#0) noComp needs to be the same for all sets 	
		hands,dataH,liH,meanH = self.pca.doPCA("hands", noComp, -1)
		test,dataT,liT,meanT  = self.pca.doPCA("test", noComp, -1)
		
		#1) define the problem: labels & samples 
		labels  = []
		samples = []
		for i in range(0, hands.shape[1]):
			samples.append(hands[0:noComp, i].tolist())
			labels.append(0)

		#2) store the problem :-/
		problem = svm.svm_problem(labels, samples)
		size    = len(samples)

		#3) generate the model from the data 
		#types   = [svm.NU_SVR, svm.NU_SVC, svm.C_SVC, svm.ONE_CLASS, svm.EPSILON_SVR]
		#kernels = [svm.LINEAR, svm.POLY, svm.RBF, svm.SIGMOID, svm.PRECOMPUTED]
		param = svm.svm_parameter(svm_type = svm.ONE_CLASS, kernel_type = svm.RBF)
		model = svm.svm_model(problem, param)

		#4) make some prediction :-s
		groundTruth = [1,1,1,1,1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
		error       = 0.0
		for i in range(0, test.shape[1]):
			prediction = model.predict(test[0:noComp, i].tolist())
			if(int(prediction) != int(groundTruth[i])):
				error += 1.0 
			print prediction

		error = float(error)/float(len(groundTruth))
		print error			
		print noComp
		return model
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
classi.classifyHands(1000)

