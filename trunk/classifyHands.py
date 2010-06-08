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
import svm
from PIL import Image
from eigenHands import *
class classifyHands:
	def __init__(self, makeData):
		self.pca = eigenHands()
		#create the data matrix if they are not there
		self.pca.makeMatrix("test")
		#self.pca.makeMatrix("hands")
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

