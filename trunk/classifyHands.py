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
		if(makeData == True):
			self.pca.makeMatrix("hands")
			self.pca.makeMatrix("garb")
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
		hands, _, _ = self.pca.doPCA("hands", noComp)
		garb, _, _  = self.pca.doPCA("garb", noComp)
		
		#1) define the problem: labels & samples 
		labels  = []
		samples = []
		for i in range(0, hands.shape[0]):
			samples.append(hands[i,:].tolist())
			labels.append(1)
		for j in range(0, garb.shape[0]):
			samples.append(garb[j,:].tolist())
			labels.append(0)

		#2) store the problem :-/
		problem = svm.svm_problem(labels, samples)
		size    = len(samples)

		#3) generate the model from the data 
		#kernels = [svm.LINEAR, svm.POLY, svm.RBF, svm.SIGMOID, svm.EPSILON_SVR, svm.NU_SVR, svm.NU_SVC, svm.C_SVC]
		param = svm.svm_parameter(kernel_type = svm.RBF, probability = 1)
		model = svm.svm_model(problem, param)

		#4) make some prediction :-s
		test, _, _ = self.pca.doPCA("test", noComp)
		for i in range(0, test.shape[0]):
			pred_label, pred_probability = model.predict_probability(test[i].tolist())
			if(pred_label == 1):
				labelName = "hands"
			else:
				labelName = "garb"
			print "label:"+str(labelName)+"["+str(pred_label)+"] >>> prob:"+str(pred_probability)
		return model
	#________________________________________________________________________
	#get the training set from video of hands
	def classifySigns(self):
		""".."""
#________________________________________________________________________
classi = classifyHands(False)
classi.classifyHands(13)

