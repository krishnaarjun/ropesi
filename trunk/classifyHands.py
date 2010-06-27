# Classifies an image as containing a "rock,paper,scissors" sign or not
# Classifies between the signs in order to distinguish between the 3 signs
#
# Input:
# - pca  = an object of the class "eigenHands" 
# - size = size of the images
# Output:
# - mergeLists(list1, list2)            => merges 2 lists: [list1, list2]
# - classifySVN(noComp,onImg,theSign)   => classifies hands from non-hands using the eigenHands defined by the "noComp" for the data given by "theSign" 
#					   "onImg" -- 1 for the reshaped image to DxD
#					   	   -- 2 for the image with PCA
#						   -- 3 for the convolved Gabors + original image         
#						   -- 4 for the convolved Gabors                    
# - classifyKNN(noComp, onImg, theSign) => classifies hands like above using Knn	 	 		
# - getDataLabels(noComp,onImg,theSign, => returns data-set and labels corresponding to the option "onImg"; 
#	isMulti) 	                   "theSign" -- is "hands" for hands vs no-hands
#                                                    -- is "rock" for rock vs paper&scissors
#                                                    -- is "paper" for papaer vs scissors  
# - myFolds(labels, classes, sets)      => creates a numer of "sets" folds over the indexes of the labels for the number of classe "classes"		
	
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
#from PIL import Image
from eigenHands import *
from numpy.random import shuffle
class classifyHands:
	def __init__(self, makeData, size):
		self.pca = eigenHands(size)
		#create the data matrix if they are not there
		if(makeData == True):
	    		self.pca.makeMatrix("garb")
	    		self.pca.makeMatrix("hands")
	    		self.pca.makeMatrix("rock")
	    		self.pca.makeMatrix("paper")
	    		self.pca.makeMatrix("scissors")
	#________________________________________________________________________
	#get the training set and the labels 
	def getDataLabels(self, onImg, theSign, isMulti):
		if(isMulti == True):
			signs = {"hands":["garb"], "rock":["paper", "scissors"]}
		else:
			signs = {"hands":["garb"], "rock":["paper", "scissors"], "paper":["scissors"]}

		if(onImg == 1):
			good = self.pca.justGetDataMat(theSign,"",True)
		elif(onImg == 2):
			good = self.pca.justGetDataMat(theSign,"PCA/",True)
		elif(onImg == 3):
			good = self.pca.justGetDataMat(theSign,"GaborImg/",True)
		elif(onImg == 4):
			good = self.pca.justGetDataMat(theSign,"Gabor/",True)
		bad     = []
		badSize = 0
		for aSign in signs[theSign]:
			if(onImg == 1):
				preBad = self.pca.justGetDataMat(aSign,"",True)
			elif(onImg == 2):
				preBad = self.pca.justGetDataMat(aSign,"PCA/",True)
			elif(onImg == 3):
				preBad = self.pca.justGetDataMat(aSign,"GaborImg/",True)
			elif(onImg == 4):
				preBad = self.pca.justGetDataMat(aSign,"Gabor/",True)
			badSize += preBad.shape[0]
			bad.append(preBad)
		labels    = numpy.ones(good.shape[0]+badSize, dtype=int)
		train     = numpy.empty((good.shape[0]+badSize, good.shape[1]), dtype=float)
		indexs    = numpy.empty(good.shape[0]+badSize, dtype=int)
		sizeSoFar = 0
		j         = 0	
		for i in range(0, good.shape[0]+badSize):
			indexs[i] = i
			if(i<good.shape[0]):
				labels[i]  = 1 				
				train[i,:] = good[i,:]
			else:
				if(isMulti == False):
					labels[i] = -1
				ind = (i-good.shape[0])-sizeSoFar
				if((ind-bad[j].shape[0]) == 0):
					sizeSoFar += bad[j].shape[0]
					j         += 1
					ind        = (i-good.shape[0])-sizeSoFar
				train[i,:] = bad[j][ind,:]
				if(isMulti == True):
					labels[i] += (j+1)
		return indexs,labels,train

	#________________________________________________________________________
	#classify images using svm
	def classifySVM(self, onImg, theSign):
		#0) get training data data and the labels
		indexs,labels,train = self.getDataLabels(onImg, theSign, False)

		#1) initialize the svm and compute the model
		if(onImg == 1): #for full images 		
			problem = mlpy.Svm(kernel='gaussian', C=1.0, kp=0.1, tol=0.001, eps=0.001, maxloops=1000, opt_offset=True)
		elif(onImg == 2): #for PCAed images
			problem = mlpy.Svm(kernel='gaussian', C=1.0, kp=0.1, tol=0.001, eps=0.001, maxloops=1000, opt_offset=True)
		elif(onImg == 3): #for gabore filters
			problem = mlpy.Svm(kernel='polynomial', kp=0.3, C=1.0, tol=0.0001, eps=0.0001, maxloops=1000, opt_offset=True)

		#2) shuffle input data to do the 10-fold split 
		shuffle(indexs)
		labels = labels[indexs]
		train  = train[indexs,:] 

		#3) define the folds, train and test
		pred_err = 0.0
		folds    = mlpy.kfoldS(cl = labels, sets = 50, rseed = random.random())
		for (trainI,testI) in folds:
			trainSet, testSet = train[trainI], train[testI]
			trainLab, testLab = labels[trainI], labels[testI]			
			learned           = problem.compute(trainSet, trainLab)
			print "it learned >>> "+str(learned)
			prediction        = problem.predict(testSet)
			print prediction
			pred_err         += mlpy.err(testLab, prediction)
			print pred_err
		avg_err = float(pred_err)/float(len(folds))
		print "\nAverage error over 50 folds:"+str(avg_err)
		return problem
	#________________________________________________________________________
	#classify images using KNN
	def classifyKNN(self, onImg, theSign, neighbors):
		#0) get training data data and the labels
		indexs,labels,train = self.getDataLabels(onImg, theSign, True)

		#1) initialize the svm and compute the model
		problem = mlpy.Knn(neighbors, dist='se')

		#2) shuffle input data to do the 10-fold split 
		shuffle(indexs)
		labels = labels[indexs]
		train  = train[indexs,:] 

		#3) define the folds, train and test
		pred_err     = 0.0
		err_rock     = 0.0
		err_paper    = 0.0
		err_scissors = 0.0
		fold_ind = 0
		folds    = self.myFolds(labels, [1,2,3], 50)
		for (trainI,testI) in folds:
			fold_ind += 1
			trainSet, testSet = train[trainI], train[testI]
			trainLab, testLab = labels[trainI], labels[testI]			
			learned           = problem.compute(trainSet, trainLab)
			print "it learned ["+str(fold_ind)+"] >>> "+str(learned)

	 		zaTime     = cv.GetTickCount() 
			totalTime  = 0
			prediction = problem.predict(testSet)
			wrong      = numpy.where(numpy.array(prediction) != numpy.array(testLab))[0]
			if(theSign == "rock"):
				labWrong       = testLab[map(None,wrong)]
				wrong_rock     = numpy.where(labWrong == 1)[0]
				wrong_paper    = numpy.where(labWrong == 2)[0]
				wrong_scissors = numpy.where(labWrong == 3)[0]
				rocks          = numpy.where(testLab == 1)[0]
				papers         = numpy.where(testLab == 2)[0]
				scissors       = numpy.where(testLab == 3)[0]
				err_rock      += float(float(wrong_rock.shape[0])/float(len(rocks)))
				err_paper     += float(float(wrong_paper.shape[0])/float(len(papers)))
				err_scissors  += float(float(wrong_scissors.shape[0])/float(len(scissors)))
			pred_err  += float(float(wrong.shape[0])/float(len(testLab)))
			print prediction
			print testLab
			zaTime     = cv.GetTickCount() - zaTime
	    		totalTime += zaTime/(cv.GetTickFrequency()*1000.0*float(len(testLab)))
			print "cumulative error %f >>> prediction time/image %gms" % (pred_err, totalTime)                	
		avg_err = float(pred_err)/float(len(folds))
		if(theSign == "rock"):		
			avg_rock     = float(err_rock)/float(len(folds))
			avg_paper    = float(err_paper)/float(len(folds))
			avg_scissors = float(err_scissors)/float(len(folds))

		print "\nAvg Error:"+str(avg_err)+" >>> Avg Rock Error:"+str(avg_rock)+" >>> Avg Paper Error:"+str(avg_paper)+" >>> Avg Scissors Error:"+str(avg_scissors)
		return problem
	#________________________________________________________________________
	#implements k bolds for multiple classes 
	def myFolds(self, labels, classes, sets):
		#1) get the indexes of the data
		orderI = numpy.ogrid[0:labels.shape[0]:1]

		#2) put one smaple of each class in each fold in train and test
		folds    = []
		foldSize = int(labels.shape[0]/sets)

		for fold in range(0, sets):
			testI1  = []
			testI2  = []
			testI   = []
			trainI1 = []
			trainI2 = []
			trainI  = []
			for aClass in classes:	
				classInd = (numpy.where(labels == aClass))[0]
				indx     = random.randint(0,len(classInd)-1)
				trainI1.append(classInd[indx])
				newindx  = random.randint(0,len(classInd)-1)
				while(newindx == indx):
					newindx = random.randint(0,len(classInd)-1)
				testI1.append(classInd[newindx])
			startTest = fold*foldSize
			endTest   = min(((fold+1)*foldSize), labels.shape[0])			
			testI2  = map(None,orderI[startTest:endTest])
			comSpls = len(filter(lambda x:x in testI1,testI2))
			testI2  = map(None,orderI[startTest:endTest+comSpls])
			testI2  = filter(lambda x:x not in testI1,testI2)
			testI   = self.mergeLists(testI1,testI2)	
			trainI2 = filter(lambda x:x not in testI,map(None,orderI))
			trainI2 = filter(lambda x:x not in trainI1,trainI2)
			trainI  = self.mergeLists(trainI1,trainI2)	
			folds.append((trainI,testI))
		return folds	 
	#________________________________________________________________________
	#merges 2 lists: list1, list2
	def mergeLists(self, list1, list2):
		merged = []
		for aList in [list1, list2]:
			for elem in aList:
				merged.append(elem)
		return merged
#________________________________________________________________________


