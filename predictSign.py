# Combines the hand-detection part with sign classification
#
# Input:
# - pca   = an object of the class "eigenHands"
# - gabor = an object of the class "gaborFilters"
# - image = image for which the prediciton needs to be done 	
#
# Output:
# - doPrediction(type,model,problem,what,image) => predicts the class of "image"; 
#						   model   -- "svm"/"knn"
#						   problem -- is the mlpy problem stored in "classi_models"
#						   what    -- "rock" for "rock & paper & scissors";
#							   -- "hands" for "hands vs garbage"
#						   type    -- 1 = on original images, 
#							      2 = on eigen-hands of original images
#							      3 = on convolved Gabors + original image
#							      4 = just on convolved Gabors
# - storeModel(model, theSign, onImg) => stores a model for a model ("svn"/"knn"); 
#					 theSign -- "rock" for "rock & paper & scissors";
#					         -- "hands" for "hands vs garbage"
#					 type    -- 1 = on original images, 
#					            2 = on eigen-hands of original images
#						    3 = on convolved Gabors + original image
#					            4 = just on convolved Gabors
# - loadModel(model, type)            => load the model given my model ("svn"/"knn") and type (1/2/3/4)
# - preprocessImg(image, type)        => preprocesses the test image corresponding to the type (1/2/3/4)
# 
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
import pickle
from eigenHands import *
from gaborFilters import *
from preprocessing import *
from classifyHands import *

class predictSign:
	def __init__(self, size, makeData, noComp):
		self.pca      = eigenHands(size)
		self.gabor    = gaborFilters(False, size)
		self.classify = classifyHands(False, size)
		self.prep     = preprocessing(size, noComp)
		if(makeData == True):
	    		self.pca.makeMatrix("garb")
	    		self.pca.makeMatrix("hands")
	    		self.pca.makeMatrix("rock")
	    		self.pca.makeMatrix("paper")
	    		self.pca.makeMatrix("scissors")

	#________________________________________________________________________
	#store the models
	def storeModel(self, model, theSign, onImg):
		if(model == "svm"):
			#0) get training data data and the labels
			indexs,labels,train = self.classify.getDataLabels(onImg, theSign, False)

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

			#3) generate the svm model
			learned   = problem.compute(train, labels)
			modelFile = open("classi_models/"+str(self.pca.sizeImg)+"SVM_"+str(theSign)+str(onImg)+".dat", "wb")
			pickle.dump(problem, modelFile)
			modelFile.close()			
		else:
			#0) get training data data and the labels
			if(theSign == "rock"):				
				indexs,labels,train = self.classify.getDataLabels(onImg, theSign, True)
			else:
				indexs,labels,train = self.classify.getDataLabels(onImg, theSign, False)
		
			print train.shape

			#1) initialize the svm and compute the model
			problem = mlpy.Knn(4, dist='se')

			#2) shuffle input data to do the 10-fold split 
			shuffle(indexs)
			labels = labels[indexs]
			train  = train[indexs,:] 

			#3) generate the Knn model
			learned   = problem.compute(train, labels)			

			print train.shape

			modelFile = open("classi_models/"+str(self.pca.sizeImg)+"Knn_"+str(theSign)+str(onImg)+".dat", "wb")
			pickle.dump(problem, modelFile)
			modelFile.close()
	#________________________________________________________________________
	#just load model
	def loadModel(self, model, zaType, theSign):
		if(model == "svm"):
			modelFile = open("classi_models/"+str(self.pca.sizeImg)+"SVM_"+str(theSign)+str(zaType)+".dat", "r")
		else:
			modelFile = open("classi_models/"+str(self.pca.sizeImg)+"Knn_"+str(theSign)+str(zaType)+".dat", "r")
		problem = pickle.load(modelFile)
		modelFile.close()
		return problem
	#________________________________________________________________________
	#does the prediction over the given image
	def doPrediction(self, zaType, model, problem, what, image):
		totalTime = 0
		if(model == "svm"):
			classifyWhat = {"hands":{"1":"hands", "-1":"garb"}, "rock": {"1":"rock", "-1":"paper or scissors"}, "paper":{"1":"paper", "-1":"scissors"}}
		else:
			classifyWhat = {"hands":{"1":"hands", "-1":"garb", "0":"none"}, "rock":{"1":"rock", "2":"paper", "3":"scissors", "0":"none"}}
		zaTime     = cv.GetTickCount() 
		testImg    = self.preprocessImg(image, zaType)
		prediction = problem.predict(testImg)

		print prediction

		zaTime     = cv.GetTickCount() - zaTime
	    	totalTime += zaTime/(cv.GetTickFrequency()*1000.0)
		if((what!="hands") or (prediction[0]==0 and what=="hands")):	
			print "it is a..."+str(classifyWhat[str(what)][str(prediction[0])])+" >>> prediction time/image %gms" % totalTime
		return str(classifyWhat[str(what)][str(prediction[0])])
	#________________________________________________________________________
	def preprocessImg(self, image, zaType):
		#1) resize the image to the needed size
		resizeImg = cv.CreateMat(self.pca.sizeImg, self.pca.sizeImg, cv.CV_8UC1)
		cv.Resize(image, resizeImg, interpolation = cv.CV_INTER_AREA)	

		#2) reshape the image as a row
		imgMat = cv.Reshape(resizeImg, 0, 1)

		if(zaType == 1): # original images
			return self.pca.cv2array(imgMat,True)
		elif(zaType == 2): # PCA over the original images
			return self.pca.projPCA(self.pca.cv2array(imgMat,True), False, "PCA/", "")		
		elif(zaType == 3): # convolve with 4 kernels and add the original image -- NO PCA
			return self.prep.doSmallManyGabors(self.pca.cv2array(imgMat,True), None, "",False)	
		elif(zaType == 4): # convolve with 4 kernels -- NO PCA
			return self.prep.doManyGabors(self.pca.cv2array(imgMat,True), None, "",False)			

		#elif(zaType == 5): # convolve with 4 kernels and add the original image and PCA
		#	return self.prep.doManyGabors(self.pca.cv2array(imgMat,True), None, "",True)
		#elif(zaType == 6): # convolve with 4 kernels and PCA
		#	return self.prep.doManyGabors(self.pca.cv2array(imgMat,True), None, "",True)			

#________________________________________________________________________
	












