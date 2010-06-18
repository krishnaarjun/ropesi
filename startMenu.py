# Calls some functions given some options:
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
from gaborFilters import *
from preprocessing import *
from classifyHands import *
from predictSign import *

print "Please choose:"
print "\t1 => check PCA on original images"
print "\t2 => check Gabor Wavelets"
print "\t3 => extend the training data"
print "\t4 => generate data convolved with multiple Gabor wavelets"
print "\t5 => generate data convolved with multiple Gabor wavelets + concatenated with the original image"
print "\t6 => do some classifications (SVN)"
print "\t7 => do some classifications (Knn)"
print "\t8 => create the FINAL models for classifiers\n"
choice  = raw_input('your choice... ') 
sizeImg = raw_input('the size of the training images ...')   
build   = raw_input('build the training matrixes (y/n) ...')   
print "\n"
buildOpt = {'y':True, 'n':False}
if(int(choice) == 1):
	dataset = raw_input('choose the dataset (r/p/s) ...')   
	noComp  = raw_input('number of components for PCA ...')         
	datas   = {'r':'rock', 'p':'paper', 's':'scissors'} 
	hands   = eigenHands(int(sizeImg))
	if(build == 'y'):
		hands.makeMatrix(datas[dataset])
	X = hands.justGetDataMat(datas[dataset])
	hands.doPCA(X, int(noComp), True, datas[dataset])
elif(int(choice) == 2):
	#lambda=[2,256], gamma=[0.2,1], psi=[0,180], theta=[0,180], sigma=[3,68]	
	"""print "you need to define the parameters of the Gabor Wavelet"
	lambd   = raw_input('define lambda (2 - 256): ...') 	
	gamma   = raw_input('define gamma (0.2 - 1): ...') 
	psi     = raw_input('define psi (0 - 180): ...') 
	theta   = raw_input('define theta (-90 - 90): ...') 
	sigma   = raw_input('define gamma (3 - 68): ...')
	dim     = raw_input('define dimension (2 - 10): ...')"""
	dataset = raw_input('choose the dataset (r/p/s) ...')
	datas   = {'r':'rock', 'p':'paper', 's':'scissors'} 
	gabor   = gaborFilters(buildOpt[str(build)],int(sizeImg))
	#gabor.setParameters(float(lambd), float(gamma), int(psi), int(theta), float(sigma), int(dim))
	gabor.setParameters(10, 1.0, 10, -45, 8.0, 2)
	data    = cv.Load("data_train/"+datas[dataset]+"Train.dat")
	gabor.convolveImg(data, True)
elif(int(choice) == 3):
	aNumber = raw_input('write an unused nr/word ...')  
	prep    = preprocessing(int(sizeImg))
	prep.getHandsVideo(aNumber)
elif(int(choice) == 4):
	dataset = raw_input('choose the dataset (r/p/s) ...')
	noComp1 = raw_input('number of components for PCA no 1...')
	noComp2 = raw_input('number of components for PCA no 2...') 		 		
	datas   = {'r':'rock', 'p':'paper', 's':'scissors'} 	
	prep    = preprocessing(int(sizeImg))
	data    = cv.Load("data_train/"+datas[dataset]+"Train"+str(sizeImg)+".dat")
	prep.doManyGabors(data, datas[dataset], int(noComp1), int(noComp2), True, False)
elif(int(choice) == 5):
	dataset = raw_input('choose the dataset (r/p/s) ...')
	noComp  = raw_input('number of components for PCA ...')
	datas   = {'r':'rock', 'p':'paper', 's':'scissors'} 	
	prep    = preprocessing(int(sizeImg))
	data    = cv.Load("data_train/"+datas[dataset]+"Train"+str(sizeImg)+".dat")
	prep.doSmallManyGabors(data, datas[dataset], int(noComp), True, False)
elif(int(choice) == 6):
	dataset = raw_input('classify h => hands vs garbage; r => rock vs paper & scissors; p => paper vb scissors ...')	
	typeu   = raw_input('choose the data 1 => original images; 2 => PCA on initial images; 3 => multiple Gabor filters + orig img; 4 => just multiple Gabor Filters...')
	datas  = {'r':'rock', 'h':'hands', 'p':'paper'} 
	classi = classifyHands(buildOpt[str(build)],int(sizeImg))	
	classi.classifySVM(int(typeu), datas[dataset])
elif(int(choice) == 7):
	dataset = raw_input('classify h => hands vs garbage; c => rock & paper & scissors ...')	
	typeu   = raw_input('choose the data 1 => original images; 2 => PCA on initial images; 3 => multiple Gabor filters + orig img; 4 => just multiple Gabor Filters...')
	datas  = {'c':'rock', 'h':'hands'} 
	classi = classifyHands(buildOpt[str(build)],int(sizeImg))	
	classi.classifyKNN(int(typeu), datas[dataset], 5)
elif(int(choice) == 8):
	model    = raw_input('model to built: s => svm; k => knn ...')	
	dataset  = raw_input('classify h => hands vs garbage; c => rock & paper & scissors ...')	
	datas    = {'c':'rock', 'h':'hands'} 
	modelOpt = {'s':'svm', 'k':'knn'} 
	typeu    = raw_input('choose the data 1 => original images; 2 => PCA on initial images; 3 => multiple Gabor filters + orig img; 4 => just multiple Gabor Filters...')
	predict  = predictSign(int(sizeImg),buildOpt[str(build)])
	predict.storeModel(modelOpt[str(model)], datas[dataset], int(typeu))
	

