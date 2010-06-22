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
choice   = raw_input('your choice... ') 
sizeImg  = raw_input('the size of the training images ...')   
build    = raw_input('build the training matrixes (y/n) ...')   
buildOpt = {'y':True, 'n':False}
print "\n"
#____________________________________________________________________________________________________
#____________________________________________________________________________________________________
#____________________________________________________________________________________________________
if(buildOpt == "y"):
	datas = ['rock','paper','scissors','hands','garb']
	for aset in data:	
		hands = eigenHands(int(sizeImg))	
		hands.makeMatrix(aset)
#____________________________________________________________________________________________________

if(int(choice) == 1):
	dataset = raw_input('choose the dataset c= > rock & paper & scissors; h => hands vs garbage ...')   
	noComp  = raw_input('number of components for PCA ...')         
	datas   = {'c':['rock','paper','scissors'], 'h':['hands','garb']} 
	hands   = eigenHands(int(sizeImg))
	_,X,_ = hands.justGetDataMat(datas[dataset][0],"",False)
	hands.doPCA(X, int(noComp), "PCA/")
	for i in range(0,len(datas[dataset])):
		projData = hands.justGetDataMat(datas[dataset][i],"",True)
		hands.projPCA(projData, False, "PCA/", datas[dataset][i])
#____________________________________________________________________________________________________

elif(int(choice) == 2):
	dataset = raw_input('choose the dataset (r/p/s) ...')
	datas   = {'r':'rock', 'p':'paper', 's':'scissors'} 
	gabor   = gaborFilters(buildOpt[str(build)],int(sizeImg))
	gabor.setParameters(10, 1.0, 10, -45, 8.0, 2)
	data    = cv.Load("data_train/"+datas[dataset]+"Train"+str(sizeImg)+".dat")
	gabor.convolveImg(data, True)
#____________________________________________________________________________________________________

elif(int(choice) == 3):
	aNumber = raw_input('write an unused nr/word ...')  
	prep    = preprocessing(int(sizeImg),0,0)
	prep.getHandsVideo(aNumber)
#____________________________________________________________________________________________________

elif(int(choice) == 4):
	noComp = raw_input('number of components for PCA no ...')
	dataset = raw_input('choose the dataset c= > rock & paper & scissors; h => hands vs garbage ...')   
	datas   = {'c':['rock','paper','scissors'], 'h':['hands','garb']} 
	hands   = eigenHands(int(sizeImg))
	_,data,txtLabels = hands.justGetDataMat(datas[dataset][0],"",False)
	prep    = preprocessing(int(sizeImg),int(noComp))
	prep.doManyGabors(data,txtLabels,dataset, False)
#____________________________________________________________________________________________________

elif(int(choice) == 5):
	noComp  = raw_input('number of components for PCA ...')
	dataset = raw_input('choose the dataset c= > rock & paper & scissors; h => hands vs garbage ...')   
	datas   = {'c':['rock','paper','scissors'], 'h':['hands','garb']} 
	hands   = eigenHands(int(sizeImg))
	_,data,txtLabels = hands.justGetDataMat(datas[dataset][0],"",False)
	prep    = preprocessing(int(sizeImg),int(noComp))
	prep.doSmallManyGabors(data,txtLabels,dataset,False)
#____________________________________________________________________________________________________

elif(int(choice) == 6):
	dataset = raw_input('classify h => hands vs garbage; r => rock vs paper & scissors; p => paper vb scissors ...')	
	typeu   = raw_input('choose the data 1 => original images; 2 => PCA on initial images; 3 => multiple Gabor filters + orig img; 4 => just multiple Gabor Filters...')
	datas  = {'r':'rock', 'h':'hands', 'p':'paper'} 
	classi = classifyHands(buildOpt[str(build)],int(sizeImg))	
	classi.classifySVM(int(typeu), datas[dataset])
#____________________________________________________________________________________________________

elif(int(choice) == 7):
	dataset = raw_input('classify h => hands vs garbage; c => rock & paper & scissors ...')	
	typeu   = raw_input('choose the data 1 => original images; 2 => PCA on initial images; 3 => multiple Gabor filters + orig img; 4 => just multiple Gabor Filters...')
	datas  = {'c':'rock', 'h':'hands'} 
	classi = classifyHands(buildOpt[str(build)],int(sizeImg))	
	classi.classifyKNN(int(typeu), datas[dataset],4)
#____________________________________________________________________________________________________

elif(int(choice) == 8):
	model    = raw_input('model to built: s => svm; k => knn ...')	
	dataset  = raw_input('classify h => hands vs garbage; c => rock & paper & scissors ...')	
	datas    = {'c':'rock', 'h':'hands'} 
	modelOpt = {'s':'svm', 'k':'knn'} 
	typeu    = raw_input('choose the data 1 => original images; 2 => PCA on initial images; 3 => multiple Gabor filters + orig img; 4 => just multiple Gabor Filters...')
	if(typeu == "3" or typeu == 4):
		noComp = raw_input('number of components for PCA no ...')
	else:
		noComp = 0
	predict  = predictSign(int(sizeImg),buildOpt[str(build)], int(noComp))
	predict.storeModel(modelOpt[str(model)], datas[dataset], int(typeu))
	

