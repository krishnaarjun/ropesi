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

print "Please choose:"
print "\t1 => check PCA on original images"
print "\t2 => check Gabor Wavelets"
print "\t3 => extend the training data"
print "\t4 => generate data convolved with multiple Gabor wavelets"
print "\t5 => do some classifications\n"
choice = raw_input('your choice... ') 
print "\n"
if(int(choice) == 1):
	dataset = raw_input('choose the dataset (r/p/s) ...') 	
	noComp  = raw_input('number of components for PCA ...') 		
	datas   = {'r':'rock', 'p':'paper', 's':'scissors'}	
	hands   = eigenHands()
	hands.makeMatrix(datas[dataset])
	X = hands.justGetDataMat(datas[dataset])
	hands.doPCA(X, int(noComp), True)
elif(int(choice) == 2):
	#lambda=[2,256], gamma=[0.2,1], psi=[0,180], theta=[0,180], sigma=[3,68]	
	print "you need to define the parameters of the Gabor Wavelet"
	lambd   = raw_input('define lambda (2 - 256): ...') 	
	gamma   = raw_input('define gamma (0.2 - 1): ...') 
	psi     = raw_input('define psi (0 - 180): ...') 
	theta   = raw_input('define theta (-90 - 90): ...') 
	sigma   = raw_input('define gamma (3 - 68): ...')
	dim     = raw_input('define dimension (2 - 10): ...')
	dataset = raw_input('choose the dataset (r/p/s) ...')
	datas   = {'r':'rock', 'p':'paper', 's':'scissors'} 
	gabor   = gaborFilters(True)
	gabor.setParameters(float(lambd), float(gamma), int(psi), int(theta), float(sigma), int(dim))
	data    = cv.Load("data_train/"+datas[dataset]+"Train.dat")
	gabor.convolveImg(data, True)
elif(int(choice) == 3):
	aNumber = raw_input('write an unused nr/word ...') 	
	prep = preprocessing()
	prep.getHandsVideo(aNumber)
elif(int(choice) == 4):
	dataset = raw_input('choose the dataset (r/p/s) ...')
	noComp1 = raw_input('number of components for PCA no 1...')
	noComp2 = raw_input('number of components for PCA no 2...') 		 		
	datas   = {'r':'rock', 'p':'paper', 's':'scissors'} 	
	prep = preprocessing()
	prep.doManyGabors(datas[dataset], int(noComp1), int(noComp2), True)
elif(int(choice) == 5):
	dataset = raw_input('classify h => hands vs garbage; r => rock vs paper&scissors; p => paper vb scissors ...')	
	typeu   = raw_input('choose the data 1 => original images; 2 => PCA on initial images; 3 => multiple Gabor filters...')
	if(int(typeu) == 2):
		noComp = raw_input('number of components for PCA...')
	else:
		noComp = 1500 
	datas   = {'r':'rock', 'h':'hands', 'p':'paper'} 
	classi = classifyHands(True)	
	classi.classifyHands(int(noComp), int(typeu), datas[dataset])







