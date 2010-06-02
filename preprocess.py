"""Class for all the preprocessing on the images => center the hand in the middle, scale"""

import sys
import urllib
import re
import copy
import math
import random
"""for openCV ;)"""
import cv
from opencv.cv import *
from opencv.highgui import *
"""from some_class import *"""
class preprocessing:
	def __init__(self):
		dirImages   = ""
		scaleWidth  = 200
		scaleHeight = 200
	"""_________________________________________________________________"""
	"""does the scaling of all the images that it reads from a direcotry"""
	def scale(self):
		cvStartWindowThread()
   		cvNamedWindow("win")
   		im = cvLoadImage("building.jpg")
   		cvShowImage("win", im)
		cvWaitKey()
		print "Nothing -- scale ;)"
	"""_________________________________________________________________"""
	"""centers the hands in the images"""
	def center(self):
		print "Nothing -- center ;)"
"""_________________________________________________________________________"""
pre = preprocessing()
pre.scale()
