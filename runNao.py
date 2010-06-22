import os
import sys
import random
path = 'os.environ.get("AL_DIR")'
home = 'os.environ.get("HOME")'

if path == "None":
        print "the environment variable AL_DIR is not set, aborting..."
        sys.exit(1)
else:
        #import naoqi lib
	alPath = "/data/Documents/semester2/AIProject/nao/lib"
        #alPath = "C:\Program Files\Aldebaran\Choregraphe 1.6.13\lib"
        sys.path.append(alPath)
        import naoqi
        from naoqi import ALBroker
        from naoqi import ALModule
        from naoqi import ALProxy
        from naoqi import ALBehavior         
BASEPATH = "/home/nao/behaviors/"

import cv
from skinFinder import *
from naoMotions import *
import thread

#_____________________________________________________________________
#1) CONNECT TO NAO AND INITIALIZE CLASS SKIN
skin  = detectSkin()
goNao = Gesture()

#2) MAKE MULTI THREADS SO THE SKIN FINDER AND NAO INITIALIZE AND RUN IN PARALLEL
try:
	thread.start_new_thread(skin.findSkin, ())
except:
	print "error for skin finder"

goNao.naoBehaviors("demo")
for i in range(0,3):
	thread.start_new_thread(skin.reinitGame, ())

	#3) LET'S PLAY NOW___________________
	goNao.naoBehaviors("play")
	
	#4) PREDICT WHO PLAYED WHAT_________________	
	goNao.naoBehaviors("move")

	moves = {0:"rock", 1:"paper", 2:"scissors"}

	print str(skin.maximum)+" "+str(skin.maxnr)

	if(skin.maxnr >= 2 and skin.maximum != "none" and skin.maximum != "garb"):
		if(skin.maximum == moves[int(goNao.naoMove)]): #0=rock, 1=paper, 2=scissors
			goNao.naoBehaviors("equal")
		elif(skin.maximum == "rock" and goNao.naoMove == 1): #rock>scissors>paper>rock
			goNao.naoBehaviors("lost")
		elif(skin.maximum == "rock" and goNao.naoMove == 2):
			goNao.naoBehaviors("won")
		elif(skin.maximum == "paper" and goNao.naoMove == 0):
			goNao.naoBehaviors("won")
		elif(skin.maximum == "paper" and goNao.naoMove == 2):
			goNao.naoBehaviors("lost")
		elif(skin.maximum == "scissors" and goNao.naoMove == 0):
			goNao.naoBehaviors("lost")
		elif(skin.maximum == "scissors" and goNao.naoMove == 1):
			goNao.naoBehaviors("won")
		#5) RESET THE VARIABLES 
		thread.start_new_thread(skin.reinitGame, ())
	




