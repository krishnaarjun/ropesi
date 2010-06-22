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
        alPath = "C:\Program Files\Aldebaran\Choregraphe 1.6.13\lib"
        sys.path.append(alPath)
        import naoqi
        from naoqi import ALBroker
        from naoqi import ALModule
        from naoqi import ALProxy
        from naoqi import ALBehavior 
        
BASEPATH = "/home/nao/behaviors/"

import cv
from skinFinder import *

#_____________________________________________________________________
#1) CONNECT TO NAO
goNao = Gesture()
skin  = detectSkin()

#2) START WITH A DEMO
goNao.naoBehaviors("demo")

for i in range(0,3):
	#3) LET'S PLAY NOW
	goNao.naoBehaviors("play")


	#3) LET'S PLAY NOW
	goNao.naoBehaviors("play")

	#4) PREDICT WHO PLAYED WHAT
	goNao.naoBehaviors("move")
	moves = {0:"rock", 1:"paper", 2:"scissors"}
	if(skin.maximum == 10):
		if(skin.maximum == moves(goNao.naoMove)) or (): #0=rock, 1=paper, 2=scissors
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
		skin.predictions = {"rock":0, "paper":0, "scissors":0, "garb":0}
		skin.maximum     = ""






