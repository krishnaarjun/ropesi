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
	alPath = "/data/Documents/nao/lib"
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
from threading import Thread
import threading

#_____________________________________________________________________
#1) CONNECT TO NAO AND INITIALIZE CLASS SKIN
ipAdd = raw_input("Please write Nao's IP address... ") 
print "\nPlease choose an action:"
print "\td => Run the demonstration of Nao's gestures during the play"
print "\tp => Play rock,paper&scissors with Nao"
choice = raw_input('your choice... ') 
if(choice == "d"):
	goNao = Gesture(ipAdd, 9559)
	goNao.naoBehaviors("demo")
	goNao.releaseNao()

elif(choice == "p"):
	skin  = detectSkin()
	goNao = Gesture(ipAdd, 9559)

	#2) MAKE MULTI THREADS SO THE SKIN FINDER AND NAO INITIALIZE AND RUN IN PARALLEL	
	skinLock = threading.Lock()
	skinLock.acquire(1)
	try:
		skinThread = Thread(target=skin.findSkin, args = ())
		skinThread.start()
	except Exception,e:
		print "error for skin finder"
		skinLock.release()
	skinLock.release()
	
	for i in range(0,3):
		#4) RESET THE VARIABLES
		skinLock.acquire(1)
		try:
			initThread = Thread(target=skin.reinitGame, args = ())
			initThread.start()
		except Exception,e:
			skinLock.release()
			print "error for reinitilising the counts"		
		skinLock.release()

		#3) LET'S PLAY NOW___________________	
		goNao.naoBehaviors("play")
		moves = {0:"rock", 1:"paper", 2:"scissors"}

		print str(skin.maximum)+"["+str(skin.maxnr)+"] >>> vs >>> "+str(moves[int(goNao.naoMove)])
		if(skin.maxnr >= 2 and skin.maximum != "none" and skin.maximum != "garb"):
			if(skin.maximum == moves[int(goNao.naoMove)]): #0=rock, 1=paper, 2=scissors
				goNao.naoBehaviors("draw")
			elif(skin.maximum == "rock" and goNao.naoMove == 1): #rock>scissors>paper>rock
				goNao.naoBehaviors("win")
			elif(skin.maximum == "rock" and goNao.naoMove == 2):
				goNao.naoBehaviors("loose")
			elif(skin.maximum == "paper" and goNao.naoMove == 0):
				goNao.naoBehaviors("loose")
			elif(skin.maximum == "paper" and goNao.naoMove == 2):
				goNao.naoBehaviors("win")
			elif(skin.maximum == "scissors" and goNao.naoMove == 0):
				goNao.naoBehaviors("win")
			elif(skin.maximum == "scissors" and goNao.naoMove == 1):
				goNao.naoBehaviors("loose")
		initThread.join()
		if(i == 2):
			skinLock.acquire(1)
			skin.stopGame = True	
			skinLock.release()
	skinThread.join()	
	goNao.releaseNao()

#______________________________________________________________________________




