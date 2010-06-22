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
#____________________________________________________________

class Gesture:
	def __init__(self):
		self.lastvalue = [0,0,0]
		self.position  = "stand"
		self.counter   = 0
		self.naoMove   = 0
		self.possBHVRS = ["demonstrate_rock.xar","demonstrate_paper.xar","demonstrate_scissors.xar","move_rpsBeginGame.xar","move_rock.xar","move_paper.xar",
"move_scissors.xar","letsPlay.xar","iwon1.xar","iwon2.xar","iwon2.xar","uwon1.xar","uwon2.xar","uwon3.xar","equal1.xar","equal2.xar", "startDemo.xar"]
		self.connectNao()
	#____________________________________________________________

	def connectNao(self):
#		print "CONNECT NAO"
		host = "192.168.0.80"
		port = 9559
		self.frame = ALProxy("ALFrameManager", host, port)
		self.motion = ALProxy("ALMotion", host, port)
	#____________________________________________________________
		
	def send_command(self, doBehavior, what):
#		print "SEND COMMAND (DOES THE MOVE)"	
		gesture_path = BASEPATH + doBehavior
		gesture_id = self.frame.newBehaviorFromFile(gesture_path, "")
		self.motion.stiffnessInterpolation("Body", 1.0, 1.0)
		self.frame.playBehavior(gesture_id)
		if(what != "demo"):
			self.frame.completeBehavior(gesture_id)
		self.after_effects("")
	#____________________________________________________________

	def after_effects(self, gesture):
		if gesture is "standup":
			self.position = "stand"
		if gesture is "sitdown":
			self.position = "sit"
	#____________________________________________________________

	def naoBehaviors(self, what):
		print "NAO BAHVIORS CHOOSING ..."+str(what)
		if(what is "demo"):
			doBehavior = self.possBHVRS[16]
			self.send_command(doBehavior, what)
			
			doBehavior = self.possBHVRS[0]
			self.send_command(doBehavior, what)
			doBehavior = self.possBHVRS[1]
			self.send_command(doBehavior, what)
			doBehavior = self.possBHVRS[2]
			self.send_command(doBehavior, what)
		elif(what is "play"):
			doBehavior = self.possBHVRS[7]
			self.send_command(doBehavior, what)
			doBehavior = self.possBHVRS[3]
			self.send_command(doBehavior, what)
		elif(what is "move"):
			self.naoMove = random.randint(0,2)	
			doBehavior   = self.possBHVRS[self.naoMove+4]
			self.send_command(doBehavior, what)            
		elif(what is "lost"):
			doBehavior = self.possBHVRS[random.randint(0,2)+8]
			self.send_command(doBehavior, what)            
		elif(what is "won"):
			doBehavior = self.possBHVRS[random.randint(0,2)+11]
			self.send_command(doBehavior, what)            
		elif(what is "equal"):
			doBehavior = self.possBHVRS[random.randint(0,1)+14]
			self.send_command(doBehavior, what) 
#____________________________________________________________


