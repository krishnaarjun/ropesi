import os
import sys
import random
import time

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
#____________________________________________________________

class Gesture:
	def __init__(self, host, port):
		self.host         = host
		self.port         = port
		self.stiffness    = 1.0
		self.naoMove      = 0

		self.frame        = None
		self.voice        = "Heather22Enhanced"
		self.speechDevice = None
		self.motion       = None
		self.possBHVRS    = {"handUp":"hand_up.xar", "demoRock":"demonstrate_rock.xar", "demoPaper":"demonstrate_paper.xar", \
							"demoScissors":"demonstrate_scissors.xar", "handDown":"hand_down.xar", \
							"doMove":["move_rock.xar","move_paper.xar","move_scissors.xar"]}
		self.win          = ['hi hi hi I won', 'Ohhh yeah I am good', 'You lost nao rules', 'Yeah I am the rock paper scissors king', 'I won']
		self.loose        = ['I lost', 'Hei you won cheater', 'It is not fair I lost', 'I guess I need to learn how to play'] 
		self.draw         = ['It was a draw', 'Nobody won', 'It was a draw lets play again', 'Rematch?']
		self.connectNao()
	#initialize all nao devices____________________________________________________________
	def connectNao(self):
		#FRAME MANAGER FOR CALLING BEHAVIORS
		try:
			self.frame  = ALProxy("ALFrameManager", self.host, self.port)
		except Exception, e:
		    print "Error when creating frame manager device proxy:"+str(e)
		    exit(1)
	

		#MOTION DEVICE FOR MOVEMENTS
		try:
			self.motion = ALProxy("ALMotion", self.host, self.port)
		except Exception, e:
		    print "Error when creating motion device proxy:"+str(e)
		    exit(1)
		#MAKE NAO STIFF (OTHERWISE IT WON'T MOVE)
		self.motion.stiffnessInterpolation("Body",self.stiffness,1.0)


		#CONNECT TO A SPEECH PROXY
		try:
		    self.speechDevice = ALProxy("ALTextToSpeech", self.host, self.port)
		except Exception, e:
		    print "Error when creating speech device proxy:"+str(e)
		    exit(1)
		try:
			self.speechDevice.setVoice(self.voice)	
		except Exception, e:
		    print "Error when setting the voice and volume: "+str(e)

	#SAY A SENTENCE___________________________________________________________________________________
	def genSpeech(self, sentence):
		try:
			self.speechDevice.post.say(sentence)
		except Exception, e:
		    print "Error when saying a sentence: "+str(e)

	#____________________________________________________________		
	def send_command(self, doBehavior, what):
		gesture_path = BASEPATH + doBehavior
		gesture_id   = self.frame.newBehaviorFromFile(gesture_path, "")
		self.frame.playBehavior(gesture_id)
		self.frame.completeBehavior(gesture_id)

	#____________________________________________________________

	def naoBehaviors(self, what):
		#print "NAO BAHVIORS CHOOSING ..."+str(what)
		if(what is "demo"):
			#INITIALIZE POSITION OF THE HAND
			self.send_command(self.possBHVRS["handUp"], what)
			
			self.genSpeech("This is how I do rock")			
			self.send_command(self.possBHVRS["demoRock"], what)
			time.sleep(1)

			self.genSpeech("This is how I do paper")		
			self.send_command(self.possBHVRS["demoPaper"], what)
			time.sleep(1)

			self.genSpeech("This is how I do scissors")			
			self.send_command(self.possBHVRS["demoScissors"], what)					
			time.sleep(1)

			#MOVE HAND DOWN TO START PALYING
			self.send_command(self.possBHVRS["handDown"], what) 

		elif(what is "play"):			
			self.genSpeech("Let us play")
			time.sleep(1)

			#choose default one of the bahviors
			print "go count signs >>>"
			self.naoMove = random.randint(0,2)
			self.send_command(self.possBHVRS["doMove"][self.naoMove], what)            

		elif(what is "win"): # NAO WON
			randnr = randint(0,len(self.win)-1)	
			self.genSpeech(self.win[randnr])
			time.sleep(1)
		elif(what is "loose"): # NAO LOST
			randnr = randint(0,len(self.loose)-1)	
			self.genSpeech(self.loose[randnr])
			time.sleep(1)
		elif(what is "draw"): # DRAW
			randnr = randint(0,len(self.draw)-1)	
			self.genSpeech(self.draw[randnr])
			time.sleep(1)

	# RELEASE THE JOINTS SO IT WON'T COMPLAIN
	def releaseNao(self):
		self.motion.stiffnessInterpolation("Body",0.0,self.stiffness)

#____________________________________________________________

