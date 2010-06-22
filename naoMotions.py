
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

class Gesture:
    def __init__(self,demo,playNow):
#    def __init__(self):
        self.demo      = demo
        self.playNow   = playNow
        self.lastvalue = [0,0,0]
        self.position  = "stand"
        self.counter   = 0
        self.possBHVRS = ["demonstrate_rock.xar","demonstrate_paper.xar","demonstrate_scissors.xar","move_rpsBeginGame.xar","move_rock.xar","move_paper.xar","move_scissors.xar","letsPlay.xar"]
        self.connect_nao()
        self.test()

    def connect_nao(self):
        host = "192.168.0.80"
        port = 9559
        self.frame = ALProxy("ALFrameManager", host, port)
        self.motion = ALProxy("ALMotion", host, port)
        

    def gesture_check(self, values):
        pass

    def send_command(self, doBehavior):
        gesture_path = BASEPATH + doBehavior
        print ">>>>>>>>>>",gesture_path
        gesture_id = self.frame.newBehaviorFromFile(gesture_path, "")
        self.motion.stiffnessInterpolation("Body", 1.0, 1.0)
        self.frame.playBehavior(gesture_id)
        if self.playNow:
            self.frame.completeBehavior(gesture_id)
        self.after_effects("")

    def after_effects(self, gesture):
        if gesture is "standup":
            self.position = "stand"
        if gesture is "sitdown":
            self.position = "sit"
            print "we are sitting"
        print self.position
        
    def test(self):
        if self.demo:
            doBehavior = self.possBHVRS[0]
            self.send_command(doBehavior)
            doBehavior = self.possBHVRS[1]
            self.send_command(doBehavior)
            doBehavior = self.possBHVRS[2]
            self.send_command(doBehavior)
        if self.playNow:
            doBehavior = self.possBHVRS[7]
            self.send_command(doBehavior)
            doBehavior = self.possBHVRS[3]
            self.send_command(doBehavior)
            doBehavior = self.possBHVRS[random.randint(0,2)+4]
            self.send_command(doBehavior)            
    
Gesture(False,True)


