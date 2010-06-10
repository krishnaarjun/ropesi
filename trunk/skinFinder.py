import sys
import cv
from hs_histogram import *

if __name__ == '__main__':

    cv.NamedWindow("inputImg", 1)

    capture = cv.CreateCameraCapture(int(0))
    
    frameCount = 0
    totalTime  = 0
    if capture:
        inputFrame = None
        while True:
            frameCount+=1
            t = cv.GetTickCount() #start timer
            frame = cv.QueryFrame(capture)
            if not frame:
                cv.WaitKey(0)
                break
            if not inputFrame:
                inputFrame = cv.CreateImage((frame.width,frame.height),
                                            cv.IPL_DEPTH_8U, frame.nChannels)
            if frame.origin == cv.IPL_ORIGIN_TL:
                cv.Copy(frame, inputFrame)
            else:
                cv.Flip(frame, inputFrame, 0)
                
            cv.ShowImage("inputImg", inputFrame) #Original image
            
            t = cv.GetTickCount() - t
            totalTime += t/(cv.GetTickFrequency()*1000.)
            if frameCount%10==0:
                print "after %i frames the average time = %gms" % (frameCount, totalTime/frameCount)
#                totalTime=0
            
            if cv.WaitKey(10) >= 0:
                break
            
    