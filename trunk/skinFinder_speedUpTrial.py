import sys
import cv
from hs_histogram import *

if __name__ == '__main__':

    cv.NamedWindow("input", 1)
    cv.NamedWindow("skinProb", 1)
    cv.NamedWindow("skinProbThreshH", 1)
    cv.NamedWindow("skinProbThreshHEroDel", 1)
    cv.NamedWindow("skinProbColor", 1)
    cv.NamedWindow("skinProbSmoothed", 1)

    capture = cv.CreateCameraCapture(int(0))
    
    frameCount = 0
    totalTime  = 0

    imageResize = 0.5
    
    hasHist = False
    
    bestFaceX    = 0
    bestFaceY    = 0
    bestFaceW    = 0
    bestFaceH    = 0

    if capture:
        inputFrame = None
        while True:
            frameCount+=1
            
            if frameCount%50==0:
                hasHist = False

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
                
            #### resize and convert images ####
            # resized color input image
            frameSmall     = cv.CreateImage((cv.Round(inputFrame.width*imageResize),cv.Round(inputFrame.height*imageResize)), 8, inputFrame.nChannels)
            cv.Resize(inputFrame, frameSmall, cv.CV_INTER_LINEAR)
            # convert resized color input image to grayscale
            frameSmallGray = cv.CreateImage((frameSmall.width,frameSmall.height), 8, 1)
            cv.CvtColor(frameSmall, frameSmallGray, cv.CV_BGR2GRAY)
            cv.EqualizeHist(frameSmallGray, frameSmallGray)
            # convert resized color input image to grayscale
            frameSmallHSV  = cv.CreateImage((frameSmall.width,frameSmall.height), 8, 3)
            cv.CvtColor(frameSmall, frameSmallHSV, cv.CV_BGR2HSV)
            ###################################

            hBins = 30
            sBins = 32

            if hasHist == False:
                ###########face detection##########
                min_neighbors = 2
                flags = cv.CV_HAAR_DO_CANNY_PRUNING
                min_size = (cv.Round(frameSmallGray.height/5),cv.Round(frameSmallGray.width/5))
                haar_scale = 1.2
                cascade = cv.Load("haarcascades\haarcascade_frontalface_alt.xml")
                
                faces = cv.HaarDetectObjects(frameSmallGray, cascade, cv.CreateMemStorage(0),
                                             haar_scale, min_neighbors, flags, min_size)

                rect = None
                biggestFace = 0
                for ((x, y, w, h), n) in faces:
                    if w*h>biggestFace:
                        biggestFace=w*h
                        bestFaceX    = x
                        bestFaceY    = y
                        bestFaceW    = w
                        bestFaceH    = h
                
                reScale = 0.3
                horScl = int(bestFaceW * reScale)
                verScl = int(bestFaceH * reScale)
                    
                #rect of middle of face
                rect = (bestFaceX+horScl,bestFaceY+verScl,bestFaceW-(horScl*2),bestFaceH-(verScl*2)) # middle part of face

                pt1 = (int(bestFaceX), int(bestFaceY))
                pt2 = (int((bestFaceX + bestFaceW)), int((bestFaceY + bestFaceH)))
                cv.Rectangle(frameSmall, pt1, pt2, cv.RGB(0, 255, 0), 3, 8, 0)
                ###################################
                ########calculate histogram########
                if biggestFace>0: 
                    #sets the Region of Interest (face) in HSV image
                    cv.SetImageROI(frameSmallHSV, rect);
                    face = cv.CreateImage(cv.GetSize(frameSmallHSV),frameSmallHSV.depth,frameSmallHSV.nChannels);
                    cv.Copy(frameSmallHSV, face);
                    cv.ResetImageROI(frameSmallHSV);
                    #get size of face area
                    faceArea = face.height*face.width

                    hist = hs_histogram(face)
                    myHist = hist.getHist(hBins, sBins)
                    (_, maxValue, _, _) = cv.GetMinMaxHistValue(myHist)
                    print "face detected >>> Histogram constructed"
                    hasHist = True
                ###################################

            else: # if hist already calculated
                skinProbImg = cv.CreateImage(cv.GetSize(frameSmallHSV),8,1);
                hueImg      = cv.CreateMat(frameSmallHSV.height, frameSmallHSV.width, cv.CV_8UC1)
                satImg      = cv.CreateMat(frameSmallHSV.height, frameSmallHSV.width, cv.CV_8UC1)
                cv.Split(frameSmallHSV, hueImg, satImg, None, None)

                binSum = face.height*face.width
                maxProbInt = 0
                for x in range(0, frameSmallHSV.height):
                    for y in range(0, frameSmallHSV.width):
                        hue = int(hueImg[x,y]/(180/hBins))
                        if hue==hBins:
                            hue =hBins-1
                        sat = int(satImg[x,y]/(256/sBins))

                        binVal           = cv.QueryHistValue_2D(myHist, hue, sat)
                        binProb          = binVal/binSum
                        probIntensity    = int(binProb * 255 / (maxValue/binSum))
                        skinProbImg[x,y] = probIntensity
                        
                        if probIntensity>maxProbInt:
                            maxProbInt=probIntensity
                cv.ShowImage("skinProb", skinProbImg) #Original skin probability image

                #threshold
                cv.InRangeS(skinProbImg,100,255,skinProbImg)
                cv.ShowImage("skinProbThresholded1", skinProbImg) #Original skin probability image after thresholding
                #smooth
                cv.Smooth(skinProbImg,skinProbImg,cv.CV_BLUR_NO_SCALE)
                cv.ShowImage("skinProbSmoothed", skinProbImg) #Original skin probability image after thresholding
                #erode
                kernelEr = cv.CreateStructuringElementEx(4,4,0,0, cv.CV_SHAPE_RECT)
                cv.Erode(skinProbImg, skinProbImg, kernelEr, 1)
                cv.ShowImage("skinProbEroded", skinProbImg) #Original skin probability image after thresholding
                #dilate
                kernelDi = cv.CreateStructuringElementEx(14,14,0,0, cv.CV_SHAPE_ELLIPSE)
                cv.Dilate(skinProbImg, skinProbImg, kernelDi, 1)
                cv.ShowImage("skinProbDilated", skinProbImg) #Original skin probability image after thresholding
                #smooth
                cv.Smooth(skinProbImg,skinProbImg,cv.CV_BLUR_NO_SCALE)
                cv.ShowImage("skinProbSmoothed", skinProbImg) #Original image

                #construct color image to add colored contour and rectangle
                skinProbColor = cv.CreateImage((skinProbImg.width,skinProbImg.height), 8, 3)
                cv.CvtColor(skinProbImg, skinProbColor, cv.CV_GRAY2BGR)
                # find contours for blobs
                seq = cv.FindContours(skinProbImg, cv.CreateMemStorage(), cv.CV_RETR_EXTERNAL);
                while seq:
                    cv.DrawContours(skinProbColor,seq,(255,0,0),(255,255,0),0,2)
                    seqRect = cv.BoundingRect(seq)
                    x = seqRect[0]
                    y = seqRect[1]
                    w = seqRect[2]
                    h = seqRect[3]
                    
                    goodRectCntr = 0
                    goodRects=[10]
                    if w>=20 and h>=20:
                        rectCntrX = x + int(w/2)
                        rectCntrY = y + int(h/2)
                        if not((rectCntrX>bestFaceX and rectCntrX<bestFaceX+bestFaceW) and(rectCntrY>bestFaceY and rectCntrY<bestFaceY+bestFaceH)):
                            #rect is not in face area
                            cv.Rectangle(skinProbColor, (x,y), (x+w,y+h), cv.RGB(0, 255, 0), 3, 8, 0)
                            goodRects[goodRectCntr] = (x,y,h,w)
                            goodRectCntr+=1
                    seq = seq.h_next()
                cv.ShowImage("skinProbColor", skinProbColor) #blobs with contours
                print goodRectCntr," usefull rectangels found"

            cv.ShowImage("input", frameSmall) #Original image
            
            t = cv.GetTickCount() - t
            totalTime += t/(cv.GetTickFrequency()*1000.)
            if frameCount%10==0:
                print "after %i frames the average time = %gms" % (frameCount, totalTime/frameCount)
            if cv.WaitKey(10) >= 0:
                break
        