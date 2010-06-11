import sys
import cv
from hs_histogram import *

if __name__ == '__main__':

    cv.NamedWindow("input", 1)
    cv.NamedWindow("histogram", 1)
    cv.NamedWindow("face", 1)
    cv.NamedWindow("skinProb", 1)
    cv.NamedWindow("skinProbThreshH", 1)
    cv.NamedWindow("skinProbThreshHEroDel", 1)
    cv.NamedWindow("skinProbColor", 1)

    capture = cv.CreateCameraCapture(int(0))
    
    frameCount = 0
    totalTime  = 0

    imageResize = 0.5

    
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

            ###########face detection##########
            min_neighbors = 2
            flags = cv.CV_HAAR_DO_CANNY_PRUNING
            min_size = (cv.Round(frameSmallGray.height/5),cv.Round(frameSmallGray.width/5))
            haar_scale = 1.2
            cascade = cv.Load("haarcascades\haarcascade_frontalface_alt.xml")
            
            faces = cv.HaarDetectObjects(frameSmallGray, cascade, cv.CreateMemStorage(0),
                                         haar_scale, min_neighbors, flags, min_size)

            rect = None
            
            for ((x, y, w, h), n) in faces:
                reScale = 0.3
                horScl = int(w * reScale)
                verScl = int(h * reScale)
                    
                #rect of middle of face
#                rect = (x+int(w*0.3),y,int(w*0.5),int(h*0.25))
                rect = (x+horScl,y+verScl,w-(horScl*2),h-(verScl*2))

                pt1 = (int(x), int(y))
                pt2 = (int((x + w)), int((y + h)))
                cv.Rectangle(frameSmall, pt1, pt2, cv.RGB(0, 255, 0), 3, 8, 0)
                ###################################

                ########calculate histogram########
                h_bins = 30
                s_bins = 32
                scale = 10
                histImg = cv.CreateImage((h_bins*scale, s_bins*scale), 8, 1)

                if rect:
                    #sets the Region of Interest in HSV image
                    cv.SetImageROI(frameSmallHSV, rect);
                    face = cv.CreateImage(cv.GetSize(frameSmallHSV),frameSmallHSV.depth,frameSmallHSV.nChannels);
                    cv.Copy(frameSmallHSV, face);
                    cv.ResetImageROI(frameSmallHSV);

                    cv.ShowImage("face", face) #inside of face

                    hist = hs_histogram(face)
                    myHist = hist.getHist()
                    (_, maxValue, _, _) = cv.GetMinMaxHistValue(myHist)

                    ########show histogram########
                    binSum=0
                    for h in range(h_bins):
                        for s in range(s_bins):
                            binSum += cv.QueryHistValue_2D(myHist, h, s)


                    for h in range(h_bins):
                        for s in range(s_bins):
                            binVal = cv.QueryHistValue_2D(myHist, h, s)
                            binProb = binVal/binSum
                            intensity = binProb * 255 / (maxValue/binSum)
                            cv.Rectangle(histImg,
                                         (h*scale, s*scale),
                                         ((h+1)*scale - 1, (s+1)*scale - 1),
                                         intensity, 
                                         cv.CV_FILLED)
                    cv.ShowImage("histogram", histImg) #histogram
                    ###################################
                    ###################################
                
                    skinProbImg = cv.CreateImage(cv.GetSize(frameSmallHSV),8,1);

                    hueImg = cv.CreateMat(frameSmallHSV.height, frameSmallHSV.width, cv.CV_8UC1)
                    satImg = cv.CreateMat(frameSmallHSV.height, frameSmallHSV.width, cv.CV_8UC1)
                    cv.Split(frameSmallHSV, hueImg, satImg, None, None)

    #                frameHueSat = cv.CreateImage((frameSmallHSV.height, frameSmallHSV.width),8,2)

    #                for x in range(0, frameSmallHSV.height):
    #                    for y in range(0, frameSmallHSV.width):
    #                        frameHueSat[x,y] = (hueImg[x,y],satImg[x,y])
                    
    #                cv.CalcBackProjection(frameHueSat,skinProbImg,myHist)

                    maxProbInt = 0
                    for x in range(0, frameSmallHSV.height):
                        for y in range(0, frameSmallHSV.width):
                            hue = int(hueImg[x,y]/(180/h_bins))
                            if hue==h_bins:
                                hue =h_bins-1
                            sat = int(satImg[x,y]/(256/s_bins))

                            binVal = cv.QueryHistValue_2D(myHist, hue, sat)
                            binProb = binVal/binSum
                            probIntensity = int(binProb * 255 / (maxValue/binSum))
                            
                            skinProbImg[x,y] = probIntensity
                            
                            if probIntensity>maxProbInt:
                                maxProbInt=probIntensity
                    
#                    cv.EqualizeHist(skinProbImg, skinProbImg)
                    
                    ### intensity according to histogramprobability###
                    for x in range(0, skinProbImg.height):
                        for y in range(0, skinProbImg.width):
                            if maxProbInt>0:
                                skinProbImg[x,y] = int((skinProbImg[x,y]/maxProbInt)*255)
                            else:
                                skinProbImg[x,y] = 0

                    cv.ShowImage("skinProb", skinProbImg) #Original image

                    cv.Smooth(skinProbImg,skinProbImg)

                    ### thresholded histogramprobability intensity###
                    for x in range(0, skinProbImg.height):
                        for y in range(0, skinProbImg.width):
                            if skinProbImg[x,y]>80:
                                skinProbImg[x,y] = 255
                            else:
                                skinProbImg[x,y] = 0
                            
                    cv.ShowImage("skinProbThreshH", skinProbImg) #Original image

                    kernel15 = cv.CreateStructuringElementEx(15,15,0,0, cv.CV_SHAPE_RECT)
                    kernel10 = cv.CreateStructuringElementEx(10,10,0,0, cv.CV_SHAPE_RECT)
                    kernel8 = cv.CreateStructuringElementEx(8,8,0,0, cv.CV_SHAPE_RECT)
                    kernel6 = cv.CreateStructuringElementEx(6,6,0,0, cv.CV_SHAPE_RECT)
                    kernel4 = cv.CreateStructuringElementEx(4,4,0,0, cv.CV_SHAPE_RECT)

                    cv.Dilate(skinProbImg, skinProbImg, kernel6, 1)
                    cv.Erode(skinProbImg, skinProbImg, kernel10, 1)

                    cv.ShowImage("skinProbThreshHEroDel", skinProbImg) #Original image
                    
                    skinProbColor = cv.CreateImage((skinProbImg.width,skinProbImg.height), 8, 3)
                    cv.CvtColor(skinProbImg, skinProbColor, cv.CV_GRAY2BGR)
                    
                    contours = cv.CreateMemStorage()
                    seq = cv.FindContours(skinProbImg, contours, cv.CV_RETR_EXTERNAL);
                    #print len(seq)
                    #cv.DrawContours(skinProbColor,seq,(255,0,0),(0,255,0),0)
                    while seq:
                        seqRect = cv.MinAreaRect2(seq)

                        seq = seq.h_next()
#                        if seqRect[2]==0:
                        cv.Rectangle(skinProbColor, seqRect[0], seqRect[1], cv.RGB(0, 255, 0), 3, 8, 0)
                    
                    cv.ShowImage("skinProbColor", skinProbColor) #Original image

            cv.ShowImage("input", frameSmall) #Original image
            
            t = cv.GetTickCount() - t
            totalTime += t/(cv.GetTickFrequency()*1000.)
            if frameCount%10==0:
                print "after %i frames the average time = %gms" % (frameCount, totalTime/frameCount)
            
            if cv.WaitKey(10) >= 0:
                break
            
    