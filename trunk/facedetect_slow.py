#!/usr/bin/python
import sys
import cv
from optparse import OptionParser
from hs_histogram import *

# Parameters for haar detection
# From the API:
# The default parameters (scale_factor=2, min_neighbors=3, flags=0) are tuned 
# for accurate yet slow object detection. For a faster operation on real video 
# images the settings are: 
# scale_factor=1.2, min_neighbors=2, flags=CV_HAAR_DO_CANNY_PRUNING, 
# min_size=<minimum possible face size

min_size = (20, 20)
image_scale = 2
haar_scale = 1.2
min_neighbors = 2
haar_flags = 0

# variables to store average hue and saturation
H=S=0
hasColor = 0

class extractBG:
    def __init__(self,motionImg):
        self.motionImg = motionImg
    
    def getDiffWithBG(self,bgImg,currFrame):
    
        motionDiff1C=60 #1 channel
        motionDiff3C=60 #3 channels
        
        if bgImg.nChannels==3:
#            print "getting diff for colorImage"
            for x in range(0, self.motionImg.height):
                for y in range(0, self.motionImg.width):
                    diff1 = abs(bgImg[x,y][0]-currFrame[x,y][0])
                    diff2 = abs(bgImg[x,y][1]-currFrame[x,y][1])
                    diff3 = abs(bgImg[x,y][2]-currFrame[x,y][2])
                    if diff1+diff2+diff3>motionDiff3C:
                        self.motionImg[x,y]=(255,255,255)
                    else:
                        self.motionImg[x,y]=(0,0,0)
            
        else:
#            print "getting diff for grayImage"
            for x in range(0, self.motionImg.height):
                for y in range(0, self.motionImg.width):
                    if abs(bgImg[x,y]-currFrame[x,y])>motionDiff1C:
                        self.motionImg[x,y]=255
                    else:
                        self.motionImg[x,y]=0

        return self.motionImg
#        cv.ShowImage("motion", diffImg) #motion

class getBG:
    def __init__(self,bgTotal):
        self.bgTotal = bgTotal
#       totalFramesForBG = frame_copy_gray
#       totalFramesForBG = getBR(totalFramesForBG)
    def addToTotal(self,newImg):
        #Add new image to total
        cv.Add(self.bgTotal,newImg,self.bgTotal,None)
        return self.bgTotal
#        totalFramesForBG.addToTotal(frame_copy_gray)
        
    def getAverage(self,frameCnt):
        #return average image (background)
        for x in range(0, self.bgTotal.height):
            for y in range(0, self.bgTotal.width):
                print "pixle value: %f" % self.bgTotal[x,y]
                self.bgTotal[x,y] = self.bgTotal[x,y]/frameCnt
                #print "pixle value: %f" % self.bgTotal[x,y]
        return self.bgTotal
#       bgImg = cv.CreateImage(cv.GetSize(frame_copy_gray),8,1)
#       bgImg = totalFramesForBG.getAverage(30)

class getSkinColor:
    def __init__(self,img):

        global hasColor
        global H
        global S
        faceFound = 0
        
        cascade = cv.Load("haarcascades\haarcascade_frontalface_alt.xml")

        #initial rect that will be shown when no face is detected
        rect = (0,0,1,1)

        # allocate temporary images
        gray = cv.CreateImage((img.width,img.height), 8, 1)
        # convert color input image to grayscale
        cv.CvtColor(img, gray, cv.CV_BGR2GRAY)
        # scale input image for faster processing
        cv.EqualizeHist(gray, gray)
        
        faces = cv.HaarDetectObjects(gray, cascade, cv.CreateMemStorage(0),
                                     haar_scale, min_neighbors, haar_flags, min_size)
        if faces:
            faceFound=1
            for ((x, y, w, h), n) in faces:

                faceX = x
                faceY = y
                faceW = w
                faceH = h
                
                reScale = 0.2
                horScl = int(faceW * reScale)
                verScl = int(faceH * reScale)
                    
                rect = (faceX+horScl,faceY+verScl,faceW-(horScl*2),faceH-(verScl*2))

                pt1 = (int(x), int(y))
                pt2 = (int((x + w)), int((y + h)))
                cv.Rectangle(img, pt1, pt2, cv.RGB(255, 0, 0), 3, 8, 0)

        cv.ShowImage("result", img)
        
        #sets the Region of Interest
        cv.SetImageROI(img, rect);
        face = cv.CreateImage(cv.GetSize(img),img.depth,img.nChannels);
        #copy subimage */
        cv.Copy(img, face);
        #always reset the Region of Interest */
        cv.ResetImageROI(img);
        cv.ShowImage("resultFace", face)

        hist = hs_histogram(face)
        myHist = hist.getHist()
        cv.ShowImage("histImg", myHist) #histogram
        [histHue,histSat] = hist.getMinMax()
        
        #as initialized in hs_histogram.py
        h_bins = 9
        s_bins = 16

        hueFromHist = int(int((180/9)*histHue)+int((180/9)*histHue+1)/2)
        satFromHist = int(int((255/9)*histSat)+int((255/9)*histSat+1)/2)
        
        if not hasColor and faceFound:
            H = hueFromHist
            S = satFromHist
            hasColor=1

class detect_and_draw:
    def __init__(self,img):

        if H==0 and S ==0:
            getSkinColor(img)

        imgHSV = cv.CreateImage(cv.GetSize(img), 8, 3)
        cv.CvtColor(img, imgHSV, cv.CV_BGR2HSV);

        hueImg = cv.CreateMat(img.height, img.width, cv.CV_8UC1)
        satImg = cv.CreateMat(img.height, img.width, cv.CV_8UC1)
        valImg = cv.CreateMat(img.height, img.width, cv.CV_8UC1)
        cv.Split(imgHSV, hueImg, satImg, valImg, None)

        cv.ShowImage("hueImg", hueImg)

        hueTrshld = cv.CreateMat(hueImg.height, hueImg.width, cv.CV_8UC1)
        hueDiff = 30
        satDiff = 80
        for x in range(0, hueTrshld.height):
            for y in range(0, hueTrshld.width):
                hueTrshld[x,y] = 0
                if hueImg[x,y]>(H-hueDiff) and hueImg[x,y]>(1) and hueImg[x,y]<(H+hueDiff):
                    if satImg[x,y]>(S-satDiff) and satImg[x,y]<(S+satDiff):
                        hueTrshld[x,y] = 255

        hueTrshldErode = cv.CreateMat(hueImg.height, hueImg.width, cv.CV_8UC1)        
        hueTrshldDilate = cv.CreateMat(hueImg.height, hueImg.width, cv.CV_8UC1)        


        kernel10 = cv.CreateStructuringElementEx(10,10,0,0, cv.CV_SHAPE_RECT)
        kernel8 = cv.CreateStructuringElementEx(8,8,0,0, cv.CV_SHAPE_RECT)
        kernel6 = cv.CreateStructuringElementEx(6,6,0,0, cv.CV_SHAPE_RECT)
        kernel4 = cv.CreateStructuringElementEx(4,4,0,0, cv.CV_SHAPE_RECT)

        cv.Erode(hueTrshld, hueTrshldErode, kernel6, 1)
        cv.Dilate(hueTrshldErode, hueTrshldDilate, kernel10, 1)

        
        cv.ShowImage("hueTrshldOr", hueTrshld) #original
        cv.ShowImage("hueTrshldDi", hueTrshldDilate) #dilated
        cv.ShowImage("hueTrshldEr", hueTrshldErode)  #eroded



if __name__ == '__main__':

    parser = OptionParser(usage = "usage: %prog [options] [filename|camera_index]")
#    parser.add_option("-c", "--cascade", action="store", dest="cascade", type="str", help="Haar cascade file, default %default", default = "haarcascades\haarcascade_frontalface_alt.xml")
    (options, args) = parser.parse_args()

    if len(args) != 1:
        parser.print_help()
        sys.exit(1)

    input_name = args[0]
    if input_name.isdigit():
        capture = cv.CreateCameraCapture(int(input_name))
    else:
        capture = None

    cv.NamedWindow("result", cv.CV_WINDOW_AUTOSIZE)
    cv.NamedWindow("resultFace", cv.CV_WINDOW_AUTOSIZE)
    cv.NamedWindow("hue", cv.CV_WINDOW_AUTOSIZE)
    cv.NamedWindow("hueImg", cv.CV_WINDOW_AUTOSIZE)
    cv.NamedWindow("hueTrshldOr", cv.CV_WINDOW_AUTOSIZE)
    cv.NamedWindow("hueTrshldEr", cv.CV_WINDOW_AUTOSIZE)
    cv.NamedWindow("hueTrshldDi", cv.CV_WINDOW_AUTOSIZE)
    cv.NamedWindow("histImg", cv.CV_WINDOW_AUTOSIZE)
    cv.NamedWindow("BG", cv.CV_WINDOW_AUTOSIZE)
    cv.NamedWindow("motion", cv.CV_WINDOW_AUTOSIZE)
    cv.NamedWindow("motionInImg", cv.CV_WINDOW_AUTOSIZE)

    frameCount=0
    totalTime=0
    
    framesForBackGround = 30
    
    useColor=0
    
    if capture:
        frame_copy = None
        while True:
            frameCount+=1
            t = cv.GetTickCount() #start timer
            frame = cv.QueryFrame(capture)
            if not frame:
                cv.WaitKey(0)
                break
            if not frame_copy:
                frame_copy = cv.CreateImage((frame.width,frame.height),
                                            cv.IPL_DEPTH_8U, frame.nChannels)
            if frame.origin == cv.IPL_ORIGIN_TL:
                cv.Copy(frame, frame_copy)
            else:
                cv.Flip(frame, frame_copy, 0)
                
            # convert input image to grayscale
            frame_copy_gray = cv.CreateImage((frame_copy.width,frame_copy.height), 8, 1)
            cv.CvtColor(frame_copy, frame_copy_gray, cv.CV_BGR2GRAY)

            if useColor==1:
                #make frameToUse HVS
                frameToUse = cv.CreateImage((frame_copy.width,frame_copy.height), 8, 3)
                cv.CvtColor(frame_copy, frameToUse, cv.CV_BGR2HSV);
            else:
                #make frameToUse gray
                frameToUse = cv.CreateImage((frame_copy.width,frame_copy.height), 8, 1)
                frameToUse = frame_copy_gray
                
            #scale down frame that will be used for calculations
            frameToUseSmall = cv.CreateImage((cv.Round(frameToUse.width / image_scale),cv.Round(frameToUse.height / image_scale)), 8, frameToUse.nChannels)
            cv.Resize(frameToUse, frameToUseSmall, cv.CV_INTER_LINEAR)

            #make RGB copy of frame
            frameCopySmallRGB = cv.CreateImage((cv.Round(frameToUse.width / image_scale),cv.Round(frameToUse.height / image_scale)), 8, 3)
            cv.Resize(frame_copy, frameCopySmallRGB, cv.CV_INTER_LINEAR)
            frameCopySmall = cv.CreateImage((frameCopySmallRGB.width,frameCopySmallRGB.height), 8, 3)
            cv.CvtColor(frameCopySmallRGB, frameCopySmall, cv.CV_BGR2HSV);

            if frameCount<=framesForBackGround:
                #use frame for background extraction
                #camera starts with dark frames. To avoid this to be used for background, possibly skip them
                if frameCount==1:
                    backgroundImg = cv.CreateImage(cv.GetSize(frameToUseSmall),8,frameToUseSmall.nChannels)
                    backgroundImg = frameToUseSmall
                else:
                    if frameToUseSmall.nChannels ==1: #gray image
                        for x in range(0, backgroundImg.height):
                            for y in range(0, backgroundImg.width):
                                backgroundImg[x,y] = ((backgroundImg[x,y]*(frameCount-1))+frameToUseSmall[x,y])/frameCount
                    else:
                        for x in range(0, backgroundImg.height):
                            for y in range(0, backgroundImg.width):
                                c1 = int((backgroundImg[x,y][0]*(frameCount-1))+frameToUseSmall[x,y][0])/frameCount
                                c2 = int((backgroundImg[x,y][1]*(frameCount-1))+frameToUseSmall[x,y][1])/frameCount
                                c3 = int((backgroundImg[x,y][2]*(frameCount-1))+frameToUseSmall[x,y][2])/frameCount
                                backgroundImg[x,y] = (c1,c2,c3)
                    cv.ShowImage("BG", backgroundImg) #background
            else:
                #finished getting background. Ready to detect
                motion = cv.CreateImage(cv.GetSize(frameToUseSmall), 8, frameToUseSmall.nChannels )
                motion = extractBG(motion)
                motion = motion.getDiffWithBG(backgroundImg,frameToUseSmall)
                cv.ShowImage("motion", motion) #binary

                #get pixels from frame that are different from the background
                motionInImg = cv.CreateImage(cv.GetSize(frameToUseSmall), 8, 3)
                for x in range(0, motion.height):
                    for y in range(0, motion.width):
                        if motion[x,y]==255:
                            motionInImg[x,y] = frameCopySmall[x,y]
                        else:
                            motionInImg[x,y] = (0,0,0)
                cv.ShowImage("motionInImg", motionInImg) #intensity
                
                detect_and_draw(motionInImg)

            if cv.WaitKey(10) >= 0:
                break
            #stop timer and show elapsed time
            t = cv.GetTickCount() - t
            totalTime += t/(cv.GetTickFrequency()*1000.)
            if frameCount%10==0:
                print "after %i frames the average time = %gms" % (frameCount, totalTime/frameCount)
    else:
        image = cv.LoadImage(input_name, 1)
        detect_and_draw(image, cascade)
        cv.WaitKey(0)

    cv.DestroyWindow("result")
    cv.DestroyWindow("resultFace")
    cv.DestroyWindow("hue")
    cv.DestroyWindow("hueImg")
    cv.DestroyWindow("hueTrshldOr")
    cv.DestroyWindow("hueTrshldEr")
    cv.DestroyWindow("hueTrshldDi")
    cv.DestroyWindow("histImg")
    cv.DestroyWindow("BG")
    cv.DestroyWindow("motion")
#    cv.DestroyWindow("motionPartOfImg")
