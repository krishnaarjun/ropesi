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
global H
global S
H=S=0
hasColor =0

class extractBG:
    def __init__(self,motionImg):
        self.motionImg = motionImg
    
    def getBG(self,bgImg,currFrame):
    
        motionDiff=30
        
        grayFrame = cv.CreateImage((currFrame.width,currFrame.height), 8, 1)
        cv.CvtColor(currFrame, grayFrame, cv.CV_BGR2GRAY)
        
#        diffImg = cv.CreateImage((currFrame.width,currFrame.height), 8, 1)
        
        for x in range(0, self.motionImg.height):
            for y in range(0, self.motionImg.width):
                if abs(bgImg[x,y]-grayFrame[x,y])>motionDiff:
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
    def __init__(self,img,hasColor):

        cascade = cv.Load("haarcascades\haarcascade_frontalface_alt.xml")

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
        cv.ShowImage("histImg", myHist) #original
        [histHue,histSat] = hist.getMinMax()
        print "hueBin: %d  >>> satBin:%d" % (histHue,histSat)
        
        #as initialized in hs_histogram.py
        h_bins = 9
        s_bins = 16

        hueFromHist = int(int((180/9)*histHue)+int((180/9)*histHue+1)/2)
        satFromHist = int(int((255/9)*histSat)+int((255/9)*histSat+1)/2)
        
        if not hasColor:
            global H
            global S         
            H = hueFromHist
            S = satFromHist
            hasColor=1

class detect_and_draw:
    def __init__(self,img):

        small_img = cv.CreateImage((cv.Round(img.width / image_scale),cv.Round(img.height / image_scale)), 8, 3)
        cv.Resize(img, small_img, cv.CV_INTER_LINEAR)

        if H!=0 and S !=0:
            getSkinColor(small_img, hasColor)

        imgHSV = cv.CreateImage(cv.GetSize(small_img), 8, 3)
        cv.CvtColor(small_img, imgHSV, cv.CV_BGR2HSV);

        hueImg = cv.CreateMat(small_img.height, small_img.width, cv.CV_8UC1)
        satImg = cv.CreateMat(small_img.height, small_img.width, cv.CV_8UC1)
        valImg = cv.CreateMat(small_img.height, small_img.width, cv.CV_8UC1)
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

    cv.NamedWindow("result", 1)
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
    
    framesForBackGround = 10
    
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
                
            frame_copy_gray = cv.CreateImage((frame_copy.width,frame_copy.height), 8, 1)
            # convert color input image to grayscale
            cv.CvtColor(frame_copy, frame_copy_gray, cv.CV_BGR2GRAY)
            
            if frameCount<framesForBackGround+1:
                #use frame for background extraction
                if frameCount==1:
                    backgroundImg = cv.CreateImage(cv.GetSize(frame_copy_gray),8,1)
                    backgroundImg = frame_copy_gray
                    print "New totalFrame initialized"
                else:
                    for x in range(0, backgroundImg.height):
                        for y in range(0, backgroundImg.width):
                            backgroundImg[x,y] = ((backgroundImg[x,y]*(frameCount-1))+frame_copy_gray[x,y])/frameCount
                    cv.ShowImage("BG", backgroundImg) #background
                    print "totalFrame updated"
            else:
                #finished getting background. Ready to detect
                motion = cv.CreateImage((frame_copy.width,frame_copy.height), 8, 1)
                motion = extractBG(motion)
                motion = motion.getBG(backgroundImg,frame_copy)
                cv.ShowImage("motion", motion) #motion
                
                motionInImg = cv.CreateImage((frame_copy.width,frame_copy.height), 8, 3)
                
                for x in range(0, motionInImg.height):
                    for y in range(0, motionInImg.width):
                        if motion[x,y]==255:
                            motionInImg[x,y] = frame_copy[x,y]
                        else:
                            motionInImg[x,y] = (0,0,0)
                        
                
                cv.ShowImage("motionInImg", motionInImg) #motion
                
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
