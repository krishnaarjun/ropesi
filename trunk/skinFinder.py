import sys
import cv
from PIL import *
from hs_histogram import *
import cProfile
from predictSign import *

class detectSkin:
    def __init__(self):
	#most predicted:
	self.predictions = {"rock":0, "paper":0, "scissors":0, "garb":0, "none":0}
	self.maximum     = ""
	self.maxnr       = 0
        self.goalImg     = cv.CreateImage((70,70), cv.IPL_DEPTH_8U, 1)
	self.predict     = predictSign(70, False, 0) # takes the size of the images as input
	#load the model of the classification
	self.problem_hand = self.predict.loadModel("knn", 1, "hands") # 1=>original images; 2=>PCA; 3=>Gabor Wavelets+original image; 4=>only Gabor Wavelets	
	self.problem_sign = self.predict.loadModel("knn", 1, "rock") # 1=>original images; 2=>PCA; 3=>Gabor Wavelets+original image; 4=>only Gabor Wavelets	

   #__________________________________________________________

    def reinitGame(self):
#	print "REINIT IN SKIN DETECT"	

	self.predictions = {"rock":0, "paper":0, "scissors":0, "garb":0, "none":0}
	self.maximum     = ""
	self.maxnr       = 0	
   #__________________________________________________________

    def findSkin(self): 	
#	print "finding skin"   
	#skin detector from now on 
        capture = cv.CreateCameraCapture(int(0))
        cascade = cv.Load("haarcascades/haarcascade_frontalface_alt.xml")
        
        frameCount  = 0
        totalTime   = 0
        frameCount  = 0
        totalTime   = 0
        imageResize = 0.5
        hasHist     = False
        showImgs    = False
        
        if capture:
            inputFrame = None
            while True:
                frameCount+=1
                
                if frameCount%50==0 or frameCount==10:
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
                bestCandidateRect = (0,0,0,0)
                
                skinProbImgOriginal = cv.CreateImage(cv.GetSize(frameSmallHSV),8,1);
                
                if hasHist == False:
                    ###########detect a face##########
                    inSideFace,face = self.findFace(frameSmallGray,cascade)
                    (bestFaceX,bestFaceY,bestFaceW,bestFaceH) = face

                    #show rectangle around detected face
                    pt1 = (int(bestFaceX), int(bestFaceY))
                    pt2 = (int((bestFaceX + bestFaceW)), int((bestFaceY + bestFaceH)))
                    cv.Rectangle(frameSmall, pt1, pt2, cv.RGB(0, 255, 0), 3, 8, 0)
                    ###################################
                    ########calculate histogram########
                    myHistMat = cv.CreateMat(hBins, sBins, 1)
                    if bestFaceW*bestFaceH>0: 
                        myHistMat,hasHist,maxHistVal = self.calcHistogram(frameSmallHSV,inSideFace,hBins,sBins)
                    ###################################
                else: # if hist already exists
                    ########calculate skin probability image########
                    skinProbImg,skinProbImgOriginal = self.getSkinProbImg(frameSmallHSV,inSideFace,hBins,sBins,face,myHistMat,maxHistVal,showImgs)
                    #make a copy of the original skin probability image
                    ###################################

                    #construct color image to add colored contour and rectangle
                    skinProbColor = cv.CreateImage((skinProbImg.width,skinProbImg.height), 8, 3)
                    cv.CvtColor(skinProbImg, skinProbColor, cv.CV_GRAY2BGR)

                    # find  best candidate contour for blobs
                    bestCandidateRect = self.findBestCandBlob(skinProbImg,face)
                    x = bestCandidateRect[0]
                    y = bestCandidateRect[1]
                    w = bestCandidateRect[2]
                    h = bestCandidateRect[3]                   

                    cv.Rectangle(skinProbColor, (x,y), (x+w,y+h), cv.RGB(0, 255, 0), 3, 8, 0)
                    if showImgs:
                        cv.ShowImage("skinProbColor", skinProbColor) #blobs with contours
                
                cv.ShowImage("input", frameSmall) #Original image
                
                hand70x70 = self.makeHandImageSquare(bestCandidateRect,frameSmall,frameCount,skinProbImgOriginal,showImgs)
                cv.ShowImage("handSquare", hand70x70) #hand

                t = cv.GetTickCount() - t
                totalTime += t/(cv.GetTickFrequency()*1000.)
                
		#save image for train
		#cv.SaveImage("train/aa345camera"+str(frameCount)+".jpg", self.goalImg)	

		#does the prediction on the given image of hand
		whatPred = self.predict.doPrediction(1, "knn", self.problem_hand, "hands", self.goalImg)
		if(whatPred == "hands"):
			whatPred = self.predict.doPrediction(1, "knn", self.problem_sign, "rock", self.goalImg)
		self.predictions[whatPred] += 1 
		for (key,values) in self.predictions.items():
			if(values>self.maxnr):
				self.maximum = key
				self.maxnr   = values

		if frameCount%10==0:
                    print "after %i frames the average time = %gms" % (frameCount, totalTime/frameCount)
	
                if cv.WaitKey(10) >= 0:
                    break	
    #________________________________________________
	
    def findFace(self,img,cascade):       
        bestFaceX    = 0
        bestFaceY    = 0
        bestFaceW    = 0
        bestFaceH    = 0

        min_neighbors = 2
        flags = cv.CV_HAAR_DO_CANNY_PRUNING
        min_size = (cv.Round(img.height/5),cv.Round(img.width/5))
        haar_scale = 1.2
        
        faces = cv.HaarDetectObjects(img, cascade, cv.CreateMemStorage(0),
                                     haar_scale, min_neighbors, flags, min_size)

        rect = None
        biggestFace = 0
        goodRects = [0,0,0,0,0,0,0,0,0,0]
        
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
    
        return rect,(bestFaceX,bestFaceY,bestFaceW,bestFaceH)
        
    def calcHistogram(self,frameSmallHSV,inSideFace,hBins,sBins):
        #sets the Region of Interest (face) in HSV image
        cv.SetImageROI(frameSmallHSV, inSideFace);
        face = cv.CreateImage(cv.GetSize(frameSmallHSV),frameSmallHSV.depth,frameSmallHSV.nChannels);
        cv.Copy(frameSmallHSV, face);
        cv.ResetImageROI(frameSmallHSV);
        #get size of face area
        faceArea = face.height*face.width

        hist = hs_histogram(face)
        myHist = hist.getHist(hBins, sBins)
        (_, maxValue, _, _) = cv.GetMinMaxHistValue(myHist)
        hasHist = True
        myHistMat = cv.CreateMat(hBins, sBins, 1)
        for h in range(hBins):
                for s in range(sBins):
                    binVal = cv.QueryHistValue_2D(myHist, h, s)
                    myHistMat[h,s] = binVal        
        return myHistMat,hasHist,maxValue

    def getSkinProbImg(self,frameSmallHSV,inSideFace,hBins,sBins,face,myHistMat,maxHistVal,showImgs):
        #
        (bestFaceX,bestFaceY,bestFaceW,bestFaceH) = face
        skinProbImg = cv.CreateImage(cv.GetSize(frameSmallHSV),8,1);
        hueImg      = cv.CreateMat(frameSmallHSV.height, frameSmallHSV.width, cv.CV_8UC1)
        satImg      = cv.CreateMat(frameSmallHSV.height, frameSmallHSV.width, cv.CV_8UC1)
        cv.Split(frameSmallHSV, hueImg, satImg, None, None)

        binSum = inSideFace[2]*inSideFace[3]
        maxProbInt = 0
        
        for x in range(0, frameSmallHSV.height):
            for y in range(0, frameSmallHSV.width):
                hue = int(hueImg[x,y]/(180/hBins))
                if hue==hBins:
                    hue =hBins-1
                sat = int(satImg[x,y]/(256/sBins))

                binVal           = myHistMat[hue,sat]
                binProb          = binVal/binSum
                probIntensity    = int(binProb * 255 / (maxHistVal/binSum))
                skinProbImg[x,y] = probIntensity
                
                if probIntensity>maxProbInt:
                    maxProbInt=probIntensity
                    
        if bestFaceW*bestFaceH>0:
        #delete probability pixels in face area
            for y in range(int(bestFaceY*0.8),int(bestFaceY*0.8+bestFaceH*1.2)):
                for x in range(bestFaceX,bestFaceX+bestFaceW):
                    skinProbImg[y,x]=0    
                    
        if showImgs:
            cv.ShowImage("skinProb", skinProbImg) #Original skin probability image

        skinProbImgOriginal = cv.CreateImage(cv.GetSize(frameSmallHSV),8,1);
        cv.Copy(skinProbImg,skinProbImgOriginal)

        skinProbImg = self.processSkinProbImg(skinProbImg,showImgs)

        return skinProbImg,skinProbImgOriginal

    def processSkinProbImg(self,skinProbImg,showImgs):
        #threshold
        cv.InRangeS(skinProbImg,100,255,skinProbImg)
        if showImgs:
            cv.ShowImage("skinProbThresholded1", skinProbImg) #Original skin probability image after thresholding
        #smooth
        cv.Smooth(skinProbImg,skinProbImg,cv.CV_BLUR_NO_SCALE)
        if showImgs:
            cv.ShowImage("skinProbSmoothed", skinProbImg) #Original skin probability image after thresholding
        #erode
        kernelEr = cv.CreateStructuringElementEx(4,4,0,0, cv.CV_SHAPE_RECT)
        cv.Erode(skinProbImg, skinProbImg, kernelEr, 1)
        if showImgs:
            cv.ShowImage("skinProbEroded", skinProbImg) #Original skin probability image after thresholding
        #dilate
        kernelDi = cv.CreateStructuringElementEx(20,15,0,0, cv.CV_SHAPE_ELLIPSE)
        cv.Dilate(skinProbImg, skinProbImg, kernelDi, 1)
        if showImgs:
            cv.ShowImage("skinProbDilated", skinProbImg) #Original skin probability image after thresholding
        #smooth
        cv.Smooth(skinProbImg,skinProbImg,cv.CV_BLUR_NO_SCALE)
        if showImgs:
            cv.ShowImage("skinProbSmoothed", skinProbImg) #Original image
        return skinProbImg
        
    def findBestCandBlob(self,skinProbImg,face):
        (bestFaceX,bestFaceY,bestFaceW,bestFaceH) = face
        seq = cv.FindContours(skinProbImg, cv.CreateMemStorage(), cv.CV_RETR_EXTERNAL);
        bestCandidateRectSize = 0
        bestCandidateRect     = (0,0,0,0)
        while seq:
            # find the biggest blob/handCandidate
#            cv.DrawContours(skinProbColor,seq,(255,0,0),(255,255,0),0,2)
            seqRect = cv.BoundingRect(seq)
            if seqRect[2]*seqRect[3]> bestCandidateRectSize:
                #update best rect
                bestCandidateRect = (seqRect[0],seqRect[1],seqRect[2],seqRect[3])
                bestCandidateRectSize = seqRect[2]*seqRect[3]
            seq = seq.h_next()

        rectCntrX = bestCandidateRect[0] + int(bestCandidateRect[2]/2)
        rectCntrY = bestCandidateRect[1] + int(bestCandidateRect[3]/2)
        if ((rectCntrX>bestFaceX and rectCntrX<bestFaceX+bestFaceW) and(rectCntrY>bestFaceY and rectCntrY<bestFaceY+bestFaceH)):
            #rect is in face area
            bestCandidateRect     = (0,0,0,0)
        else:
            x = bestCandidateRect[0]
            y = bestCandidateRect[1]
            w = bestCandidateRect[2]
            h = bestCandidateRect[3]
#            cv.Rectangle(skinProbColor, (x,y), (x+w,y+h), cv.RGB(0, 255, 0), 3, 8, 0)
        return bestCandidateRect
            
    def makeHandImageSquare(self,bestCandidateRect,frameSmall,frameCount,skinProbImgOriginal,showImgs):
        if bestCandidateRect != (0,0,0,0):
            cv.SetImageROI(skinProbImgOriginal, bestCandidateRect);
            cv.SetImageROI(frameSmall, bestCandidateRect);
            
            if frameCount%10==0:
                if showImgs:
                    cv.DestroyWindow("hand1")
                    cv.NamedWindow("hand1", 1)
            cv.InRangeS(skinProbImgOriginal,10,256,skinProbImgOriginal)
            kernelEr = cv.CreateStructuringElementEx(3,3,0,0, cv.CV_SHAPE_ELLIPSE)
            cv.Erode(skinProbImgOriginal, skinProbImgOriginal, kernelEr, 1)
            kernelDi = cv.CreateStructuringElementEx(11,7,0,0, cv.CV_SHAPE_ELLIPSE)
            cv.Dilate(skinProbImgOriginal, skinProbImgOriginal, kernelDi, 1)

            for x in range(0,bestCandidateRect[2]):
                for y in range(0,bestCandidateRect[3]):
                    if skinProbImgOriginal[y,x]==0:
                        frameSmall[y,x]=(0,0,0)

            handNoBGGray = cv.CreateImage((bestCandidateRect[2],bestCandidateRect[3]), 8, 1)
            cv.CvtColor(frameSmall, handNoBGGray, cv.CV_BGR2GRAY)
            cv.EqualizeHist(handNoBGGray, handNoBGGray)
            
            if showImgs:
                cv.ShowImage("hand1", handNoBGGray) #hand
            
            bestHandW  = bestCandidateRect[2]
            bestHandH  = bestCandidateRect[3]

            handHW     = max(bestHandH,bestHandW)
            
            handSquare = cv.CreateImage((handHW,handHW), 8, 1)

            xStart = int((handHW-bestHandW)/2)
            xEnd   = handHW - int((handHW-bestHandW)/2)-1
            yStart = int((handHW-bestHandH)/2)
            yEnd   = handHW - int((handHW-bestHandH)/2)-1
            
            for y in range(0,handHW):
                for x in range(0,handHW):
                    if y>=yStart and y<yEnd and x>=xStart and x<xEnd:
                        handSquare[y,x] = handNoBGGray[y-yStart,x-xStart]
                    else:
                        handSquare[y,x] = 0

            hand70x70 = cv.CreateImage((70,70), 8, 1)
            cv.Resize(handSquare,hand70x70,cv.CV_INTER_LINEAR)
	    self.goalImg = hand70x70	
            return hand70x70
                    
if __name__ == '__main__':
#    cProfile.run('detectSkin()')
    detectSkin()
    
