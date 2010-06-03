#!/usr/bin/python
"""
This program is demonstration for face and object detection using haar-like features.
The program finds faces in a camera image or video stream and displays a red box around them.

Original C implementation by:  ?
Python implementation by: Roman Stanchak, James Bowman
"""
import sys
import cv
from optparse import OptionParser

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

class getHistValues:
    def __init__(self,img,roi=None):
        if roi:
            cv.SetImageROI( img._hue, roi );
            cv.SetImageROI( img._val, roi );
            cv.SetImageROI( img._sat, roi );

        cv.CalcHist([img._hue, img._sat, img._val], img._histHSV, 0 );
        (_, _, _, maxBin) = cv.GetMinMaxHistValue( img._histHSV );

        # raise offset and multiply bins to best HSV values
        (h, s, v) = [x + 1 for x in maxBin];
        (hv, sv, vv) = ( 180/img._histBins, 255/25, 255/25);

        values = (h*hv, s*sv, v*vv);

        if roi:
            cv.ResetImageROI( img._hue );
            cv.ResetImageROI( img._sat );
            cv.ResetImageROI( img._val );

        return values;


class detect_and_draw:
    def __init__(self,img,cascade):
            
        rect = (0,0,1,1)

        # allocate temporary images
        gray = cv.CreateImage((img.width,img.height), 8, 1)
        small_img = cv.CreateImage((cv.Round(img.width / image_scale),
                       cv.Round (img.height / image_scale)), 8, 1)

        # convert color input image to grayscale
        cv.CvtColor(img, gray, cv.CV_BGR2GRAY)

        # scale input image for faster processing
        cv.Resize(gray, small_img, cv.CV_INTER_LINEAR)

        cv.EqualizeHist(small_img, small_img)

        if(cascade):
            t = cv.GetTickCount()
            faces = cv.HaarDetectObjects(small_img, cascade, cv.CreateMemStorage(0),
                                         haar_scale, min_neighbors, haar_flags, min_size)
            t = cv.GetTickCount() - t
            print "detection time = %gms" % (t/(cv.GetTickFrequency()*1000.))
            if faces:
                for ((x, y, w, h), n) in faces:
                    
                    faceX = int(x * image_scale)
                    faceY = int(y * image_scale)
                    faceW = w * image_scale
                    faceH = h * image_scale
                    
                    reScale = 0.2
                    horScl = int(faceW * reScale)
                    verScl = int(faceH * reScale)
                    
                    rect = (faceX+horScl,faceY+verScl,faceW-(horScl*2),faceH-(verScl*2))
                    # the input to cv.HaarDetectObjects was resized, so scale the 
                    # bounding box of each face and convert it to two CvPoints
                    pt1 = (int(x * image_scale), int(y * image_scale))
                    pt2 = (int((x + w) * image_scale), int((y + h) * image_scale))
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

        faceHSV = cv.CreateImage(cv.GetSize(face), 8, 3)
        cv.CvtColor(face, faceHSV, cv.CV_BGR2HSV);

        print getHistValues(faceHSV)

if __name__ == '__main__':

    parser = OptionParser(usage = "usage: %prog [options] [filename|camera_index]")
#    parser.add_option("-c", "--cascade", action="store", dest="cascade", type="str", help="Haar cascade file, default %default", default = "C:\OpenCV2.1\data\haarcascades/haarcascade_frontalface_alt.xml")
    parser.add_option("-c", "--cascade", action="store", dest="cascade", type="str", help="Haar cascade file, default %default", default = "..\haarcascades\haarcascade_frontalface_alt.xml")
    (options, args) = parser.parse_args()

    print options.cascade
    
    cascade = cv.Load(options.cascade)
    
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

    if capture:
        frame_copy = None
        while True:
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
            
            detect_and_draw(frame_copy, cascade)

            if cv.WaitKey(10) >= 0:
                break
    else:
        image = cv.LoadImage(input_name, 1)
        detect_and_draw(image, cascade)
        cv.WaitKey(0)

    cv.DestroyWindow("result")
    cv.DestroyWindow("resultFace")


