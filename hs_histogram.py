import sys
import cv

class hs_histogram:
    def __init__(self, src):
        self.src = src
        self.hist = None
    def getHist(self, hBins, sBins):
        # Extract the H and S planes
        h_plane = cv.CreateMat(self.src.height, self.src.width, cv.CV_8UC1)
        s_plane = cv.CreateMat(self.src.height, self.src.width, cv.CV_8UC1)
        cv.Split(self.src, h_plane, s_plane, None, None)
        planes = [h_plane, s_plane]

        hist_size = [hBins, sBins]
        # hue varies from 0 (~0 deg red) to 180 (~360 deg red again */
        h_ranges = [0, 180]
        # saturation varies from 0 (black-gray-white) to 255 (pure spectrum color)
        s_ranges = [0, 255]
        ranges = [h_ranges, s_ranges]

        self.hist = cv.CreateHist([hBins, sBins], cv.CV_HIST_ARRAY, ranges, 1)
        cv.CalcHist([cv.GetImage(i) for i in planes], self.hist)

        return self.hist
        
    def getMinMax(self):
        (min,max,minIdx,maxIdx) = cv.GetMinMaxHistValue(self.hist)
        return maxIdx
