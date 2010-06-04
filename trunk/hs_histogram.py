import sys
import cv

class hs_histogram:
    def __init__(self, src):
        self.src = src
        self.hist = None
    def getHist(self):
        # Convert to HSV
        hsv = cv.CreateImage(cv.GetSize(self.src), 8, 3)
        cv.CvtColor(self.src, hsv, cv.CV_BGR2HSV)

        # Extract the H and S planes
        h_plane = cv.CreateMat(self.src.height, self.src.width, cv.CV_8UC1)
        s_plane = cv.CreateMat(self.src.height, self.src.width, cv.CV_8UC1)
        cv.Split(hsv, h_plane, s_plane, None, None)
        planes = [h_plane, s_plane]

        h_bins = 9
        s_bins = 16
        hist_size = [h_bins, s_bins]
        # hue varies from 0 (~0 deg red) to 180 (~360 deg red again */
        h_ranges = [0, 180]
        # saturation varies from 0 (black-gray-white) to
        # 255 (pure spectrum color)
        s_ranges = [0, 255]
        ranges = [h_ranges, s_ranges]
        scale = 10
        self.hist = cv.CreateHist([h_bins, s_bins], cv.CV_HIST_ARRAY, ranges, 1)
        cv.CalcHist([cv.GetImage(i) for i in planes], self.hist)
        (_, max_value, _, _) = cv.GetMinMaxHistValue(self.hist)

        hist_img = cv.CreateImage((h_bins*scale, s_bins*scale), 8, 3)

        for h in range(h_bins):
            for s in range(s_bins):
                bin_val = cv.QueryHistValue_2D(self.hist, h, s)
                intensity = cv.Round(bin_val * 255 / max_value)
                cv.Rectangle(hist_img,
                             (h*scale, s*scale),
                             ((h+1)*scale - 1, (s+1)*scale - 1),
                             cv.RGB(intensity, intensity, intensity), 
                             cv.CV_FILLED)
        return hist_img
        
    def getMinMax(self):
        (min,max,minIdx,maxIdx) = cv.GetMinMaxHistValue(self.hist)
        return maxIdx
