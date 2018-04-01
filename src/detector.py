import cv2

import numpy as np
import math


from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")

from matplotlib import pyplot as plt


class Detector:
    video_path = ""
    to_chromatic = True
    to_col = True
    threshold = 0.5
    sti = np.zeros((1, 1))

    def __init__(self):
        pass

    def set_video(self, path):
        self.video_path = path

    def set_mode(self, to_chromatic=True, to_col=True):
        self.to_chromatic = to_chromatic
        self.to_col = to_col

    def set_threshold(self, threshold):
        self.threshold = threshold

    def detect(self):
        pass