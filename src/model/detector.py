import cv2
from src.model.detectresult import DetectResult
import numpy as np
import math

from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")
from matplotlib import pyplot as plt


class Detector:
    cap = None
    video_path = ""
    to_chromatic = True
    to_col = True
    threshold = 0.5
    sti = np.zeros((1, 1))

    def __init__(self):
        pass

    def set_video(self, path):
        self.video_path = path

    def set_mode(self, to_chromatic=True):
        self.to_chromatic = to_chromatic

    def set_threshold(self, threshold):
        self.threshold = threshold

    def get_frame(self, frame_no):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame_image = self.cap.read()
        if not ret:
            return None
        return cv2.cvtColor(frame_image, cv2.COLOR_BGR2RGB)

    def show_frame(self, image_rgb):
        plt.imshow(image_rgb)
        plt.show()
    def detect(self):
        pass