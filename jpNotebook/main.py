import cv2
import numpy as np
import math
# from src.utils import *
from matplotlib import pyplot as plt
from .utils import *

def intersection_method(video_path, mode, threshold=0.5):
    """

    :param video_path:
    :param threshold:
    :param mode:
    :return:
    """
    # convert each column/row to STI

    video_path = '../media/video_3_down_wipe.mp4'
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError('File {} not found'.format(video_path))

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # convert each column/row to STI
    if mode == 'row':
        stis = np.ndarray((height, width, frame_count, 3), dtype=np.uint8)
        for i in range(height):
            stis[i] = to_sti(cap, 'row', i)
    elif mode == 'column':
        stis = np.ndarray((width, height, frame_count, 3), dtype=np.uint8)
        for i in range(width):
            stis[i] = to_sti(cap, 'column', i)

    # BGR -> GR
    stis_rg = np.ndarray(list(stis.shape[:-1]) + [2], dtype=np.float32)
    for i in range(width):
        stis_rg[i] = bgr_to_rg(stis[i])

    # implements histogram intersection on GR color channels
    threshold = 0.8
    wipe_positions, wipe_frames = intersection(stis_rg, threshold)

    number = wipe_positions.shape[0]
    print("The number of transition detected: {}".format(number))
    if number < width // 2:
        print("Warning: too few valid positions are detected. The result may be inaccurate.")
        print("Try to increase the threshold")
    if number > width * 2:
        print("Warning: too many valid positions are detected. The result may be inaccurate.")
        print("Try to decrease the threshold")

    #
    if mode == 'column':
        width_or_height = width
    elif mode == 'row':
        width_or_height = height

    start, end, sqr_error = linear_regression_column(wipe_positions, wipe_frames, width_or_height)

    return number, start, end, sqr_error
