import cv2
import numpy as np
from matplotlib import pyplot as plt


def to_sti_column(cap, column_num=0):
    """
    Extract STI of middle column from a video
    :param cap: cv2.VideoCapture object
    :return: STI image
    """
    if not cap.isOpened():
        print("The video is not opened!")
        return None

    # get necessary information
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    sti_column = np.ndarray((height, frame_count, 3), dtype=np.uint8)
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        selected_column = frame[:, column_num, :]
        sti_column[:, i, :] = selected_column
        i += 1

    return sti_column

def bgr_to_rg(image_bgr):
    # convert type from uint8 to float
    image_bgr_copy = np.array(image_bgr, dtype=np.float32) / 256
    # create new image with only 2 channels (G, R)
    image_rg = np.ndarray(list(image_bgr.shape[:-1]) + [2], dtype=np.float32)
    # + 0.00000001 to avoid divided by zero
    bgr_sum = (image_bgr_copy[:, :, 0] + image_bgr_copy[:, :, 1] + image_bgr_copy[:, :, 2] + 0.0000001)
    # red
    image_rg[:, :, 0] = image_bgr_copy[:, :, 2] / bgr_sum
    # green
    image_rg[:, :, 1] = image_bgr_copy[:, :, 1] / bgr_sum
    return image_rg

def cal_hist_rg(image_rg, bin_size):
    """
    Calculate histogram of a **GR** channel image
    :param image_rg: image (np.array) with two color channel (green, red) value from 0 to 1
    :param bin_size: the bin size of histogram
    :return: hist np.2darray with shape (bin_size, bin_size)
    """
    return cv2.calcHist([image_rg], [0, 1], None, [bin_size, bin_size], [0, 1, 0, 1])

def normalize(hist):
    """
    Normalize the histogram such that the sum of the matrix equal to 1
    :param hist: histogram np.array
    :return: normalized
    """
    total = np.sum(hist)
    return np.zeros(hist.shape) if total == 0 else hist / total
