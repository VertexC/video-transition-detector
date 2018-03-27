import cv2
import numpy as np
from matplotlib import pyplot as plt


def to_sti_column(cap):
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
    sti_column = np.zeros((height, frame_count, 3), dtype=np.uint8)
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        middle_column = frame[:, width // 2, :]
        sti_column[:, i, :] = middle_column
        i += 1

    return sti_column

def bgr_to_gr(image_bgr):
    # convert type from uint8 to float
    image_bgr_copy = np.array(image_bgr, dtype=np.float32) / 256
    # create new image with only 2 channels (G, R)
    image_gr = np.zeros((image_bgr.shape[0], image_bgr.shape[1], 2), dtype=np.float32)
    # + 0.00000001 to avoid divided by zero
    image_gr[:, :, 0] = image_bgr_copy[:, :, 1] / (
                image_bgr_copy[:, :, 0] + image_bgr_copy[:, :, 1] + image_bgr_copy[:, :, 2] + 0.0000001)
    image_gr[:, :, 1] = image_bgr_copy[:, :, 2] / (
                image_bgr_copy[:, :, 2] + image_bgr_copy[:, :, 1] + image_bgr_copy[:, :, 2] + 0.0000001)
    return image_gr

def cal_hist_gr(image_gr, bin_size):
    """
    Calculate histogram of a **GR** channel image
    :param image_gr: image (np.array) with two color channel (green, red) value from 0 to 1
    :param bin_size: the bin size of histogram
    :return: hist np.2darray with shape (bin_size, bin_size)
    """
    return cv2.calcHist([image_gr], [0, 1], None, [bin_size, bin_size], [0, 1, 0, 1])

def normalize(hist):
    """
    Normalize the histogram such that the sum of the matrix equal to 1
    :param hist: histogram np.array
    :return: normalized
    """
    total = np.sum(hist)
    return np.zeros(hist.shape) if total == 0 else hist / total
