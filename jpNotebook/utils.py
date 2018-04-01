import cv2
import numpy as np
from matplotlib import pyplot as plt
import math


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

def intersection(sti_columns_rg, threshold):
    width, height, frame_count = sti_columns_rg.shape[:3]
    bin_size = 1 + math.floor(math.log(height, 2))

    H = np.ndarray((width, frame_count, bin_size, bin_size))

    # compute hist for one column from STI
    for col in range(width):
        for f in range(frame_count):
            H[col, f] = normalize(cal_hist_rg(sti_columns_rg[col, :, f:f + 1, :], bin_size))

    # create np.1darray for histogram intersection
    I = np.ndarray((width, frame_count - 1))

    # contruct the column * frame graph
    wipe_cols = []
    wipe_frames = []

    # collect all col, frame of wipe transitions
    for col in range(width):
        #     print("in column: {}".format(col))
        for f in range(I.shape[1]):
            # I[i] intersects histogram of frames at time i+1 and i
            I[col, f] = np.sum(np.minimum(H[col, f + 1], H[col, f]))
            # TODO: add cooldown time to avoid several continuous frames are all regraded as wipe
            if I[col, f] < threshold:
                wipe_cols.append(col)
                wipe_frames.append(f)
    #             print("Wipe at: {}".format(f + 1))

    wipe_cols = np.array(wipe_cols).reshape(-1, 1)
    wipe_frames = np.array(wipe_frames).reshape(-1, 1)

    return wipe_cols, wipe_frames

def linear_regression_column(wipe_cols, wipe_frames, width):
    from sklearn import linear_model

    regr = linear_model.LinearRegression()
    regr.fit(wipe_cols, wipe_frames)

    cols_test = np.array([0, width - 1]).reshape(-1, 1)
    frames_pred = regr.predict(cols_test)

    plt.scatter(wipe_cols, wipe_frames, color='black', linewidths=1)
    plt.plot(cols_test, frames_pred, color='blue', linewidth=3)

    # plt.xticks(())
    # plt.yticks(())

    plt.show()
    print("start/end frame: {}".format(np.round(sorted(frames_pred.flatten()))))
