import cv2
import numpy as np

import matplotlib
matplotlib.use("TkAgg")

def pixel_copy_sti(cap, use_row=False):
    """
    Extract STI of middle column from a video
    :param cap: cv2.VideoCapture object
    :param: bool flag
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
    sti_row = np.zeros((width, frame_count, 3), dtype=np.uint8)
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        middle_column = frame[:, width // 2, :]
        middle_row = frame[height // 2, :, :]
        sti_column[:, i, :] = middle_column
        sti_row[:, i, :] = middle_row
        i += 1

    if use_row:
        return sti_row
    else:
        return sti_column
