import cv2
import numpy as np
from matplotlib import pyplot as plt


def wtf():
    print("hello")


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