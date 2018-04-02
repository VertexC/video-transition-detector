import cv2
import numpy as np

import matplotlib
matplotlib.use("TkAgg")

def pixel_copy_sti(video_path, use_row=False):
    """
    Extract STI of middle column from a video
    :param cap: cv2.VideoCapture object
    :param: bool flag
    :return: STI image
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError('File {} not found'.format(video_path))

    FRAME_HEIGHT = 32
    FRAME_WIDTH = 32
    # get necessary information
    height = FRAME_HEIGHT
    width = FRAME_WIDTH
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    try:
        sti_column = np.zeros((height, frame_count, 3), dtype=np.uint8)
        sti_row = np.zeros((width, frame_count, 3), dtype=np.uint8)
    except ValueError:
        print("Invalid Video File.")
        exit(1)

    i = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_sieze = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_CUBIC)
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        middle_column = frame_sieze[:, width // 2, :]
        middle_row = frame_sieze[height // 2, :, :]
        sti_column[:, i, :] = middle_column
        sti_row[:, i, :] = middle_row
        i += 1

    if use_row:
        return sti_row
    else:
        return sti_column
