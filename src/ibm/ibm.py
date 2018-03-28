import cv2


import numpy as np
import math

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt


def value_divide(a, n):
    """
    :param a: input value in range[0, m]
    :param n: quantize a by divide [0, m] to n bins
    :return: quantized value of a
    """
    return int(a * (n - 1)) / n


def rgb_to_chroma(rgb_m):
    """
    :param rgb_m: image with RGB channel
    :return: image with r,g channel, where {r,g} = {R,G}/{R+G+B} and r,g in [0,1]
    """
    row = col = channel = 0
    try:
        row, col, channel = rgb_m.shape
    except:
        print("Error: Invalid m type. Expect (m,n,channel).")
        exit(1)
    if channel != 3:
        print("Error: Invalid channel number. Expect 3.")
        exit(1)
    chroma_m = np.zeros((row, col, 2), dtype=np.double)
    for i in range(row):
        for j in range(col):
            if sum(rgb_m[i][j][:]) == 0:
                chroma_m[i][j] = [0, 0]
            else:
                chroma_m[i][j][0] = rgb_m[i][j][0] / sum(rgb_m[i][j][:])
                chroma_m[i][j][1] = rgb_m[i][j][1] / sum(rgb_m[i][j][:])
    return chroma_m


def to_histogram(col_chroma, n):
    """
    :param col_chroma: shape(n,2), all element in col_chroma should be in [0,1]
    :param n: number of bins
    :return: histogram of col_chroma as n*n
    """
    row = channel = 0
    try:
        row, channel = col_chroma.shape
    except:
        print("Error: Invalid col_chroma type. Expect (row, channel).")
        exit(1)
    if channel != 2:
        print("Error: Invalid channel number. Expect 2")
        exit(1)
    histogram = np.zeros((n, n))
    for i in range(row):
        r = int(col_chroma[i][0] * (n - 1))
        g = int(col_chroma[i][1] * (n - 1))
        histogram[r][g] += 1
    total = np.sum(histogram)
    return np.true_divide(histogram, total)


def IBM(video_path, to_chroma=True):
    """
    :param video_path: file path of video
    :param to_chroma: flag, whether use rgb_to_chroma rather than raw image data
    :return:
    """
    FRAME_WIDTH = 32
    FRAME_HEIGHT = 32

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height = FRAME_HEIGHT
    width = FRAME_WIDTH
    print(frame_count)
    # initialize

    # bin of histogram
    n = int(1 + math.log(height, 2))

    a_m = np.zeros((n ** 2, n ** 2), dtype=np.double)
    for i in range(n ** 2):
        for j in range(n ** 2):
            xi = i // n
            yi = i % n
            xj = j // n
            yj = j % n
            a_m[i, j] = 1 - math.sqrt((xi / n - xj / n) ** 2 + (yi / n - yj / n) ** 2) / math.sqrt(2)

    # apply IBM
    sti = np.zeros((width, frame_count - 1), dtype=np.double)
    histogram_vector_table = []
    f = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.release()
            break
        frame_resize = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_CUBIC)
        frame_chroma = rgb_to_chroma(frame_resize)
        for col in range(width):
            histogram_column = to_histogram(frame_chroma[:, col, :], n)
            v = histogram_column.reshape(histogram_column.size)
            if f == 0:
                histogram_vector_table.append(v)
            else:
                v_last = histogram_vector_table[col]
                z = v - v_last
                d_2 = (np.asmatrix(z) * np.asmatrix(a_m) * np.asmatrix(z).T)[0, 0]
                # update table
                histogram_vector_table[col] = v[:]
                sti[col, f - 1] = d_2
        # print(histogram_vector_table)
        print("frame:%d" % f)
        f += 1

    # # test: print sti as a image
    # sti_img = np.zeros((width, frame_count - 1, 1), dtype=np.uint8)
    # for i in range(width):
    #     for f in range(frame_count - 1):
    #         sti_img[i, f] = sti[i, f] * 255
    # cv2.imshow('STI', sti_img)
    # k = cv2.waitKey(0)
    # cv2.destroyAllWindows()

    X = []
    Y = []
    for f in range(frame_count - 1):
        for i in range(width):
            if sti[i, f] > 0.1:
                X.append(f)
                Y.append(i)
    X_one = np.vstack([np.asarray(X), np.ones(len(X))]).T
    k, c = np.linalg.lstsq(X_one, Y, rcond=-1)[0]

    # test: print linear regression result on sti
    print(k, c)
    plt.figure(1)
    print(X)
    print(Y)
    X = np.array(X)
    Y = np.array(Y)
    plt.plot(X, Y, 'bo')
    plt.plot(X, k * X + c, 'r--')
    plt.show()


# apply histogram
if __name__ == '__main__':
    # X = [1, 2, 3]
    # Y = [4, 5, 6]
    # plt.figure(1)
    # k = 1
    # c = 2
    # plt.plot(X, Y, 'bo')
    # plt.plot(X, X, 'r--')
    # plt.show()

    video_path = './media/left_wipe.avi'
    IBM(video_path)
