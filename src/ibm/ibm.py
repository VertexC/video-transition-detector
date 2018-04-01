import cv2

import numpy as np
import math

import matplotlib

matplotlib.use("TkAgg")
from matplotlib import pyplot as plt


class IBM:
    video_path = ""
    to_chromatic = True
    to_col = True
    sti = np.zeros((1, 1))

    def __init__(self):
        pass

    def set_video(self, path):
        self.video_path = path

    def set_mode(self, to_chromatic=True, to_col=True):
        self.to_chromatic = to_chromatic
        self.to_col = to_col

    def rgb_to_chroma(self, rgb_m):
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
                total = sum(rgb_m[i][j][:])
                if total < 15:
                    chroma_m[i][j] = [0, 0]
                else:
                    # red
                    chroma_m[i][j][0] = rgb_m[i][j][2] / total
                    # green
                    chroma_m[i][j][1] = rgb_m[i][j][1] / total
        return chroma_m

    def to_histogram(slef, col_chroma, n):
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

    def ibm(self):
        """
        :param video_path: file path of video
        :param to_chroma: flag, whether use rgb_to_chroma rather than raw image data
        :return:
        """
        FRAME_WIDTH = 32
        FRAME_HEIGHT = 32

        cap = cv2.VideoCapture(self.video_path)
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

        # initialize sti
        if self.to_col:
            sti = np.zeros((width, frame_count - 1), dtype=np.double)
        else:
            sti = np.zeros((height, frame_count - 1), dtype=np.double)

        histogram_vector_table = []
        f = 0

        # sampling
        while True:
            ret, frame = cap.read()
            if not ret:
                cap.release()
                break
            frame_resize = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_CUBIC)
            frame_chroma = self.rgb_to_chroma(frame_resize)
            if self.to_col:
                for col in range(width):
                    histogram_column = self.to_histogram(frame_chroma[:, col, :], n)
                    v = histogram_column.reshape(histogram_column.size)
                    if f == 0:
                        histogram_vector_table.append(v)
                    else:
                        v_last = histogram_vector_table[col]
                        z = np.abs(v - v_last)
                        d_2 = (np.asmatrix(z) * np.asmatrix(a_m) * np.asmatrix(z).T)[0, 0]

                        # if d_2 > 0.2:
                        #     print(z)

                        # update table
                        histogram_vector_table[col] = v[:]
                        sti[col, f - 1] = d_2
            else:
                for row in range(height):
                    histogram_row = self.to_histogram(frame_chroma[row, :, :], n)
                    v = histogram_row.reshape(histogram_row.size)
                    if f == 0:
                        histogram_vector_table.append(v)
                    else:
                        v_last = histogram_vector_table[row]
                        z = np.abs(v - v_last)
                        d_2 = (np.asmatrix(z) * np.asmatrix(a_m) * np.asmatrix(z).T)[0, 0]

                        # if d_2 > 0.2:
                        #     print(z)

                        # update table
                        histogram_vector_table[row] = v[:]
                        sti[row, f - 1] = d_2

            # print(histogram_vector_table)
            print("frame:%d" % f)
            f += 1
            # set sti
            self.sti = sti

    def linear_regression(self):
        """
        apply test on result sti
        :param sti:
        :return: void
        """
        row, col = self.sti.shape

        # # test: print sti as a image
        # sti_img = np.zeros((row, col, 1), dtype=np.uint8)
        # for i in range(row):
        #     for f in range(col - 1):
        #         sti_img[i, f] = sti[i, f] * 255
        # cv2.imshow('STI', sti_img)
        # k = cv2.waitKey(0)
        # cv2.destroyAllWindows()

        X = []
        Y = []
        for f in range(col):
            for i in range(row):
                if self.sti[i, f] > 0.25:
                    if self.to_col:
                        print("at f:%d, col%d, sti:%f" % (f, i, self.sti[i, f]))
                    else:
                        print("at f:%d, row%d, sti:%f" % (f, i, self.sti[i, f]))
                    X.append(f)
                    Y.append(i)

        if len(X) < 3 or len(Y) < 3:
            return False
        else:
            X_one = np.vstack([np.asarray(X), np.ones(len(X))]).T
            args, residuals = np.linalg.lstsq(X_one, Y, rcond=-1)[:2]
            k, c = args
            print(residuals)
            print("wipe detected, direction:", end="")
            if self.to_col:
                print("vertical,", end="")
                if k < 0:
                    print("move-left")
                else:
                    print("move-right")
            else:
                print("horizontal,", end="")
                if k < 0:
                    print("move-up")
                else:
                    print("move-down")
            # test: print linear regression result on sti
            plt.figure(1)
            X = np.array(X)
            Y = np.array(Y)
            plt.plot(X, Y, 'bo')
            plt.plot(X, k * X + c, 'r--')
            plt.show()
            return True

    def detect(self):
        self.ibm()
        if self.linear_regression():
            return True
        else:
            return False

    def __str__(self):
        return "method: IBM, to_chromatic: " + str(self.to_chromatic) + ", to_col: " + str(self.to_col)


if __name__ == '__main__':
    # video_path = '../../media/left_wipe.avi'
    # video_path = '../../media/video_1_horizontal_wipe.mp4'
    # video_path = '../../media/video_2_horizontal_wipe.mp4'
    # video_path = '../../media/video_1_vertical_wipe.mp4'
    # video_path = '../../media/video_2_vertical_wipe.mp4'
    video_path = '../../media/video_3_down_wipe.mp4'
    # video_path = '../../media/video_4_left_wipe.mp4'
    # video_path = '../../media/video_4_up_wipe.mp4'


    to_chroma = True

    model = IBM()
    model.set_video(video_path)
    model.set_mode(to_chromatic=to_chroma, to_col=True)
    if not model.detect():
        model.set_mode(to_chromatic=to_chroma, to_col=False)
        model.detect()

