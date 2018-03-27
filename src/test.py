import cv2
import numpy as np
import matplotlib as mpl
import src.utils as util
import math

mpl.use('TkAgg')


def image_test():
    # image part
    print(cv2.__version__)
    img = cv2.imread("./a1p5.jpeg", 0)
    # imread FLAG
    # 1: cv2.IMREAD_COLOR : Loads a color image. Any transparency of image will be neglected. It is the default flag.
    # 0: cv2.IMREAD_GRAYSCALE : Loads image in grayscale mode
    # -1: cv2.IMREAD_UNCHANGED : Loads image as such including alpha channel
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', img)
    k = cv2.waitKey(0)
    if k == 27 or k == ord('q'):  # wait for ESC key to exit
        cv2.destroyAllWindows()
    elif k == ord('s'):  # wait for 's' key to save and exit
        cv2.imwrite('messigray.png', img)
    cv2.destroyAllWindows()


# video part
def video_test():
    cap = cv2.VideoCapture('./source/left_wipe.avi')

    while (cap.isOpened()):
        ret, frame = cap.read()
        # print(frame.size)
        # print(frame[0,:][:])
        if frame is None:
            # try to sampling
            cap.set()
            break
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cap.release()


# apply histogram
if __name__ == '__main__':
    # image_test()
    # video_test()
    cap = cv2.VideoCapture('./source/left_wipe.avi')
    # sti = util.pixel_copy_sti(cap, use_row = False)
    # cv2.imshow('STI', sti)
    # k = cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # implementation of histogram_sti
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # initialize
    n = int(1 + math.log(height, 2))  # bin of histogram


    # quantize 0 - 255 RGB value into 0 - n-1 value
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


    a_m = np.zeros((n ** 2, n ** 2), dtype=np.double)
    for i in range(n ** 2):
        for j in range(n ** 2):
            xi = i // n
            yi = i % n
            xj = j // n
            yj = j % n
            a_m[i, j] = 1 - math.sqrt((xi / n - xj / n) ** 2 + (yi / n - yj / n) ** 2) / math.sqrt(2)
    # print(a_m)

    sti = np.zeros((width, frame_count - 1), dtype=np.double)  # set double for test
    last_column = np.zeros((height, 1, 3), dtype=np.uint8)
    last_column_histogram = np.zeros((n, n, 2), dtype=np.double)
    histogram_vector_table = []
    f = 0
    # apply IBM directly
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_chroma = rgb_to_chroma(frame)
        for col in range(width):
            histogram_column = to_histogram(frame_chroma[:, col, :], n)
            v = histogram_column.reshape(histogram_column.size)
            if f == 0:
                histogram_vector_table.append(v)
            else:
                v_last = histogram_vector_table[col]
                z = v - v_last
                # print(z)
                d_2 = (np.asmatrix(z) * np.asmatrix(a_m) * np.asmatrix(z).T)[0, 0]
                # print(d_2)
                # update table
                histogram_vector_table[col] = v[:]
                sti[col, f - 1] = d_2
        # print(histogram_vector_table)
        print(f)
        f += 1
    print((sti))

    sti_img = np.zeros((width, frame_count - 1, 1), dtype=np.uint8)
    for i in range(width):
        for f in range(frame_count - 1):
            sti_img[i, f] = sti[i, f] * 255
    cv2.imshow('STI', sti_img)
    k = cv2.waitKey(0)
    cv2.destroyAllWindows()
