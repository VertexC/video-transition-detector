from src.model.detector import *

class IbmDetector(Detector):

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
                if self.to_chromatic:
                    total = sum(rgb_m[i][j][:])
                    if total < 15:
                        chroma_m[i][j] = [0, 0]
                    else:
                        # red
                        chroma_m[i][j][0] = rgb_m[i][j][2] / total
                        # green
                        chroma_m[i][j][1] = rgb_m[i][j][1] / total
                else:
                    # red
                    chroma_m[i][j][0] = rgb_m[i][j][2]
                    # green
                    chroma_m[i][j][1] = rgb_m[i][j][1]

        return chroma_m

    def to_histogram(self, col_chroma, n):
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
        if self.to_chromatic:
            color_range = 1
        else:
            color_range = 255
        for i in range(row):
            r = int(col_chroma[i][0] * (n - 1) / color_range)
            g = int(col_chroma[i][1] * (n - 1) / color_range)
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
        if self.to_chromatic:
            threshold = 0.25
        else:
            threshold = 0.5
        for f in range(col):
            for i in range(row):
                if self.sti[i, f] > threshold:
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
            print("wipe detected, direction:[", end="")
            if self.to_col:
                print("vertical,", end="")
                if k < 0:
                    print("move-left", end="")
                else:
                    print("move-right", end="")
            else:
                print("horizontal,", end="")
                if k < 0:
                    print("move-up", end="")
                else:
                    print("move-down", end="")
            print("], ", end="")
            print("from frame: %d to %d" % (X[0], X[-1]))
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
