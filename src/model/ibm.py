from src.model.detector import *


class IbmDetector(Detector):
    detect_result = None
    threshold = 0.25


    def show_result(self):
        if not self.detect_result:
            print("No result detected")
            return
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.video_path)

        self.show_frame(self.detect_result.start_frame_image)
        self.show_frame(self.detect_result.middle_frame_image)
        self.show_frame(self.detect_result.end_frame_image)

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
        if not cap.isOpened():
            raise FileNotFoundError('File {} not found'.format(self.video_path))
        self.cap = cap

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        height = FRAME_HEIGHT
        width = FRAME_WIDTH
        # print(frame_count)

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
        try:
            if self.to_col:
                sti = np.zeros((width, frame_count - 1), dtype=np.double)
            else:
                sti = np.zeros((height, frame_count - 1), dtype=np.double)
        except ValueError:
            print("Invalid Video Type")
            exit(1)

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
            # print("frame:%d" % f)
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
            self.threshold = 0.25
        else:
            self.threshold = 0.6
        for f in range(col):
            for i in range(row):
                if self.sti[i, f] > self.threshold:
                    # if self.to_col:
                    #     print("at f:%d, col%d, sti:%f" % (f, i, self.sti[i, f]))
                    # else:
                    #     print("at f:%d, row%d, sti:%f" % (f, i, self.sti[i, f]))
                    X.append(f)
                    Y.append(i)

        if len(X) < 3 or len(Y) < 3:
            return False
        else:
            X_one = np.vstack([np.asarray(X), np.ones(len(X))]).T
            args, residuals = np.linalg.lstsq(X_one, Y, rcond=-1)[:2]
            k, c = args

            # set result message
            if self.to_col:
                type = "horizontal"
                if k < 0:
                    direction = "move left"
                else:
                    direction = "move right"
            else:
                type = "vertical"
                if k < 0:
                    direction = "move up"
                else:
                    direction = "move down"
            error = residuals[0]
            start_frame_no = X[0]
            end_frame_no = X[-1]

            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise FileNotFoundError('File {} not found'.format(self.video_path))
            self.cap = cap

            start_frame_image = self.get_frame(start_frame_no)
            middle_frame_image = self.get_frame((start_frame_no + end_frame_no) // 2)
            end_frame_image = self.get_frame(end_frame_no)

            self.detect_result = DetectResult(type, direction, error, start_frame_no, end_frame_no,
                                              start_frame_image, middle_frame_image, end_frame_image)

            # test: print linear regression result on sti
            # plt.figure(1)
            # X = np.array(X)
            # Y = np.array(Y)
            # plt.plot(X, Y, 'bo')
            # plt.plot(X, k * X + c, 'r--')
            # plt.show()

            return True

    def detect(self):
        self.ibm()
        if self.linear_regression():
            return self.detect_result
        else:
            self.to_col = not self.to_col
            self.ibm()
            if self.linear_regression():
                return self.detect_result
            else:
                # set result message
                self.result = DetectResult(None, None, None, None, None, message="Wipe not detected.")

    def __str__(self):
        return "method: IBM, to_chromatic: " + str(self.to_chromatic) + ", to_col: " + str(self.to_col)


if __name__ == '__main__':
    from src.model.intersection import IntersectionDetector


    def test_intersection_method():
        # video_path = 'media/left_wipe.avi'
        # video_path = 'media/video_1_horizontal_wipe.mp4'
        # video_path = 'media/video_2_horizontal_wipe.mp4'
        # video_path = 'media/video_1_vertical_wipe.mp4'
        # video_path = 'media/video_2_vertical_wipe.mp4'
        # video_path = 'media/video_3_down_wipe.mp4'
        video_path = '../../media/video_4_left_wipe.mp4'
        # video_path = '../../media/video_4_up_wipe.mp4'

        to_chroma = False

        model = IbmDetector()
        model.set_video(video_path)
        model.set_mode(to_chromatic=to_chroma)
        model.set_threshold(0.5)
        detect_result = model.detect()
        if detect_result == None:
            print("No transition detected")
        else:
            print(detect_result)
            model.show_result()


    if __name__ == '__main__':
        test_intersection_method()
