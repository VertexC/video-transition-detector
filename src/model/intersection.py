from src.model.detector import *
from src.model.detectresult import DetectResult
from sklearn import linear_model
from sklearn.metrics import mean_squared_error


class IntersectionDetector(Detector):
    ACCEPTED_MODE = ['column', 'row']
    mode = 'column'
    cap = None

    detect_result = None

    def set_mode(self, to_chromatic=True, to_col=True):
        self.to_chromatic = to_chromatic
        self.to_col = to_col
        self.mode = 'column' if to_col else 'row'

    def to_sti(self, column_or_row_num=0):
        """
        Extract STI of given column or row from a video
        :param column_or_row_num: specify which column or row should be extracted
        :return: STI image
        """
        if not self.cap.isOpened():
            print("The video is not opened!")
            return None

        if self.mode.lower() not in self.ACCEPTED_MODE:
            raise ValueError('Unknown mode: {}. Accepted List: {}'.format(self.mode, self.ACCEPTED_MODE))

        # get necessary information
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        height_or_width = 0
        if self.mode == 'column':
            height_or_width = height
        elif self.mode == 'row':
            height_or_width = width

        sti = np.ndarray((height_or_width, frame_count, 3), dtype=np.uint8)
        i = 0
        while True:
            ret, frame = self.cap.read()
            if not ret: break
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.mode == 'column':
                selected_column_or_row = frame[:, column_or_row_num, :]
            elif self.mode == 'row':
                selected_column_or_row = frame[column_or_row_num, :, :]
            else:
                # should not happen
                selected_column_or_row = 0
            sti[:, i, :] = selected_column_or_row
            i += 1

        return sti

    def bgr_to_rg(self, image_bgr):
        # convert type from uint8 to float
        image_bgr_copy = np.array(image_bgr, dtype=np.float32) / 256
        # create new image with only 2 channels (G, R)
        image_rg = np.ndarray(list(image_bgr.shape[:-1]) + [2], dtype=np.float32)
        # + 0.00000001 to avoid divided by zero
        bgr_sum = (image_bgr_copy[:, :, 0] + image_bgr_copy[:, :, 1] + image_bgr_copy[:, :, 2] + 0.0000001)
        # red
        image_rg[:, :, 0] = image_bgr_copy[:, :, 2]
        # green
        image_rg[:, :, 1] = image_bgr_copy[:, :, 1]
        return image_rg

    def cal_hist_rg(self, image_rg, bin_size):
        """
        Calculate histogram of a **GR** channel image
        :param image_rg: image (np.array) with two color channel (green, red) value from 0 to 1
        :param bin_size: the bin size of histogram
        :return: hist np.2darray with shape (bin_size, bin_size)
        """
        return cv2.calcHist([image_rg], [0, 1], None, [bin_size, bin_size], [0, 1, 0, 1])

    def normalize(self, hist):
        """
        Normalize the histogram such that the sum of the matrix equal to 1
        :param hist: histogram np.array
        :return: normalized
        """
        total = np.sum(hist)
        return np.zeros(hist.shape) if total == 0 else hist / total

    def intersection(self, stis_rg, threshold):
        """
        Use STI made of columns to generate all the transitions (position, time).
        Though is function is written for STI of columns, but it also works for rows
        :param stis_rg: sti with shape (width, height, frame_count, 2)
        :param threshold: 0 ~ 1, the larger, the more strict
        :return: wipe_cols, wipe_frames, can be drawn using scatter()
        """
        width, height, frame_count = stis_rg.shape[:3]
        bin_size = 1 + math.floor(math.log(height, 2))

        H = np.ndarray((width, frame_count, bin_size, bin_size))

        # compute hist for one column from STI
        for col in range(width):
            for f in range(frame_count):
                H[col, f] = self.normalize(self.cal_hist_rg(stis_rg[col, :, f:f + 1, :], bin_size))

        # create np.1darray for histogram intersection
        I = np.ndarray((width, frame_count - 1))

        # contruct the column * frame graph
        wipe_positions = []
        wipe_frames = []

        # collect all col, frame of wipe transitions
        for col in range(width):
            #     print("in column: {}".format(col))
            for f in range(I.shape[1]):
                # I[i] intersects histogram of frames at time i+1 and i
                I[col, f] = np.sum(np.minimum(H[col, f + 1], H[col, f]))
                if I[col, f] < threshold:
                    wipe_positions.append(col)
                    wipe_frames.append(f)
        #             print("Wipe at: {}".format(f + 1))

        wipe_positions = np.array(wipe_positions).reshape(-1, 1)
        wipe_frames = np.array(wipe_frames).reshape(-1, 1)

        return wipe_positions, wipe_frames

    def linear_regression_column(self, wipe_positions, wipe_frames, width_or_height):
        """
        Use linear regression to predict the start / end of the transition
        :param wipe_positions: the col or row that a wipe happens on
        :param wipe_frames: the frame that a wipe happens at
        :param width_or_height: width when sti is made of columns
        :return:
        """

        # check the validity of data
        if not wipe_positions.shape == wipe_frames.shape:
            raise ValueError("The shape of first two inputs must match")
        if wipe_positions.shape[0] == 0:
            raise ValueError("The number of data should be greater than zero")

        regr = linear_model.LinearRegression()
        regr.fit(wipe_positions, wipe_frames)
        positions_test = np.array([0, width_or_height - 1]).reshape(-1, 1)

        frames_pred = regr.predict(positions_test)

        plt.scatter(wipe_positions, wipe_frames, color='black', linewidths=1)
        plt.plot(positions_test, frames_pred, color='blue', linewidth=3)

        # plt.xticks(())
        # plt.yticks(())

        plt.show()
        start, end = np.round(sorted(frames_pred.flatten()))
        sqr_error = mean_squared_error(wipe_frames, regr.predict(wipe_positions))
        # print("start/end frame: {}/{}".format(start, end))
        # print("Mean Square Error: {}".format(sqr_error))

        direction = None

        if self.mode == 'column':
            direction = 'move left' if frames_pred[1] - frames_pred[0] < 0 else 'move right'
        elif self.mode == 'row':
            direction = 'move up' if frames_pred[1] - frames_pred[0] < 0 else 'move down'

        return direction, start, end, sqr_error

    def detect_one(self):
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise FileNotFoundError('File {} not found'.format(self.video_path))

        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # convert each column/row to STI
        try:
            if self.mode == 'row':
                stis = np.ndarray((height, width, frame_count, 3), dtype=np.uint8)
                for i in range(height):
                    stis[i] = self.to_sti(i)
            elif self.mode == 'column':
                stis = np.ndarray((width, height, frame_count, 3), dtype=np.uint8)
                for i in range(width):
                    stis[i] = self.to_sti(i)
        except ValueError:
            print("Invalid Video Type.")
            exit(1)

        # BGR -> GR
        stis_rg = np.ndarray(list(stis.shape[:-1]) + [2], dtype=np.float32)
        for i in range(width):
            stis_rg[i] = self.bgr_to_rg(stis[i])

        # implements histogram intersection on GR color channels
        threshold = 0.8
        wipe_positions, wipe_frames = self.intersection(stis_rg, threshold)

        number = wipe_positions.shape[0]
        # print("The number of transition detected: {}".format(number))

        message = ''

        if number < width // 2:
            message += "Warning: too few valid positions are detected. The result may be inaccurate.\n"
            message += "Try to increase the threshold\n"

        if number > width * 2:
            message += "Warning: too many valid positions are detected. The result may be inaccurate."
            message += "Try to decrease the threshold"

        #
        if self.mode == 'column':
            width_or_height = width
        elif self.mode == 'row':
            width_or_height = height

        direction, start, end, sqr_error = self.linear_regression_column(wipe_positions, wipe_frames, width_or_height)

        type = "horizontal" if self.mode == "col" else "vertical"
        return DetectResult(type, direction, abs(number - width_or_height), start, end, message)

    def show_frame(self, frame):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ret, some_frame = self.cap.read()
        plt.imshow(cv2.cvtColor(some_frame, cv2.COLOR_BGR2RGB))
        plt.show()

    def show_result(self):

        if not self.detect_result:
            print("No result detected")
            return
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.video_path)

        start = self.detect_result.start_frame
        end = self.detect_result.end_frame
        self.show_frame(start)
        self.show_frame((start + end) // 2)
        self.show_frame(end)

    def detect(self):
        best_result = None
        for mode in self.ACCEPTED_MODE:
            self.mode = mode
            try:
                detect_result = self.detect_one()
            except ValueError:
                continue
            if best_result == None:
                best_result = detect_result
            elif best_result.detected_error > detect_result.detected_error:
                best_result = detect_result
        self.detect_result = best_result
        return best_result
