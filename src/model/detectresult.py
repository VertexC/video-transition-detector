import numpy as np
class DetectResult:

    def __init__(self, transition_type: str, transition_direction: str, detected_error: int,
                 start_frame_no: int, end_frame_no: int,
                 start_frame_image: np.ndarray, middle_frame_image: np.ndarray, end_frame_image: np.ndarray,
                 message: str=None):
        """

        :param transition_type: 'column' | 'row'
        :param transition_direction: 'left', 'right', 'up', 'down'
        :param detected_error: a measure of error, the larger the worse
        :param start_frame_no: integer
        :param end_frame_no: integer
        :param message: Any warning message
        """
        self.transition_type = transition_type
        self.transition_direction = transition_direction
        self.detected_error = detected_error
        self.start_frame_no = int(start_frame_no)
        self.end_frame_no = int(end_frame_no)
        self.start_frame_image = start_frame_image
        self.middle_frame_image = middle_frame_image
        self.end_frame_image = end_frame_image
        self.message = message

    def __str__(self):
        return "Detect result: \nType: {};\nDirection: {};\nStart: {};\nEnd: {};\nMessage: {}"\
            .format(self.transition_type, self.transition_direction, self.start_frame_no, self.end_frame_no, self.message)
