class DetectResult:

    def __init__(self, transition_type: str, transition_direction: str, detected_error: int,
                 start_frame: int, end_frame: int,
                 message: str=None):
        """

        :param transition_type: 'column' | 'row'
        :param transition_direction: 'left', 'right', 'up', 'down'
        :param detected_error: a measure of error, the larger the worse
        :param start_frame: integer
        :param end_frame: integer
        :param message: Any warning message
        """
        self.transition_type = transition_type
        self.transition_direction = transition_direction
        self.detected_error = detected_error
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.message = message

    def __str__(self):
        return "Detect result: \nType: {};\nDirection: {};\nStart: {};\nEnd: {};\nMessage: {}"\
            .format(self.transition_type, self.transition_direction, self.start_frame, self.end_frame, self.message)
