class DetectResult:

    def __init__(self, transition_type: str, transition_direction: str,
                 start_frame: int, end_frame: int,
                 message: str=None):
        self.transition_type = transition_type
        self.transition_direction = transition_direction
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.message = message

    def __str__(self):
        return "Detect result: \nType: {};\nDirection: {};\nStart: {};\nEnd: {};\nMessage: {}"\
            .format(self.transition_type, self.transition_direction, self.start_frame, self.end_frame, self.message)