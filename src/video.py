# python packages
import logging

# 3rd party packages
import cv2

class Video():

    def __init__(self, path):
        """
        Initializes the video

        :return: None
        """
        logging.debug(f'Initializing the video')

        # load the video file
        self.capture = cv2.VideoCapture(path)

        # get the frame width
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))

        # get the frame height
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # get the frames per second
        self.fps = int(self.capture.get(cv2.CAP_PROP_FPS))

        # get the number of frames
        self.n_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))

    def __del__(self):
        """
        destroys the video object properly

        :return: None
        """
        self.capture.release()

    def has_frame(self):
        """
        Check if the video has a next frame

        :return: boolean
        """
        return self.capture.isOpened()

    def get_frame(self):
        """
        Get the next frame

        :return: boolean, frame, id
        """
        ret, frame = self.capture.read()
        id = int(self.capture.get(cv2.CAP_PROP_POS_FRAMES))
        return ret, frame, id
