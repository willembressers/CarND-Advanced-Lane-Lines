# python packages
import logging

# 3rd party packages
import cv2
import numpy as np

class Transform:

    def __init__(self, width, height):
        """
        Initializes the Transform

        :param width: The width of the image.
        :param height: The height of the image.

        :return: None
        """
        logging.debug(f'Initializing the Transform')
        self.width = width
        self.height = height

        # Source points
        src = np.float32([[546, 450], [732, 450], [width, height-10], [0, height-10]])

        # Destination points
        dst = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

        # calculate the perspective transform matrix
        self.matrix = cv2.getPerspectiveTransform(src, dst)

         # calculate the INVERSE perspective transform matrix
        self.inverse_matrix = cv2.getPerspectiveTransform(dst, src)

    def perspective(self, image):
        """
        Apply the perspective transform

        :param image: The image to work on

        :return: image
        """

        # Warp the image using OpenCV warpPerspective()
        return cv2.warpPerspective(image, self.matrix, (self.width, self.height))