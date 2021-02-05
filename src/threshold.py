# python packages
import logging

# 3rd party packages
import cv2
import numpy as np

class Threshold:

    def __init__(self, color_threshold=(170, 255), gradient_threshold=(20, 100)):
        """
        Initializes the threshold

        :param color_threshold: The color threshold (min, max).
        :param gradient_threshold: The gradient threshold (min, max).

        :return: None
        """
        logging.debug(f'Initializing the threshold')
        self.color_threshold = color_threshold
        self.gradient_threshold = gradient_threshold

    def color_and_gradient(self, image):
        """
        Apply an color and gradient threshold on the image
        
        :param image: The image to work on
        
        :return: binary image
        """
        # Convert to HLS color space and separate the V channel
        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        l_channel = hls[:,:,1]
        s_channel = hls[:,:,2]

        # Sobel x
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
        abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
        
        # Threshold x gradient
        gradient_binary = np.zeros_like(scaled_sobel)
        gradient_binary[(scaled_sobel >= self.gradient_threshold[0]) & (scaled_sobel <= self.gradient_threshold[1])] = 1
        
        # Threshold color channel
        color_binary = np.zeros_like(s_channel)
        color_binary[(s_channel >= self.color_threshold[0]) & (s_channel <= self.color_threshold[1])] = 1

        # Stack each channel
        # color_binary = np.dstack((np.zeros_like(gradient_binary), gradient_binary, color_binary)) * 255

        # Combine the two binary thresholds
        combined_binary = np.zeros_like(gradient_binary)
        combined_binary[(color_binary == 1) | (gradient_binary == 1)] = 255

        return combined_binary