# python packages
import os
import pickle
import logging

# 3rd party packages
import cv2
import numpy as np

class Camera:
    matrix_file = "data/processed/matrix.pkl"
    training_directory = "data/raw/calibration"

    def __init__(self, image_size, chessboard_pattern=(9, 6)):
        """
        Initializes the camera

        :param image_size: Whats the size (width, height) of the image?
        :param directory: Where is the directory of the chessboard images?
        :param chessboard_pattern: What chessboard pattern size is used?

        :return: None
        """
        logging.debug(f'Initializing the camera')
        self.chessboard_pattern = chessboard_pattern

        # check if the camera_matrix exists
        if not os.path.exists(self.matrix_file):
            self.generate_calibration_matrix()

        # load the calibration matrix
        self.load_calibration_matrix(image_size)

    def undistort(self, image):
        """
        Loads the camera calibration matrix + distorion coefficients

        :return None
        """
        return cv2.undistort(image, self.camera_matrix, self.distortion_coefficients, None, None)

    def load_calibration_matrix(self, image_size):
        """
        Loads the camera calibration matrix + distorion coefficients

        :return None
        """
        logging.debug(f'Loading camera matrix')

        # Read in the camera calibration matrix
        dist_pickle = pickle.load(open(self.matrix_file, "rb"))
        object_points = dist_pickle["object_points"]
        image_points = dist_pickle["image_points"]

        # Apply camera calibration given object points and image points
        ret, camera_matrix, distortion_coefficients, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, image_size, None, None)

        self.camera_matrix = camera_matrix
        self.distortion_coefficients = distortion_coefficients

    def generate_calibration_matrix(self):
        """
        Generate the camera calibration matrix + distorion coefficients
        and stores it in a pickle file

        :return: None
        """
        logging.debug(f'Camera matrix not found, generating...')

        # prepare object points
        points = np.zeros((self.chessboard_pattern[0] * self.chessboard_pattern[1], 3), np.float32)
        points[:, :2] = np.mgrid[0:self.chessboard_pattern[0], 0:self.chessboard_pattern[1]].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        object_points = []
        image_points = []

        # loop over files in the directory
        for file in os.listdir(self.training_directory):

            if file.endswith((".jpg")):

                file_path = os.path.join(self.training_directory, file)
                logging.debug(f'Finding corners @ {file_path}')

                # read the image
                image = cv2.imread(file_path)

                # convert to grayscale
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # Find the chessboard corners
                ret, corners = cv2.findChessboardCorners(gray, self.chessboard_pattern, None)

                # If found, add object points, image points
                if ret == True:
                    object_points.append(points)
                    image_points.append(corners)

        # Save the camera calibration result for later use
        pickle.dump({
            "object_points":object_points, 
            "image_points":image_points
        }, open(self.matrix_file, "wb"))