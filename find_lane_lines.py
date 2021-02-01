# python packages
import os
import pickle
import logging

# 3rd party packages
import cv2
import numpy as np


def camera_calibration(image_size, directory='camera_cal', pattern_size=(9, 6)):
    """
    Calculates the camera matrix and distortion coefficients

    Based on the given input directory and chessboard pattern size, it determines 
    once image_points and objects points. Once the points are determined the output
    can be calculated based on the given image_size

    :param A: text
    :param B: text
    :param C: text
    
    :return: something
    """
    matrix_file = "camera_matrix.pkl"

    if os.path.exists(matrix_file):

        # Read in the camera calibration matrix
        dist_pickle = pickle.load(open(matrix_file, "rb"))
        object_points = dist_pickle["object_points"]
        image_points = dist_pickle["image_points"]

    else:

        # prepare object points
        points = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        points[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        object_points = []
        image_points = []

        # loop over files in the directory
        for file in os.listdir(directory):

            file_path = os.path.join(directory, file)
            logging.debug(f'Finding corners @ {file_path}')

            # read the image
            image = cv2.imread(file_path)

            # convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

            # If found, add object points, image points
            if ret == True:
                object_points.append(points)
                image_points.append(corners)

        # Save the camera calibration result for later use
        pickle.dump({"object_points":object_points, "image_points":image_points}, open(matrix_file, "wb"))

    # Apply camera calibration given object points and image points
    ret, camera_matrix, distortion_coefficients, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, image_size, None, None)

    # return the 
    return camera_matrix, distortion_coefficients


def color_and_gradient_thresholding(image, s_thresh=(170, 255), sx_thresh=(20, 100)):
    """
    Image processing pipeline
    
    :param A: text
    :param B: text
    :param C: text
    
    :return: something
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
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # Stack each channel
    # color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary)) * 255

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 255

    return combined_binary


def perspective_transform(image, width, height):
    """
    Image processing pipeline
    
    :param A: text
    :param B: text
    :param C: text
    
    :return: something
    """
    # Source points
    src = np.float32([[546, 460], [732, 460], [width, height-10], [0, height-10]])

    # Destination points
    dst = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

    # calculate the perspective transform matrix
    matrix = cv2.getPerspectiveTransform(src, dst)

    # Warp the image using OpenCV warpPerspective()
    return cv2.warpPerspective(image, matrix, (width, height))


def find_lane_pixels(binary_warped, nwindows=9, margin=100, minpix=50):
    """
    Image processing pipeline

    :param A: text
    :param B: text
    :param C: text
    
    :return: something
    """
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)

    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint    

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):

        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height

        ### Find the four below boundaries of the window ###
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0,255,0), 2) 
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0,255,0), 2) 
        
        ### Identify the nonzero pixels in x and y within the window ###
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        ### If you found > minpix pixels, recenter next window ###
        ### (`right` or `leftx_current`) on their mean position ###
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))

        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img

def detect_lane_lines(binary_warped):
    """
    Image processing pipeline
    
    :param A: text
    :param B: text
    :param C: text
    
    :return: something
    """
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    ### Fit a second order polynomial to each using `np.polyfit` ###
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )

    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')

    return out_img

def determine_lane_curvature():
    """
    Calculates the curvature of polynomial functions in pixels.
    
    :param A: text
    :param B: text
    :param C: text
    
    :return: something
    """
    
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    
    ##### Implement the calculation of R_curve (radius of curvature) #####
    left_A = left_fit[0]
    left_B = left_fit[1]
    left_curverad = ((1 + (2 * left_A * y_eval + left_B)**2)**1.5) / np.absolute(2 * left_A)

    right_A = right_fit[0]
    right_B = right_fit[1]
    right_curverad = ((1 + (2 * right_A * y_eval + right_B)**2)**1.5) / np.absolute(2 * right_A)
    
    
    return left_curverad, right_curverad


def pipeline(image, width, height, camera_matrix, distortion_coefficients):
    """
    Image processing pipeline
    
    :param A: text
    :param B: text
    :param C: text
    
    :return: something
    """
    # don't touch the original
    clone = np.copy(image)

    # Distortion correction
    clone = cv2.undistort(clone, camera_matrix, distortion_coefficients, None, None)
    
    # Color/gradient threshold
    clone = color_and_gradient_thresholding(clone)

    # Perspective transform
    clone = perspective_transform(clone, width, height)

    # Detect lane lines
    clone = detect_lane_lines(clone)

    # Determine the lane curvature
    # determine_lane_curvature()

    return clone

def main(file='project_video.mp4'):
    """
    Image processing pipeline
    
    :param A: text
    :param B: text
    :param C: text
    
    :return: something
    """

    # load the video
    capture = cv2.VideoCapture(file)

    # get video meta-data
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    n_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    # define the output video
    out = cv2.VideoWriter(f'output_videos/{file}.avi', cv2.VideoWriter_fourcc('M','J','P','G'), fps, (width, height))

    # camera calibration
    camera_matrix, distortion_coefficients = camera_calibration(image_size=(width, height))

    while(True):
        # Capture frame-by-frame
        ret, frame = capture.read()

        # Stop when there is no frame
        if ret == False:
            break

        # process the frame
        output = pipeline(frame, width, height, camera_matrix, distortion_coefficients)

        # write the frame to the output
        out.write(output)

        # Display the resulting frame
        cv2.imshow('frame', frame)
        cv2.imshow('output', output)

        # wait 1 milisecond on keypress = (esc)
        key_press = cv2.waitKey(1) & 0xFF
        if key_press == 27:
            break
    
    # When everything done, release the video capture and video write objects
    capture.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()