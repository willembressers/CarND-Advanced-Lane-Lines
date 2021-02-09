# python packages
import logging

# 3rd party packages
import cv2
import numpy as np

# custom packages
from line import Line

class Lane():

    def __init__(self, height, nwindows=9, margin=100, minpix=50):
        """
        Initializes the Lane

        :return: None
        """

        self.nwindows = nwindows
        self.margin = margin
        self.minpix = minpix
        self.ploty = np.linspace(0, height - 1, height)
        self.ym_per_pix = 30 / 720
        self.xm_per_pix = 3.7 / 700

        # initialize the lines
        self.left_line = Line(height)
        self.right_line = Line(height) 

    def sliding_windows(self, binary_image):
        """
        Searches for the lane lines by:
        - take a historgram of all (bottom half) values
        - calculate the maximum values before (left) the center en after (right) the center
        - loop over the search windows (which are centered around the maximum histogram values)
        - identify nonzero pixels
        - update the window centers for the next itteration

        :param binary_image: The binary input image

        :return: image
        """
        # Create an output image to draw on and visualize the result
        out_img = np.dstack((binary_image, binary_image, binary_image))

        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(binary_image.shape[0] // self.nwindows)

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_image[binary_image.shape[0] // 2:,:], axis=0)

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] // 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint    

        # Current positions to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(self.nwindows):

            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_image.shape[0] - (window + 1) * window_height
            win_y_high = binary_image.shape[0] - window * window_height

            ### Find the four below boundaries of the window ###
            win_xleft_low = leftx_current - self.margin
            win_xleft_high = leftx_current + self.margin
            win_xright_low = rightx_current - self.margin
            win_xright_high = rightx_current + self.margin
            
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
            if len(good_left_inds) > self.minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))

            if len(good_right_inds) > self.minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        self.left_line.allx = nonzerox[left_lane_inds]
        self.left_line.ally = nonzeroy[left_lane_inds] 
        self.right_line.allx = nonzerox[right_lane_inds]
        self.right_line.ally = nonzeroy[right_lane_inds]

        # fit polynomals
        self.left_line.fit_polynomial(self.ym_per_pix, self.xm_per_pix)
        self.right_line.fit_polynomial(self.ym_per_pix, self.xm_per_pix)

        # Colors in the left and right lane regions
        out_img = self.left_line.color_pixels(out_img, [255, 0, 0])
        out_img = self.right_line.color_pixels(out_img, [0, 0, 255])

        return out_img

    def previous_fits(self, binary_image):
        """
        Searches for the lane lines by:
        - using the lane lines polygons from the previous frame
        - add a margin around it
        - try to fit a new polygon

        :param binary_image: The binary input image

        :return: image
        """

        # Grab activated pixels
        nonzero = binary_image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        ### Set the area of search based on activated x-values ###
        ### within the +/- margin of our polynomial function ###
        left_lane_inds = ((nonzerox > (self.left_line.current_fit[0] * (nonzeroy**2) + self.left_line.current_fit[1] * nonzeroy + self.left_line.current_fit[2] - self.margin)) & (nonzerox < (self.left_line.current_fit[0] * (nonzeroy**2) + self.left_line.current_fit[1] * nonzeroy + self.left_line.current_fit[2] + self.margin)))
        right_lane_inds = ((nonzerox > (self.right_line.current_fit[0] * (nonzeroy**2) + self.right_line.current_fit[1] * nonzeroy + self.right_line.current_fit[2] - self.margin)) & (nonzerox < (self.right_line.current_fit[0] * (nonzeroy**2) + self.right_line.current_fit[1] * nonzeroy + self.right_line.current_fit[2] + self.margin)))
    
        # Extract left and right line pixel positions
        self.left_line.allx = nonzerox[left_lane_inds]
        self.left_line.ally = nonzeroy[left_lane_inds] 
        self.right_line.allx = nonzerox[right_lane_inds]
        self.right_line.ally = nonzeroy[right_lane_inds]

        # fit polynomals
        self.left_line.fit_polynomial(self.ym_per_pix, self.xm_per_pix)
        self.right_line.fit_polynomial(self.ym_per_pix, self.xm_per_pix)

        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((binary_image, binary_image, binary_image))*255
        window_img = np.zeros_like(out_img)

        # Color in left and right line pixels
        out_img = self.left_line.color_pixels(out_img, [255, 0, 0])
        out_img = self.right_line.color_pixels(out_img, [0, 0, 255])

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_pts = self.left_line.draw_search_polygon(self.margin, window_img, [255, 0, 0])
        right_line_pts = self.right_line.draw_search_polygon(self.margin,  window_img, [0, 0, 255])

        # Draw the lane onto the warped blank image
        return cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    def draw(self, warped, undist, transform):
        """
        Draws the lane on the image:

        :param warped: The transformed binary undistored image 
        :param undist: The undistored image to draw on
        :param transform: The transformation object (for inverse transformation)

        :return: image
        """

        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([self.left_line.calculate_polynomial(), self.ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([self.right_line.calculate_polynomial(), self.ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, transform.inverse_matrix, (undist.shape[1], undist.shape[0])) 
        
        # Combine the result with the original image
        return cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    def calculate_offset(self, binary_image):
        """
        Calculates the offset to the center
        - by looking in the bottom window
        - and collecting all the x values 
        - calculate an average per line
        - determine the center of the lane
        - determine the center of the frame
        - calculating the (absolute) offset between the centers

        :param binary_image: The transformed binary undistored image 

        :return: integer
        """

        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(binary_image.shape[0] // self.nwindows)

        # get the bottom window
        bottom_window = binary_image.shape[0] - window_height

        # get an average of all x values in the bottom window
        left_line_x_bottom = np.mean(self.left_line.allx[self.left_line.ally >= bottom_window])
        right_line_x_bottom = np.mean(self.right_line.allx[self.right_line.ally >= bottom_window])
        
        # determine the center of the lane
        lane_center = left_line_x_bottom + (right_line_x_bottom - left_line_x_bottom) / 2
        
        # determine the frame center
        frame_center = binary_image.shape[1] / 2

        # calculate the offset
        return abs(lane_center - frame_center) * self.xm_per_pix

    def detect_lines(self, binary_image):
        """
        Detects the lines in the given image
        - collects the lane curvature
        - collects the offset

        :param binary_image: The transformed binary undistored image 

        :return: image, float, integer
        """

        if self.left_line.detected and self.right_line.detected:
            out_img = self.previous_fits(binary_image)
        else:
            out_img = self.sliding_windows(binary_image)

        # calculate the curvature
        curvature = np.mean([
            self.left_line.radius_of_curvature,
            self.right_line.radius_of_curvature
        ])

        # calculate the offset
        offset = self.calculate_offset(binary_image)

        return out_img, curvature, offset