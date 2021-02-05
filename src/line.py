# python packages
import logging

# 3rd party packages
import numpy as np

# Define a class to receive the characteristics of each line detection
class Line():
    
    def __init__(self, height):
        logging.debug(f'Initializing the line')

        # was the line detected in the last iteration?
        self.detected = False

        # x values of the last n fits of the line
        self.recent_xfitted = []

        # average x values of the fitted line over the last n iterations
        self.bestx = None

        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None

        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]

        # radius of curvature of the line in some units
        self.radius_of_curvature = None

        # distance in meters of vehicle center from the line
        self.line_base_pos = None

        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')

        # x values for detected line pixels
        self.allx = None

        # y values for detected line pixels
        self.ally = None

        # Generate x and y values for plotting
        self.ploty = np.linspace(0, height - 1, height)

        # Define y-value where we want radius of curvature
        self.y_eval = np.max(self.ploty)

    def fit_polynomial(self):
        # Fit a second order polynomial
        self.current_fit = np.polyfit(self.ally, self.allx, 2)
        self.detected = True

    def calculate_polynomial(self):
        return self.current_fit[0] * self.ploty**2 + self.current_fit[1] * self.ploty + self.current_fit[2]
        
    def color_pixels(self, image, color):
        image[self.ally, self.allx] = color
        return image

    def generate_polygon_points(self, margin):
        fitx = self.calculate_polynomial()
        line_window1 = np.array([np.transpose(np.vstack([fitx - margin, self.ploty]))])
        line_window2 = np.array([np.flipud(np.transpose(np.vstack([fitx + margin, self.ploty])))])
        return np.hstack((line_window1, line_window2))

    def calculate_curvature(self):
        # calculation of R_curve (radius of curvature)
        return ((1 + (2 * self.current_fit[0] * self.y_eval + self.current_fit[1])**2)**1.5) / np.absolute(2 * self.current_fit[0])
        
