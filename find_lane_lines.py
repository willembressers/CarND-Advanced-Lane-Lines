# python packages
import os
import logging

# 3rd party packages
import cv2
import click

# custom packages
from src.lane import Lane
from src.video import Video
from src.camera import Camera
from src.threshold import Threshold
from src.transform import Transform

@click.command()
@click.argument('path', type=click.Path(exists=True))
@click.option('-v', '--verbose', count=True)
def main(path, verbose):
    """
    Main proccessing functionality

    :param path: Path to the input (image, video) to process.
    :param verbose: Let the application be more explicit. (default: False)

    :return: None
    """
    
    # be more explicit
    if verbose: 
        logging.basicConfig(level=logging.DEBUG)

    # check the file extension, in order how to process it.
    file_name, file_extension = os.path.splitext(path)
    if file_extension in ['.mp4']:
        process_video(path)

def process_video(path):
    """
    Process the video

    :param path: Path to the input (image, video) to process.

    :return: None
    """
    # initialize the objects
    video = Video(path)
    threshold = Threshold()
    lane = Lane(height=video.height)
    camera = Camera(image_size=(video.width, video.height))
    transform = Transform(width=video.width, height=video.height)

    # loop over all frames
    while(video.has_frame()):

        # Capture frame-by-frame
        ret, frame = video.get_frame()

        # Stop when there is no frame
        if ret == False:
            break

        # process the frame
        output = pipeline(frame, camera, threshold, transform, lane)

        # Display the resulting frame
        cv2.imshow('frame', frame)
        cv2.imshow('output', output)

        # wait 1 milisecond on keypress = (esc)
        key_press = cv2.waitKey(1) & 0xFF
        if key_press == 27:
            break

    # When everything done, release the video capture and video write objects
    del(video)
    cv2.destroyAllWindows()


def pipeline(image, camera, threshold, transform, lane):
    # Distortion correction
    undist = camera.undistort(image)
    
    # Color/gradient threshold
    thresholded = threshold.color_and_gradient(undist)

    # Perspective transform
    warped = transform.perspective(thresholded)

    # Detect lane lines
    output = lane.detect_lines(warped)

    # Determine the lane curvature
    # lane.determine_curvature(ploty, left_fit, right_fit)
    
    return output


if __name__ == '__main__':
    main()