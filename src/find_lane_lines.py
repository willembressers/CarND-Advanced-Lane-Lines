# python packages
import os
import logging

# 3rd party packages
import cv2
import click
import numpy as np

# custom packages
from lane import Lane
from video import Video
from camera import Camera
from threshold import Threshold
from transform import Transform

@click.command()
@click.argument('path', type=click.Path(exists=True))
@click.option('-v', '--verbose', count=True)
@click.option('-s', '--show', count=True)
def main(path, verbose, show):
    """
    Main proccessing functionality

    :param path: Path to the input (image, video) to process.
    :param verbose: Let the application be more explicit. (default: False)
    :param show: Show the outcome

    :return: None
    """
    
    # be more explicit
    if verbose: 
        logging.basicConfig(level=logging.DEBUG)

    # check the file extension, in order how to process it.
    file_name, file_extension = os.path.splitext(os.path.basename(path))

    # process video
    if file_extension in ['.mp4']:
        logging.debug(f'Processing video: {path}')
        process_video(file_name, path, show)
    
    # process image
    if file_extension in ['.jpg']:
        logging.debug(f'Processing image: {path}')
        process_image(file_name, path, show)


def process_image(file_name, path, show):
    """
    Loads a given image path, and applies the pipeline to it

    :param file_name: The name of the file (without extension)
    :param path: Path to the input (image, video) to process.
    :param show: Show the outcome

    :return: None
    """

    # load the image
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    height, width = image.shape[:2]

    # initialize the objects
    threshold = Threshold()
    lane = Lane(height=height)
    camera = Camera(image_size=(width, height))
    transform = Transform(width=width, height=height)

    # process the frame
    output = pipeline(image, camera, threshold, transform, lane)

    # save the output
    cv2.imwrite(f'data/processed/output_images/{file_name}.jpg', output)


def process_video(file_name, path, show):
    """
    Loads a given video path, and applies the pipeline to it (frame by frame)

    :param file_name: The name of the file (without extension)
    :param path: Path to the input (image, video) to process.
    :param show: Show the outcome

    :return: None
    """
    # initialize the objects
    video = Video(path)
    threshold = Threshold()
    lane = Lane(height=video.height)
    camera = Camera(image_size=(video.width, video.height))
    transform = Transform(width=video.width, height=video.height)

    # define the output video
    out = cv2.VideoWriter(f'data/processed/output_videos/{file_name}.avi', cv2.VideoWriter_fourcc('M','J','P','G'), video.fps, (video.width, video.height))

    # loop over all frames
    while(video.has_frame()):

        # Capture frame-by-frame
        ret, frame, id = video.get_frame()

        # show progress
        if id % 100 == 0:
            logging.debug(f'Progress: {(id / video.n_frames) * 100:.2f}%')

        # Stop when there is no frame
        if ret == False:
            break

        # process the frame
        output = pipeline(frame, camera, threshold, transform, lane)

        # write the frame to the output
        out.write(output)

        if show:
            # Display the resulting frame
            cv2.imshow('output', output)

            # wait 1 milisecond on keypress = (esc)
            key_press = cv2.waitKey(1) & 0xFF
            if key_press == 27:
                break

    # When everything done, release the video capture and video write objects
    del(video)
    out.release()
    cv2.destroyAllWindows()

def put_text(image, text, origin):
    """
    Add text to the image in an uniform manner.

    :param image: The image to write the text on
    :param text: What to write
    :param origin: Where to put in on the image

    :return: image
    """
    return cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

def draw_boxes(image, nr_boxes=5, offset=10):
    """
    Draws a number of UI (overlay) boxes on the window

    :param image: The image to put the boxes on
    :param nr_boxes: how many boxes
    :param offset: Whats the offset between the boxes and the borders

    :return: image
    """

    boxes = []
    height, width = image.shape[:2]
    
    # determine the box width (based on the number of boxes and the offset)
    box_width = int((width - ((nr_boxes + 1) * offset)) / nr_boxes)
    box_height = int((box_width / width) * height) + (2 * offset)
    
    # copy the image
    mask = image.copy()

    # determine the top_left & bottom_right corners for the first box
    pt1 = (offset, offset)
    pt2 = (offset + box_width, offset + box_height)

    # loop over the boxes
    for _ in range(nr_boxes):

        # collect the box
        boxes.append((pt1, pt2))

        # draw the box
        cv2.rectangle(mask, pt1=pt1, pt2=pt2, color=(0, 0, 0), thickness=cv2.FILLED)
        
        # update the corners for the next box
        pt1 = (pt2[0] + offset, offset)
        pt2 = (pt1[0] + box_width, offset + box_height)
 
    # merge the overlay with the original
    return cv2.addWeighted(src1=mask, alpha=0.2, src2=image, beta=0.3, gamma=0), boxes

def picture_in_picture(background_image, foreground_image, box, nr_channels=3, offset=10):
    """
    Adds a picture into a box on the background image

    :param background_image: The background image
    :param foreground_image: The image to put in the box
    :param box: what box (dimensions) to put in on
    :param nr_channels: how many channels does the foreground image have (2 = gray, 3 = rgb, 4 = rgba)
    :param offset: Whats the offset within the boxes

    :return: image
    """

    # extract the box_width
    box_width = box[1][0] - box[0][0]

    # get the height and width of the foreground image
    height, width = foreground_image.shape[:2]

    # calculate the new height
    box_height = int((box_width / width) * height)

    # resize the image
    resized = cv2.resize(foreground_image, (box_width, box_height))

    # convert 2d (gray / binary) images to 3d
    if nr_channels == 2:
        resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)

    # convert 4d images to 3d
    if nr_channels == 4:
        alpha_channel = dataworkz[:,:,3]
        rgb_channels = dataworkz[:,:,:3]

        # White Background Image
        white_background_image = np.ones_like(rgb_channels, dtype=np.uint8) * 255

        # Alpha factor
        alpha_factor = alpha_channel[:,:,np.newaxis].astype(np.float32) / 255.0
        alpha_factor = np.concatenate((alpha_factor,alpha_factor,alpha_factor), axis=2)

        # Transparent Image Rendered on White Background
        base = rgb_channels.astype(np.float32) * alpha_factor
        white = background_image[box[0][1] + (2 * offset):box[0][1] + box_height + (2 * offset), box[0][0]:box[1][0], :] * (1 - alpha_factor)
        resized = base + white

    # position the resized image into the box
    background_image[box[0][1] + (2 * offset):box[0][1] + box_height + (2 * offset), box[0][0]:box[1][0], :] = resized

    return background_image

def enrich(image, curvature, offset, thresholded, warped, lines):
    """
    Enrich the image with:
    - text (curvature & offset)
    - picture overlays (thresholded, transformed, lines found)

    :return: image
    """

    # draw some background boxes
    image, boxes = draw_boxes(image)

    # add some text
    image = put_text(image, f'Curvature : {curvature:.2f}m', (boxes[0][0][0] + 10, boxes[0][0][0] + 15))
    image = put_text(image, f'Offset : {offset:.2f}m', (boxes[0][0][0] + 10, boxes[0][0][0] + 45))
    image = put_text(image, 'thresholded', (boxes[1][0][0] + 10, boxes[0][0][0] + 15))
    image = put_text(image, 'transformed', (boxes[2][0][0] + 10, boxes[0][0][0] + 15))
    image = put_text(image, 'lines detected', (boxes[3][0][0] + 10, boxes[0][0][0] + 15))

    # add the images
    image = picture_in_picture(image, thresholded, boxes[1], 2)
    image = picture_in_picture(image, warped, boxes[2], 2)
    image = picture_in_picture(image, lines, boxes[3])
    image = picture_in_picture(image, dataworkz, boxes[4], 4)

    return image

def pipeline(image, camera, threshold, transform, lane):
    """
    The actual image processing pipeline, this is the same for video frames as a single image

    :return: image
    """

    # Distortion correction
    undist = camera.undistort(image)
    
    # Color/gradient threshold
    thresholded = threshold.color_and_gradient(undist)

    # Perspective transform
    warped = transform.perspective(thresholded)

    # Detect lane lines
    lines, curvature, offset = lane.detect_lines(warped)

    # draw the lane
    output = lane.draw(warped, undist, transform)

    # enrich the frame
    output = enrich(output, curvature, offset, thresholded, warped, lines)
    
    return output


if __name__ == '__main__':
    # read company logo
    dataworkz = cv2.imread('data/external/dw.png', cv2.IMREAD_UNCHANGED)
    main()