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
        cv2.imshow('output', output)

        # wait 1 milisecond on keypress = (esc)
        key_press = cv2.waitKey(1) & 0xFF
        if key_press == 27:
            break

    # When everything done, release the video capture and video write objects
    del(video)
    cv2.destroyAllWindows()

def put_text(image, text, origin):
    return cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

def draw_boxes(image, nr_boxes=4, offset=10):
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

def picture_in_picture(background_image, foreground_image, box, is_2d=False, offset=10):
    # extract the box_width
    box_width = box[1][0] - box[0][0]

    # get the height and width of the foreground image
    height, width = foreground_image.shape[:2]

    # calculate the new height
    box_height = int((box_width / width) * height)

    # resize the image
    resized = cv2.resize(foreground_image, (box_width, box_height))

    # convert 2d (gray / binary) images to 3d
    if is_2d:
        resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)

    # position the resized image into the box
    background_image[box[0][1] + (2 * offset):box[0][1] + box_height + (2 * offset), box[0][0]:box[1][0], :] = resized

    return background_image


def enrich(image, thresholded, warped, lines):
    # draw some background boxes
    image, boxes = draw_boxes(image)

    # add some text
    image = put_text(image, '- Press (esc) to quit', (boxes[0][0][0] + 10, boxes[0][0][0] + 15))
    image = put_text(image, '- Curvature radius: m', (boxes[0][0][0] + 10, boxes[0][0][0] + 30))
    image = put_text(image, 'thresholded', (boxes[1][0][0] + 10, boxes[0][0][0] + 15))
    image = put_text(image, 'transformed', (boxes[2][0][0] + 10, boxes[0][0][0] + 15))
    image = put_text(image, 'lines detected', (boxes[3][0][0] + 10, boxes[0][0][0] + 15))

    # add the images
    image = picture_in_picture(image, thresholded, boxes[1], True)
    image = picture_in_picture(image, warped, boxes[2], True)
    image = picture_in_picture(image, lines, boxes[3])

    return image

def pipeline(image, camera, threshold, transform, lane):
    # Distortion correction
    undist = camera.undistort(image)
    
    # Color/gradient threshold
    thresholded = threshold.color_and_gradient(undist)

    # Perspective transform
    warped = transform.perspective(thresholded)

    # Detect lane lines
    lines = lane.detect_lines(warped)

    # draw the lane
    output = lane.draw(image, warped, undist, transform)

    # enrich the frame
    output = enrich(output, thresholded, warped, lines)
    
    return output


if __name__ == '__main__':
    main()