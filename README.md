# Advanced Lane Finding

<iframe width="560" height="315" src="https://www.youtube.com/embed/vn48abobflA" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Project Organization

    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    └── src                <- Source code for use in this project.

## Installation
I suggest to create an new virtual (python) environment with `mkvirtualenv`, and then install the dependencies with.
```bash
pip install -r requirements.txt
```

## Usage
Run the python file `src/find_lane_lines.py`. It requires an input path which can be a video or image. In the `data/raw/` are some `test_videos` and `test_images`. You can use the flag `-v, --verbose` to let the application be more explicit. You can use the flag `-s, --show` to show the live process on video's.
```bash
python src/find_lane_lines.py data/raw/test_videos/project_video.mp4 -s -v
```

## Project description
The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

### Camera calibration
Before the application applies the `processing pipeline` to the image or video it checks attempts to load the camera calibration matrix. If the matrix is not available, it will generate one based on the images in `data/raw/calibration`.
```python
# check if the camera_matrix exists
if not os.path.exists(self.matrix_file):
    self.generate_calibration_matrix()

# load the calibration matrix
self.load_calibration_matrix(image_size)
```

### Pipeline
Now we can apply the same pipeline on an image or every frame in an video.
```python
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
```

#### Distortion correction
The distorion correction will apply the camera matrix on every image, which will correct for any camera distortions.
```python
cv2.undistort(image, self.camera_matrix, self.distortion_coefficients, None, None)
```

#### Color/gradient threshold
First i'll convert the image from `RGB` formal to `HLS` format. Now i can use the `saturation` channel to create an binary image with all zero's except for the pixels where the saturation lays within my thresholds.
```python
# Convert to HLS color space and separate the V channel
hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
s_channel = hls[:,:,2]

# Threshold color channel
color_binary = np.zeros_like(s_channel)
color_binary[(s_channel >= self.color_threshold[0]) & (s_channel <= self.color_threshold[1])] = 1
```

Since i've allready converted the image to the `HLS` format, i can use the `lightness` channel as input for my (sobel) horizontal edge detector. Now i can create an binary image with all zero's except for the pixels where there is an edge. 
```python
l_channel = hls[:,:,1]

# Sobel x
sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

# Threshold x gradient
gradient_binary = np.zeros_like(scaled_sobel)
gradient_binary[(scaled_sobel >= self.gradient_threshold[0]) & (scaled_sobel <= self.gradient_threshold[1])] = 1
```

Now i combine the 2 binary images to 1 binary image, which highlights bright colors and edges.
```python
# Combine the two binary thresholds
combined_binary = np.zeros_like(gradient_binary)
combined_binary[(color_binary == 1) | (gradient_binary == 1)] = 255
```

#### Perspective transform
```python

```

#### Detect lane lines
```python

```

#### draw the lane
```python

```

#### enrich the frame
```python

```
