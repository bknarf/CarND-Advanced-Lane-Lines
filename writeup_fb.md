
**Advanced Lane Finding Project**

The techniques employed in this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

### Writeup

#### Overview
All relevant code resides in alfp.py.
Input and output settings can be directly set in `Pipeline.___init___`.
`Pipeline.image_paths` and `Pipeline.video_paths` are tuple-based, which allows `Pipeline.video_paths`
to also handle subclips and exporting single frames from videos.
`Pipeline.process_result` can be altered to export other stages to an output video.
The best usage is to alter it to `'all_vis'`, which not only displays the calculated lane, but also which algorithm
for finding lane marking pixels was used (sliding window or search poly).

Parameters are contained in `Pipeline.___init___` or as static (class) variables on `LaneMarking`.


### Camera Calibration

The functionality to do camera calibration is located in its own class `Camera`.
`Camera.__init__` encapsulates all required parametrization.
Instances expose the `undistort` method, which allows undistorting images.
The class also has an experimental feature for selftesting the calibration result.
The idea was to compare calibration results from distorted chessboard images and
undistorted chessboard images and then compare the results.
The calibration on undistorted images was expected to yield *lower* correction values,
but this feature did not reach an usable state, but can be seen in `Camera.selftest`.
Pictures with chessboard corners can be exported by setting `Camera.calibration_chessboard_path`
to a directory.
![a chessboard image](/chessboard/calibration2.corners.jpg "a chessboard image")

During development it was identified that `calibration1.jpg`, `calibration4.jpg` and `calibration5.jpg`
do not show the chessboard pattern completely. `Camera.calibrate` tries to use 
them for calibration but fails silently. `Camera.chessboard_notfound` can be inspected
on `Camera` instances.

In general `Camera` can be instantiated separately from `Pipeline` if needed.
`Camera` is only instantiated once per `Pipeline` and each `Pipeline` instance can handle
multiple images and videos.

### Pipeline
The **a**dvanced **l**ane **f**inding **p**ipeline (hence the filename `alfp.py`) has four phases.
+ `Pipeline.__init__` contains the necessary configuration, instantiates a `Camera` and calculates the transformation
matrix for perspective transformation.

+ `Pipeline.start` must be called from the outside. The processing does not start directly at the of the constructor.
This allows for experimentation to manipulate the `Pipeline` instance
`Pipeline.start` then iterates through all files and calls `Pipeline.init_for_file` for each file.
`Pipeline.init_for_file` takes resets changing values of the pipeline, so a new file can be processed.

+ The actual processing of images is then done in `Pipeline.process`  

`Pipeline.process` internally produces intermediate images for each stage which can be easily exported.
The available stage-outputs are
+ 'distorted' is identical to the input image
+ 'undistorted' is undistorted by `Camera.undistort`
+ 'trapezoid' shows the trapezoid super imposed on the input image.
This is merely for visualization and debugging and not fed into further pipeline stages.
+ 'bin_thresh' shows which pixels are activated for lanefinding. `Pipeline.binary_threshold` creates this output.
+ 'perspective_transform' is the output of the perspective transform. The trapezoid is transformed into a topdown
view by `Pipeline.perspective_transform`
+ 'find_lane' shows the output of the lane finding of this frame. The pipeline does only one lane finding run per
side, which can either be done by `LaneMarking.find_by_sliding_windows` and `LaneMarking.find_by_poly`
+ 'inverse_perspective_transform' is 'find_lane' transformed back into the initial perspective.
+ 'all_vis' combines all visualizations and includes the curvatures for both lane markings and the position of the
vehicle within the lane. 'all_vis' is an alternative to 'lanearea'
+ 'lanearea' only displays the calculated curvatures, position within the lane and the lane area.

#### Distortion removal

Distortion is removed by calling `Camera.undistort`.


#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
