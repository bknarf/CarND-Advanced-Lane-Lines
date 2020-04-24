import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import os

from moviepy.editor import VideoFileClip
from pathlib import Path


class Camera:
    def __init__(self, calibration_paths, chessboard_dimensions, chessboard_accepted_failures, do_selftest,
                 calibration_chessboard_path):
        self.calibration_paths = calibration_paths
        self.chessboard_accepted_failures = chessboard_accepted_failures
        self.chessboard_dimensions = chessboard_dimensions
        self.calibration_chessboard_path = calibration_chessboard_path
        calimages = ((ip, mpimg.imread(ip)) for ip in calibration_paths)
        self.calibrated, self.camera_matrix, self.distort_coeff, self.chessboard_found, self.chessboard_notfound = Camera.calibrate(
            calimages, chessboard_dimensions, calibration_chessboard_path)
        self.selftest_result = (False, "not run, was not configured in constructor")
        self.do_selftest = do_selftest
        if do_selftest:
            self.selftest()

    @staticmethod
    def calibrate(calibrationimages, chessboard_dimensions, calibration_chessboard_path):
        chessboard_found = []
        chessboard_notfound = []
        shape = []
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((chessboard_dimensions[0] * chessboard_dimensions[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboard_dimensions[0], 0:chessboard_dimensions[1]].T.reshape(-1, 2)
        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d points in real world space
        imgpoints = []  # 2d points in image plane.

        if calibration_chessboard_path:
            if not os.path.exists(calibration_chessboard_path):
                os.makedirs(calibration_chessboard_path)

        for ip, calimg in calibrationimages:

            calimg_mono = cv2.cvtColor(calimg, cv2.COLOR_BGR2GRAY)
            shape = calimg_mono.shape
            ret, corners = cv2.findChessboardCorners(calimg_mono, (chessboard_dimensions[0], chessboard_dimensions[1]),
                                                     None)

            if ret:
                chessboard_found.append(ip)
                imgpoints.append(corners)
                objpoints.append(objp)
                if calibration_chessboard_path:
                    icbc = cv2.drawChessboardCorners(calimg, (chessboard_dimensions[0], chessboard_dimensions[1]),
                                                     corners, ret)
                    mpimg.imsave(
                        os.path.join(calibration_chessboard_path,
                                     os.path.splitext(os.path.basename(ip))[0] + f".corners.jpg"),
                        icbc)
            else:
                chessboard_notfound.append(ip)

        ret, camera_matrix, distort_coeff, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,
                                                                              (shape[1], shape[0]), None, None)
        return ret, camera_matrix, distort_coeff, chessboard_found, chessboard_notfound

    def undistort(self, img):
        return cv2.undistort(img, self.camera_matrix, self.distort_coeff, None, self.camera_matrix)

    def selftest(self):
        """
        The selftest undistorts all calibrations images and runs the calibration again.
        the expectation is that running the calibration on undistorted images results in lower distortion
        coefficients
        """

        if self.calibrated:
            if len(self.chessboard_notfound) > self.chessboard_accepted_failures[0]:
                self.selftest_result = (
                    False,
                    f"Chessboard detection failed on {len(self.chessboard_notfound)} "
                    + f"initial calibration images. "
                    + f"Accepted failures {self.chessboard_accepted_failures[0]} "
                    + f"Images: {self.chessboard_notfound}")
            else:
                # undistorting calibration images
                undist_images = ((ip + "#undistort", self.undistort(mpimg.imread(ip))) for ip in
                                 self.calibration_paths)
                # test calibration on the undistorted images
                st_calibrated, st_camera_matrix, st_distort_coeff, st_chessboard_found, st_chessboard_notfound = Camera.calibrate(
                    undist_images, self.chessboard_dimensions, False)

                if len(st_chessboard_notfound) > self.chessboard_accepted_failures[1]:
                    self.selftest_result = (
                        False,
                        f"Chessboard detection failed on {len(st_chessboard_notfound)} "
                        + f"undistorted calibration images. "
                        + f"Accepted failures {self.chessboard_accepted_failures[1]} "
                        + f"Images: {st_chessboard_notfound}")
                else:
                    self.selftest_result = (True, "good to go")
        else:
            self.selftest_result = (False, "Camera not calibrated")

    def __str__(self):
        return ','.join([
            "Class:Camera",
            f"calibrated:{self.calibrated}",
            f"selftest_result:{self.selftest_result}",
            f"chessboard_notfound:{self.chessboard_notfound}"]
        )


class Sobel:

    def abs_sobel_thresh(img_1c, orient='x', thresh_min=0, thresh_max=255):
        # Apply x or y gradient with the OpenCV Sobel() function
        # and take the absolute value
        if orient == 'x':
            abs_sobel = np.absolute(cv2.Sobel(img_1c, cv2.CV_64F, 1, 0))
        if orient == 'y':
            abs_sobel = np.absolute(cv2.Sobel(img_1c, cv2.CV_64F, 0, 1))
        # Rescale back to 8 bit integer
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
        # Create a copy and apply the threshold
        binary_output = np.zeros_like(scaled_sobel)
        # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
        binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

        # Return the result
        return binary_output

    # Define a function to return the magnitude of the gradient
    # for a given sobel kernel size and threshold values
    def mag_thresh(img_1c, sobel_kernel=3, mag_thresh=(0, 255)):
        # Take both Sobel x and y gradients
        sobelx = cv2.Sobel(img_1c, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(img_1c, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Calculate the gradient magnitude
        gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
        # Rescale to 8 bit
        scale_factor = np.max(gradmag) / 255
        gradmag = (gradmag / scale_factor).astype(np.uint8)
        # Create a binary image of ones where threshold is met, zeros otherwise
        binary_output = np.zeros_like(gradmag)
        binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

        # Return the binary image
        return binary_output

    # Define a function to threshold an image for a given range and Sobel kernel
    def dir_threshold(img_1c, sobel_kernel=3, thresh=(0, np.pi / 2)):
        # Calculate the x and y gradients
        sobelx = cv2.Sobel(img_1c, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(img_1c, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Take the absolute value of the gradient direction,
        # apply a threshold, and create a binary image result
        absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        binary_output = np.zeros_like(absgraddir)
        binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

        # Return the binary image
        return binary_output


class Pipeline:
    def __init__(self, config):

        # just for reference and debugging
        self.config = config

        self.camera = Camera(**config["camera"])
        self.series_name = False
        self.frame_number = False
        self.histogram = False
        self.histogram_peaks = False
        self.lane_poly = False
        self.right_fit = False

        self.nonzero = False
        self.nonzeroy = False
        self.nonzerox = False
        self.ploty = False

        self.remember_results = config["remember_results"]
        self.expire_results= config["expire_results"]
        self.expire_confidence = config["expire_results"]

        self.max_detection = config["max_detection"]
        self.detected = []

        self.search_poly_threshold = config["search_poly_threshold"]

        self.results = False


        self.naive_windows = config["naive_windows"]
        self.naive_margin = config["naive_margin"]
        self.naive_minpix = config["naive_minpix"]
        self.naive_window_color = config["naive_window_color"]
        self.naive_side_color = config["naive_side_color"]
        self.naive_polyline_color = config["naive_polyline_color"]

        self.poly_margin = config["poly_margin"]
        self.poly_search_color=config["poly_search_color"]
        self.poly_side_color=config["poly_side_color"]
        self.poly_polyline_color=config["poly_polyline_color"]


        self.trapezoid = np.array(config["trapezoid"])
        src = np.float32(self.trapezoid)

        dst = np.float32([self.trapezoid[0], [self.trapezoid[0][0], 0],
                          [self.trapezoid[3][0], 0], self.trapezoid[3]])

        self.transform_matrix = cv2.getPerspectiveTransform(src, dst)

        if (("image_paths" in config) and config["image_paths"]):
            # pipeline is in self-controlled image mode
            # each entry in image_paths is a tuple (in,out) of filepaths
            self.image_paths = config["image_paths"]
        else:
            self.image_paths = False

        self.exportframes = False
        if (("video_paths" in config) and config["video_paths"]):
            # pipeline is in self-controlled video mode
            # each entry in videopaths is a tuple (in,out, (startstop), (exportframes)) of filepaths
            self.video_paths = config["video_paths"]
        else:
            self.video_paths = False

    def start(self):
        if (self.image_paths):
            for ip_in, ip_out in self.image_paths:
                # create output directory
                Path(Path(ip_out).parent).mkdir(parents=True, exist_ok=True)
                self.init_for_file(ip_in, ip_out, "image", (0))
                img_in = mpimg.imread(ip_in)
                img_out = self.process(img_in)
                if len(img_out.shape) == 2:
                    mpimg.imsave(ip_out, img_out, None, None, "gray")
                else:
                    mpimg.imsave(ip_out, img_out)

        if (self.video_paths):
            for vp_in, vp_out, startstop, exportframes in self.video_paths:
                # create output directory
                Path(Path(vp_out).parent).mkdir(parents=True, exist_ok=True)
                self.init_for_file(vp_in, vp_out, "video", exportframes)
                if startstop:
                    in_clip = VideoFileClip(vp_in).subclip(startstop[0], startstop[1])
                else:
                    in_clip = VideoFileClip(vp_in)
                out_clip = in_clip.fl_image(self.process)
                out_clip.write_videofile(vp_out, audio=False)

    def init_for_file(self, input_file, output_file, mode, exportframes):

        self.results = [[],[]]

        self.input_file = input_file
        self.output_file = output_file
        self.frame_number = 0
        self.lane_poly = [False, False]
        self.mode = mode
        self.stage_output = self.config["output"][mode]["stages"]
        self.exportframes = exportframes
        return self

    def draw_trapezoid(self, img_RGB):
        color = [161, 16, 112]
        thickness = 10
        cv2.line(img_RGB, tuple(self.trapezoid[0].astype(int)),
                 tuple(self.trapezoid[1].astype(int)), color, thickness)
        cv2.line(img_RGB, tuple(self.trapezoid[1].astype(int)),
                 tuple(self.trapezoid[2].astype(int)), color, thickness)
        cv2.line(img_RGB, tuple(self.trapezoid[2].astype(int)),
                 tuple(self.trapezoid[3].astype(int)), color, thickness)
        cv2.line(img_RGB, tuple(self.trapezoid[3].astype(int)),
                 tuple(self.trapezoid[0].astype(int)), color, thickness)
        return img_RGB

    def binary_threshold(self, img_RGB):
        HLS = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2HLS)
        gray = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2GRAY)
        R = img_RGB[:, :, 0]
        G = img_RGB[:, :, 1]
        B = img_RGB[:, :, 2]
        H = HLS[:, :, 0]
        L = HLS[:, :, 1]
        S = HLS[:, :, 2]

        H_Sobel = Sobel.mag_thresh(H, sobel_kernel=3, mag_thresh=(30, 100))

        Rthresh = (200, 255)
        binaryR = np.zeros_like(R)
        binaryR[(R > Rthresh[0]) & (R <= Rthresh[1])] = 1

        Sthresh = (150, 255)
        binaryS = np.zeros_like(S)
        binaryS[(S > Sthresh[0]) & (S <= Sthresh[1])] = 1

        Hthresh = (20, 70)
        binaryH = np.zeros_like(H)
        binaryH[(H > Hthresh[0]) & (H <= Hthresh[1])] = 1

        return binaryS

    def perspective_transform(self, img_1C):
        return cv2.warpPerspective(img_1C, self.transform_matrix, (img_1C.shape[1], img_1C.shape[0]))

    def find_lane_pixels(self, binary_warped):
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
        # Create an output image to draw on and visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] // 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # HYPERPARAMETERS
        # Choose the number of sliding windows
        nwindows = 9
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50

        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(binary_warped.shape[0] // nwindows)
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
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                          (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low),
                          (win_xright_high, win_y_high), (0, 255, 0), 2)

            # Identify the nonzero pixels in x and y within the window #
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # If you found > minpix pixels, recenter next window on their mean position
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

    def naive_find_single_lane(self, lr_ind, topdown, peak):

        res = {}
        nonzeroy = self.nonzeroy
        nonzerox = self.nonzerox
        ploty = self.ploty

        # HYPERPARAMETERS
        # Choose the number of sliding windows
        nwindows = self.naive_windows
        # Set the width of the windows +/- margin
        margin = self.naive_margin
        # Set minimum number of pixels found to recenter window
        minpix = self.naive_minpix

        window_height = np.int(topdown.shape[0] // nwindows)

        x_current = peak
        # Step through the windows one by one
        lane_inds = []
        res["window_rectangles"] = []
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = topdown.shape[0] - (window + 1) * window_height
            win_y_high = topdown.shape[0] - window * window_height
            win_x_low = x_current - margin
            win_x_high = x_current + margin
            res["window_rectangles"].append(((win_x_low, win_y_low),(win_x_high, win_y_high)))


            # Identify the nonzero pixels in x and y within the window #
            good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                         (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]

            # Append these indices to the lists
            lane_inds.append(good_inds)

            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_inds) > minpix:
                x_current = np.int(np.mean(nonzerox[good_inds]))

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            lane_inds = np.concatenate(lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass

        # Extract left and right line pixel positions
        res["xs"] = nonzerox[lane_inds]
        res["ys"] = nonzeroy[lane_inds]
        res["detected"] = len(res["xs"])

        # Fit a second order polynomial
        res["lane_poly"] = np.polyfit(res["ys"], res["xs"], 2)

        res["fitx"] = res["lane_poly"][0] * ploty ** 2 + res["lane_poly"][1] * ploty + res["lane_poly"][2]



        return res

    def naive_visualize_single_lane(self, lr_ind, result, rgb_vis, found_pixels=True, boxes = True, polyline = True):


        if boxes:
            for bt in result["window_rectangles"]:
                cv2.rectangle(rgb_vis, bt[0],bt[1], self.naive_window_color[lr_ind], 2)

        if found_pixels:
            rgb_vis[result["ys"], result["xs"]] = self.naive_side_color[lr_ind]

        if polyline:

            #plot the polyline
            offs = 1
            line_window1 = np.array([np.transpose(np.vstack([result["fitx"] - offs, self.ploty]))])
            line_window2 = np.array([np.flipud(np.transpose(np.vstack([result["fitx"] + offs,
                                                                       self.ploty])))])
            line_pts = np.hstack((line_window1, line_window2))
            cv2.fillPoly(rgb_vis, np.int_([line_pts]), self.naive_polyline_color[lr_ind])

        return rgb_vis


    def naive_find_lanes(self, lr, topdown):
        """
        finds one or two lanes without knowledge from previous frames
        :param left: if the left lane should be searched
        :param right: if the right lane should be searched
        :param topdown: the topdown view of the lanes
        :param visualize: If True the algorithms is visualized, if false not
        :param rgb_vis: use this RGB image instead of copying the topdown input
        :return:
        """

        midpoint = np.int(topdown.shape[0] // 2)
        peaks = [False, False]
        if lr[0]:
            # leftside
            histogram = np.sum(topdown[topdown.shape[0] // 2:, 0:midpoint], axis=0)
            peaks[0] = np.argmax(histogram)
        if lr[1]:
            # rightside
            histogram = np.sum(topdown[topdown.shape[0] // 2:, midpoint:], axis=0)
            peaks[1] = np.argmax(histogram) + midpoint


        for i in [0, 1]:
            if (lr[i]):
                res = {"valid": True, "frame_number": self.frame_number, "peak" : peaks[i], "source" : "naive_find_lanes"}
                res.update(self.naive_find_single_lane(i,topdown,peaks[i]))

                self.results[i] = [res] + self.results[i]
                if len(self.results[i]) > self.remember_results:
                    self.results[i] = self.results[i][0:self.remember_results]

    def poly_find_lanes(self, lr, topdown, search_polys = (None,None)):
        for i in [0, 1]:
            if  lr[i] :
                res = {"valid" : True, "frame_number" : self.frame_number, "source" : "poly_find_lanes" }
                res.update(self.poly_find_single_lane(i, topdown, search_polys[i]))
                self.results[i] = [res] + self.results[i]
                if len(self.results[i]) > self.remember_results:
                    self.results[i] = self.results[i][0:self.remember_results]


    def poly_find_single_lane(self, lr_ind, topdown, search_poly):

        res = {}

        ploty = self.ploty
        nonzero = self.nonzero
        nonzerox = self.nonzerox
        nonzeroy = self.nonzeroy

        margin = self.poly_margin

        ### TO-DO: Set the area of search based on activated x-values ###
        ### within the +/- margin of our polynomial function ###
        ### Hint: consider the window areas for the similarly named variables ###
        ### in the previous quiz, but change the windows to our new search area ###
        res["lane_inds"] = ((nonzerox > (search_poly[0] * (nonzeroy ** 2) + search_poly[1] * nonzeroy +
                                  search_poly[2] - margin)) & (
                                 nonzerox < (search_poly[0] * (nonzeroy ** 2) +
                                             search_poly[1] * nonzeroy + search_poly[
                                                 2] + margin)))

        # Again, extract left and right line pixel positions
        res["xs"] = nonzerox[res["lane_inds"] ]
        res["ys"] = nonzeroy[res["lane_inds"] ]

        res["detected"] = len(res["xs"])

        # Fit new polynomials
        res["lane_poly"] = np.polyfit(res["ys"] , res["xs"] , 2)


        ### TO-DO: Calc both polynomials using ploty, left_fit and right_fit ###
        res["fitx"] = search_poly[0] * ploty ** 2 + search_poly[1] * ploty + search_poly[2]

        return res

    def poly_visualize_single_lane(self, lr_ind, result, rgb_vis, found_pixels=True, search_area=True, poly_line=True):


        if search_area:
            # Generate a polygon to illustrate the search window area
            # And recast the x and y points into usable format for cv2.fillPoly()
            line_window1 = np.array([np.transpose(np.vstack([result["fitx"] - self.poly_margin, self.ploty]))])
            line_window2 = np.array([np.flipud(np.transpose(np.vstack([result["fitx"] + self.poly_margin,
                                                                            self.ploty])))])
            line_pts = np.hstack((line_window1, line_window2))

            # Draw the lane onto the warped blank image
            cv2.fillPoly(rgb_vis, np.int_([line_pts]), self.poly_search_color[lr_ind])

        if found_pixels:
            rgb_vis[result["ys"], result["xs"]] = self.poly_side_color[lr_ind]

        if poly_line:
            # Plot the polynomial lines onto the image
            margin = 2
            line_window1 = np.array([np.transpose(np.vstack([result["fitx"] - margin, self.ploty]))])
            line_window2 = np.array([np.flipud(np.transpose(np.vstack([result["fitx"] + margin,
                                                                       self.ploty])))])

            line_pts = np.hstack((line_window1, line_window2))
            cv2.fillPoly(rgb_vis, np.int_([line_pts]), self.poly_polyline_color[lr_ind])

        return rgb_vis



    def confidence_in_result(self,result):
        d = self.frame_number -1 - result["frame_number"]
        ageing = (1-self.expire_confidence)/self.expire_results
        self.detected.append((result["detected"] / self.max_detection) - d*ageing)
        return (result["detected"] / self.max_detection) - d*ageing

    def process(self, inimg):





        # kick out too old results
        for i in [0, 1]:
            self.results[i] = [a for a in self.results[i] if
                                     self.frame_number - a["frame_number"] <= self.expire_results]

        stage = {}
        stage['distorted'] = inimg
        stage['undistorted'] = self.camera.undistort(inimg)
        stage['trapezoid'] = self.draw_trapezoid(np.copy(stage['undistorted']))
        stage["bin_thresh"] = self.binary_threshold(stage['undistorted'])
        stage["perspective_transform_gray"] = self.perspective_transform(
            cv2.cvtColor(inimg, cv2.COLOR_RGB2GRAY))
        stage["perspective_transform"] = self.perspective_transform(stage['bin_thresh'])

        self.nonzero = stage["perspective_transform"].nonzero()
        self.nonzeroy = np.array(self.nonzero[0])
        self.nonzerox = np.array(self.nonzero[1])
        self.ploty = np.linspace(0, stage["perspective_transform"].shape[0] - 1,
                                 stage["perspective_transform"].shape[0])


        #decide for each side which algorithm to use for this frame
        use_poly_find = [False,False]
        poly_lanes = [False,False]
        for i in [0,1]:

            if len(self.results[i]) > 0 and self.confidence_in_result(self.results[i][0]) > self.search_poly_threshold :
                use_poly_find[i] = True
                poly_lanes[i] = self.results[i][0]["lane_poly"]
            else:
                use_poly_find[i] = False

        self.poly_find_lanes(use_poly_find, stage["perspective_transform"], poly_lanes)

        self.naive_find_lanes((not(use_poly_find[0]),not(use_poly_find[1])), stage["perspective_transform"])



        #search the results with the highest confidence for each side and visualize them

        stage["find_lane"] = cv2.cvtColor(stage["perspective_transform"], cv2.COLOR_GRAY2RGB)

        for i in [0,1]:
            if self.results[i][0]["source"] == "poly_find_lanes":
                stage["find_lane"] = self.poly_visualize_single_lane(i, self.results[i][0],stage["find_lane"])
            else:
                stage["find_lane"]=self.naive_visualize_single_lane(i, self.results[i][0],stage["find_lane"])


        if (self.mode == "image"):
            for st in self.stage_output:
                out_img = stage[st]
                if len(out_img.shape) == 2:
                    mpimg.imsave(os.path.join(os.path.splitext(self.output_file)[0]
                                              + '.' + st + os.path.splitext(self.output_file)[1]), out_img,
                                 None, None, "gray")
                else:
                    mpimg.imsave(os.path.join(os.path.splitext(self.output_file)[0]
                                              + '.' + st + os.path.splitext(self.output_file)[1]), out_img)
        elif (self.exportframes and self.frame_number in self.exportframes):
            for st in self.stage_output:
                out_img = stage[st]
                if len(out_img.shape) == 2:
                    mpimg.imsave(os.path.join(os.path.splitext(self.output_file)[0]
                                              + '.' + str(self.frame_number) + '.' + st + ".jpg"), out_img,
                                 None, None, "gray")
                else:
                    mpimg.imsave(os.path.join(os.path.splitext(self.output_file)[0]
                                              + '.' + str(self.frame_number) + '.' + st + ".jpg"), out_img)

        self.frame_number = self.frame_number + 1
        return stage["find_lane"]


def configure():
    pipeline_config = {}
    pipeline_config["camera"] = {}
    pipeline_config["camera"]["calibration_paths"] = glob.glob("camera_cal/*")
    pipeline_config["camera"]["calibration_chessboard_path"] = False
    pipeline_config["camera"]["chessboard_dimensions"] = [9, 6]
    pipeline_config["camera"]["do_selftest"] = False
    # first number initial calibration, second number calibration failures on undistorted images
    pipeline_config["camera"]["chessboard_accepted_failures"] = [3, 4]



    # gimp coordinates, clockwise, bottom left first, taken from thresholded image
    measured_trapezoid = [
        (194, 719), (581, 460),
        (702, 460), (1115, 719)
    ]


    pipeline_config["trapezoid"] = measured_trapezoid

    #how many results to keep
    pipeline_config["remember_results"] = 5

    #when to expire results
    pipeline_config["expire_results"] = 25

    #how much confidence in a result which will expire next frame
    pipeline_config["expire_confidence"] = 0

    #how many pixels in a perfectly identified lane
    pipeline_config["max_detection"] = 20000

    #how high must the confidence of a fitted polygon be to go from naive finding to poly finding
    pipeline_config["search_poly_threshold"] = 0.5


    pipeline_config["naive_windows"] = 9
    pipeline_config["naive_margin"] = 100
    pipeline_config["naive_minpix"] = 50
    pipeline_config["naive_window_color"] = ((0, 255, 0), (0, 255, 0))
    pipeline_config["naive_side_color"] = ((0, 255, 0), (255, 0, 0))
    pipeline_config["naive_polyline_color"] = ((255, 255, 0), (255, 255, 0))

    pipeline_config["poly_margin"] = 100
    pipeline_config["poly_search_color"] = ((30, 151, 227), (30, 151, 227))
    pipeline_config["poly_side_color"] = ((255, 0, 230), (255, 23, 42))
    pipeline_config["poly_polyline_color"] = ((255, 111, 0), (255, 111, 0))


    pipeline_config["output"] = {}
    pipeline_config["output"]["image"] = {}
    # possible stages are: ['undistorted','bin_thresh']
    pipeline_config["output"]["image"]["stages"] = ["undistorted", "find_lane", "perspective_transform"]
    pipeline_config["output"]["video"] = {}
    # this only affects exported single frames, see the last tuple of each videopath
    pipeline_config["output"]["video"]["stages"] = ["undistorted", "find_lane", "perspective_transform"]

    imgpaths = []
    for p in glob.glob("test_images/*"):
        imgpaths.append((p, os.path.join("test_images_output", os.path.basename(p))))

    pipeline_config["image_paths"] = imgpaths

    pipeline_config["video_paths"] = [("project_video.mp4", "output_videos/project_video_out.mp4", (0,5), (10,50,250))]
    return pipeline_config


pl = Pipeline(configure())
pl.start()
print(pl.detected)
