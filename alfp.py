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


class Pipeline:
    def __init__(self, config):

        # just for reference and debugging
        self.config = config

        self.poly_summary = []
        self.diverging_lanes_frames = False
        self.low_detection_frames = False
        self.invalid_lane_frames = False
        self.no_lane_frames = False
        self.one_lane_frames = False
        self.converging_lanes_frames = False
        self.converging_lanes_threshold = config["converging_lanes_threshold"]
        self.diverging_lanes_frames = False
        self.diverging_lanes_threshold = config["diverging_lanes_threshold"]

        self.frames_of_interest_summary = {}

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
        self.expire_results = config["expire_results"]
        self.expire_confidence = config["expire_results"]

        self.max_detection = config["max_detection"]
        self.min_detection = config["min_detection"]

        self.search_poly_threshold = config["search_poly_threshold"]

        self.results = False

        self.naive_windows = config["naive_windows"]
        self.naive_margin = config["naive_margin"]
        self.naive_minpix = config["naive_minpix"]
        self.naive_window_color = config["naive_window_color"]
        self.naive_side_color = config["naive_side_color"]
        self.naive_polyline_color = config["naive_polyline_color"]

        self.poly_margin = config["poly_margin"]
        self.poly_search_color = config["poly_search_color"]
        self.poly_side_color = config["poly_side_color"]
        self.poly_polyline_color = config["poly_polyline_color"]

        self.curvature_font = config["curvature_font"]
        self.curvature_font_scale = config["curvature_font_scale"]
        self.curvature_ym_per_pix = config["curvature_ym_per_pix"]
        self.curvature_xm_per_pix = config["curvature_xm_per_pix"]

        self.trapezoid = np.array(config["trapezoid"])
        self.region_of_interest_offsets = config["region_of_interest_offsets"]
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

        self.results = [[], []]

        self.diverging_lanes_frames = []
        self.low_detection_frames = []
        self.invalid_lane_frames = []
        self.no_lane_frames = []
        self.one_lane_frames = []
        self.converging_lanes_frames = []
        self.frames_of_interest_summary[input_file] = {"diverging_lanes_frames": self.diverging_lanes_frames,
                                                       "low_detection_frames": self.low_detection_frames,
                                                       "invalid_lanes_frames": self.invalid_lane_frames,
                                                       "no_lane_frames": self.no_lane_frames,
                                                       "one_lane_frames": self.one_lane_frames
                                                       }

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

    def region_of_interest(self, img, vertices):
        """
        Applies an image mask.

        Only keeps the region of the image defined by the polygon
        formed from `vertices`. The rest of the image is set to black.
        `vertices` should be a numpy array of integer points.
        """
        # defining a blank mask to start with
        mask = np.zeros_like(img)

        # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        if len(img.shape) > 2:
            channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255

        vertices = np.array([[vertices]], dtype=np.int32)
        # filling pixels inside the polygon defined by "vertices" with the fill color
        cv2.fillPoly(mask, vertices, ignore_mask_color)

        # returning the image only where mask pixels are nonzero
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image

    def binary_threshold(self, img_RGB):

        HLS = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2HLS)
        H = HLS[:, :, 0]
        L = HLS[:, :, 1]
        S = HLS[:, :, 2]

        Hthresh = (0, 100)
        Sthresh = (100, 255)
        Lthresh = (200, 255)

        binaryHLS = np.zeros_like(S)
        binaryHLS[(((S > Sthresh[0]) & (S <= Sthresh[1])) | (L > Lthresh[0]) & (L <= Lthresh[1])) & (
                (H > Hthresh[0]) & (H <= Hthresh[1]))] = 255

        mt = self.trapezoid
        offsets = self.region_of_interest_offsets

        to_mask = ((mt[0][0] - offsets[0][0], mt[0][1] + offsets[0][1]),
                   (mt[1][0] - offsets[1][0], mt[1][1] - offsets[1][1]),
                   (mt[2][0] + offsets[1][0], mt[2][1] - offsets[1][1]),
                   (mt[3][0] + offsets[0][0], mt[3][1] + offsets[0][1]))

        return self.region_of_interest(binaryHLS, to_mask)

    def perspective_transform(self, img_in):
        return cv2.warpPerspective(img_in, self.transform_matrix, (img_in.shape[1], img_in.shape[0]))

    def inverse_perspective_transform(self, img_in):
        return cv2.warpPerspective(img_in, self.transform_matrix, (img_in.shape[1], img_in.shape[0]),
                                   flags=cv2.WARP_INVERSE_MAP)

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
            res["window_rectangles"].append(((win_x_low, win_y_low), (win_x_high, win_y_high)))

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

        if res["detected"] > self.min_detection:
            # Fit a second order polynomial
            res["lane_poly"] = np.polyfit(res["ys"], res["xs"], 2)

            res["fitx"] = res["lane_poly"][0] * ploty ** 2 + res["lane_poly"][1] * ploty + res["lane_poly"][2]
            res["valid"] = True
        else:
            res["valid"] = False

        return res

    def naive_visualize_single_lane(self, result, rgb_vis, found_pixels=True, boxes=True, polyline=True):
        lr_ind = result["lr_ind"]
        if boxes:
            for bt in result["window_rectangles"]:
                cv2.rectangle(rgb_vis, bt[0], bt[1], self.naive_window_color[lr_ind], 2)

        if found_pixels:
            rgb_vis[result["ys"], result["xs"]] = self.naive_side_color[lr_ind]

        if polyline:
            # plot the polyline
            offs = 10
            line_window1 = np.array([np.transpose(np.vstack([result["fitx"] - offs, self.ploty]))])
            line_window2 = np.array([np.flipud(np.transpose(np.vstack([result["fitx"] + offs,
                                                                       self.ploty])))])
            line_pts = np.hstack((line_window1, line_window2))
            cv2.fillPoly(rgb_vis, np.int_([line_pts]), self.naive_polyline_color[lr_ind])

        return rgb_vis

    def naive_find_lanes(self, lr, topdown):

        midpoint = np.int(topdown.shape[1] // 2)
        peaks = [False, False]
        if lr[0]:
            # leftside
            histogram = np.sum(topdown[topdown.shape[0] // 2:, 0:midpoint], axis=0)
            peaks[0] = np.argmax(histogram)
            if (peaks[0] == 0):
                peaks[0] = int(midpoint / 2)
        if lr[1]:
            # rightside
            histogram = np.sum(topdown[topdown.shape[0] // 2:, midpoint:], axis=0)
            peaks[1] = np.argmax(histogram) + midpoint
            if (peaks[1] == midpoint):
                peaks[1] = int((midpoint / 2) * 3)

        for i in [0, 1]:
            if (lr[i]):
                res = {"frame_number": self.frame_number, "peak": peaks[i], "source": "naive_find_lanes", "lr_ind": i}
                res.update(self.naive_find_single_lane(i, topdown, peaks[i]))

                self.results[i] = [res] + self.results[i]
                if len(self.results[i]) > self.remember_results:
                    self.results[i] = self.results[i][0:self.remember_results]

    def poly_find_lanes(self, lr, topdown, search_polys=(None, None)):
        for i in [0, 1]:
            if lr[i]:
                res = {"valid": True, "frame_number": self.frame_number, "source": "poly_find_lanes", "lr_ind": i}
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
        res["xs"] = nonzerox[res["lane_inds"]]
        res["ys"] = nonzeroy[res["lane_inds"]]

        res["detected"] = len(res["xs"])

        if res["detected"] > self.min_detection:
            # Fit new polynomials
            res["lane_poly"] = np.polyfit(res["ys"], res["xs"], 2)
            res["fitx"] = search_poly[0] * ploty ** 2 + search_poly[1] * ploty + search_poly[2]
            res["valid"] = True
        else:
            res["valid"] = False
        return res

    def poly_visualize_single_lane(self, result, rgb_vis, found_pixels=True, search_area=True, poly_line=True):
        lr_ind = result["lr_ind"]
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
            margin = 10
            line_window1 = np.array([np.transpose(np.vstack([result["fitx"] - margin, self.ploty]))])
            line_window2 = np.array([np.flipud(np.transpose(np.vstack([result["fitx"] + margin,
                                                                       self.ploty])))])

            line_pts = np.hstack((line_window1, line_window2))
            cv2.fillPoly(rgb_vis, np.int_([line_pts]), self.poly_polyline_color[lr_ind])

        return rgb_vis

    def visualize_result(self, result, rgb_vis):
        if result["source"] == "poly_find_lanes":
            return self.poly_visualize_single_lane(result, rgb_vis)
        elif result["source"] == "naive_find_lanes":
            return self.naive_visualize_single_lane(result, rgb_vis)
        else:
            print(f"unknown source for visualizing:{result['source']}")

    def confidence_in_result(self, result):
        if (result["valid"]):
            d = self.frame_number - 1 - result["frame_number"]
            ageing = (1 - self.expire_confidence) / self.expire_results
            return (result["detected"] / self.max_detection) - d * ageing
        else:
            return -1

    def better_worse(self, a, b):
        # the more points, the better
        activated_sum = []
        # the closer the peak of the poly is to ymax, the better is the quality
        poly_peak_y = []
        for a_or_b in (a, b):
            activated_sum.append(len(a_or_b["ys"]))
            poly_peak_y.append(-a_or_b["lane_poly"][1] / (2 * a_or_b["lane_poly"][0]))

        activated_sum_ratios = [activated_sum[0] / activated_sum[1], activated_sum[1] / activated_sum[0]]
        poly_peak_y_ratios = [poly_peak_y[0] / poly_peak_y[1], poly_peak_y[1] / poly_peak_y[0]]
        if activated_sum_ratios[0] + poly_peak_y_ratios[0] > activated_sum_ratios[1] + poly_peak_y_ratios[1]:
            return (a, b)
        else:
            return (b, a)

    def put_curvature(self, inimg, result_lr):
        curvats = []
        curv_centers=[inimg.shape[1]//4,(inimg.shape[1]//4)*3,(inimg.shape[1]//2)]
        for i in [0, 1]:
            if result_lr != False:
                curvats.append(((1 + (2 * result_lr[0]["lane_poly"][0] * self.ploty[-1] * self.curvature_ym_per_pix +
                                      result_lr[0]["lane_poly"][1]) ** 2) ** 1.5) / np.absolute(
                    2 * result_lr[0]["lane_poly"][0]))
        if len(curvats) == 2:
            curvats.append((curvats[0] + curvats[1]) / 2)

        for center, curv in zip(curv_centers, curvats):
            # get boundary of this text
            txt = "{:.2f}".format(curv)
            textsize = cv2.getTextSize(txt, self.curvature_font, self.curvature_font_scale, 2)[0]
            cv2.putText(inimg, txt, (center - textsize[0] // 2, textsize[1]), self.curvature_font,
                        self.curvature_font_scale, (0, 0, 0), 2)

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

        # decide for each side which algorithm to use for this frame
        use_poly_find = [False, False]
        poly_lanes = [False, False]
        for i in [0, 1]:

            if len(self.results[i]) > 0 and not (
                    "converged_worse" in self.results[i][0].keys()) and self.confidence_in_result(
                self.results[i][0]) > self.search_poly_threshold:
                use_poly_find[i] = True
                poly_lanes[i] = self.results[i][0]["lane_poly"]
            else:
                use_poly_find[i] = False

        self.poly_find_lanes(use_poly_find, stage["perspective_transform"], poly_lanes)

        self.naive_find_lanes((not (use_poly_find[0]), not (use_poly_find[1])), stage["perspective_transform"])

        stage["find_lane"] = cv2.cvtColor(stage["perspective_transform"], cv2.COLOR_GRAY2RGB)

        first_valid_res = [False, False]
        for i in [0, 1]:
            first_valid_res[i] = next((r for r in self.results[i] if r["valid"]), False)

        if first_valid_res == [False, False]:
            print("no valid lanes, neither left or right, nothing to visualize")
            self.no_lane_frames.append(self.frame_number)
        elif first_valid_res[0] != False and first_valid_res[1] != False:
            # two valid results, great
            curve_same_way = first_valid_res[0]["lane_poly"][0] > 0 == first_valid_res[1]["lane_poly"][0] > 0
            curve_diff = abs(first_valid_res[0]["lane_poly"][0] - first_valid_res[1]["lane_poly"][0])
            self.poly_summary.append((first_valid_res[0]["lane_poly"][0], first_valid_res[1]["lane_poly"][0]))
            if curve_same_way and curve_diff > self.diverging_lanes_threshold:
                # the lanes diverge enough to be inspected for correction
                self.diverging_lanes_frames.append(self.frame_number)
            elif not curve_same_way and curve_diff > self.converging_lanes_threshold:
                # the lanes converge enough to be inspected for correction
                self.converging_lanes_frames.append(self.frame_number)
                better, worse = self.better_worse(first_valid_res[0], first_valid_res[1])

                # calculate average fitx for the frames before this frame
                fitx_sum = 0
                fitx_valids = 0
                for resf in self.results[worse["lr_ind"]]:
                    if resf["valid"]:
                        fitx_valids = fitx_valids + 1
                        fitx_sum = fitx_sum + resf["fitx"][-1]
                worse["fitx"] = better["fitx"] - better["fitx"][-1] + fitx_sum / fitx_valids
                # we mark it
                worse["converged_worse"] = True
                better["converged_better"] = True


            for res in first_valid_res:
                stage["find_lane"] = self.visualize_result(res, stage["find_lane"])
        else:
            # one valid lane
            self.one_lane_frames.append(self.frame_number)
            for res in first_valid_res:
                if res != False:
                    stage["find_lane"] = self.visualize_result(res, stage["find_lane"])

        stage["inverse_perspective_transform"] = self.inverse_perspective_transform(stage["find_lane"])

        stage["final_result"] = stage['undistorted'].copy()
        stage["final_result"][stage["inverse_perspective_transform"] > 0] = 0
        self.put_curvature(stage["final_result"], first_valid_res)
        stage["final_result"] = stage["final_result"] + stage["inverse_perspective_transform"]

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
        return stage["final_result"]


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

    # this is the trapezoid to later use for perspective transform
    # point[1] and point[2] are projected to y = 0 and the x of point[0] and point[3]
    pipeline_config["trapezoid"] = measured_trapezoid

    pipeline_config["region_of_interest_offsets"] = ((100, 0), (30, 30))
    # how many results to keep
    pipeline_config["remember_results"] = 5

    # when to expire results
    pipeline_config["expire_results"] = 25

    # how much confidence in a result which will expire next frame
    pipeline_config["expire_confidence"] = 0

    # how many pixels in a perfectly identified lane
    pipeline_config["max_detection"] = 20000

    # how many pixels before even trying to fit a polynomial
    pipeline_config["min_detection"] = 100

    # how high must the confidence of a fitted polygon be to go from naive finding to poly finding
    pipeline_config["search_poly_threshold"] = 0.2

    pipeline_config["converging_lanes_threshold"] = 0.001
    pipeline_config["diverging_lanes_threshold"] = 0.001

    pipeline_config["naive_windows"] = 12
    pipeline_config["naive_margin"] = 160
    pipeline_config["naive_minpix"] = 40
    pipeline_config["naive_window_color"] = ((0, 255, 0), (0, 255, 0))
    pipeline_config["naive_side_color"] = ((0, 255, 0), (255, 0, 0))
    pipeline_config["naive_polyline_color"] = ((255, 255, 0), (255, 255, 0))

    pipeline_config["poly_margin"] = 80
    pipeline_config["poly_search_color"] = ((30, 151, 227), (30, 151, 227))
    pipeline_config["poly_side_color"] = ((255, 0, 230), (255, 23, 42))
    pipeline_config["poly_polyline_color"] = ((255, 111, 0), (255, 111, 0))

    pipeline_config["curvature_font"] = cv2.FONT_HERSHEY_SIMPLEX
    pipeline_config["curvature_font_scale"] = 1

    # Define conversions in x and y from pixels space to meters
    pipeline_config["curvature_ym_per_pix"] = 30 / 720  # meters per pixel in y dimension
    pipeline_config["curvature_xm_per_pix"] = 3.7 / 700  # meters per pixel in x dimension
    pipeline_config["output"] = {}
    pipeline_config["output"]["image"] = {}
    # possible stages are: ['undistorted','bin_thresh']
    pipeline_config["output"]["image"]["stages"] = ["undistorted", "find_lane", "perspective_transform", "find_lane"]
    pipeline_config["output"]["video"] = {}
    # this only affects exported single frames, see the last tuple of each videopath
    pipeline_config["output"]["video"]["stages"] = ["undistorted", "bin_thresh", "perspective_transform", "find_lane"]

    imgpaths = []
    for p in glob.glob("test_images/*"):
        imgpaths.append((p, os.path.join("test_images_output", os.path.basename(p))))

    # pipeline_config["image_paths"] = imgpaths
    # (input, output, (start,stop),(export_frames)
    pipeline_config["video_paths"] = [
        ("project_video.mp4", "output_videos/project_video_out.mp4", False, False)]
    return pipeline_config


pl = Pipeline(configure())
pl.start()
print(f"invalid_lane_frames: {pl.invalid_lane_frames}")
print(f"diverging_lane_frames: {pl.diverging_lanes_frames}")
print(f"converging_lane_frames: {pl.converging_lanes_frames}")
print(f"no_lane_frames: {pl.no_lane_frames}")
print(f"one_lane_frames: {pl.one_lane_frames}")
