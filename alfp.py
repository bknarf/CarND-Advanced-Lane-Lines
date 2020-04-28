import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import os

from moviepy.editor import VideoFileClip
from pathlib import Path


class Camera:
    def __init__(self):

        # can be used to save images from calibration process

        self.calibration_chessboard_path = False
        self.chessboard_dimensions = [9, 6]

        # first number initial calibration, second number calibration failures on undistorted images
        self.chessboard_accepted_failures = [3, 4]
        self.calibration_paths = glob.glob("camera_cal/*")
        calimages = ((ip, mpimg.imread(ip)) for ip in self.calibration_paths)
        self.calibrated, self.camera_matrix, self.distort_coeff, self.chessboard_found, self.chessboard_notfound = Camera.calibrate(
            calimages, self.chessboard_dimensions, self.calibration_chessboard_path)
        self.selftest_result = (False, "not run, was not configured in constructor")
        self.do_selftest = False
        if self.do_selftest:
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


class LaneMarking():

    def __init__(self, poly, side, detected_x, detected_y, detection_count, frame_number, valid, searchpoly=[],
                 window_rectangles=[], averaged_frames=[]):
        self.poly = poly
        self.searchpoly = searchpoly
        self.side = side
        self.detection_count = detection_count
        self.frame_number = frame_number
        self.valid = valid
        self.detected_x = detected_x
        self.detected_y = detected_y
        self.window_rectangles = window_rectangles
        if not (valid) or detection_count == 0:
            self.quality = 0
        else:
            self.quality = LaneMarking.max_detection / self.detection_count
        self.averaged_frames = averaged_frames

    # how many pixels in a perfectly identified lane
    max_detection = 20000

    # how many pixels before even trying to fit a polynomial
    min_detection = 100
    searchpoly_margin = 80

    window_count = 12
    window_margin = 160
    window_recenter_minpix = 40

    searchcolor = ((43, 217, 254)[::-1], (254, 217, 43)[::-1])
    sidecolor = ((82, 170, 94)[::-1], (127, 85, 125)[::-1])
    linecolor = ((255, 0, 0)[::-1], (255, 0, 0)[::-1])
    lanewidth = 10
    lanearea_color = (220, 220, 220)[::-1]

    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 892  # meters per pixel in x dimension
    curvature_font = cv2.FONT_HERSHEY_SIMPLEX
    curvature_fontscale = 1

    @staticmethod
    def find_by_poly(side, frame_number, searchpoly, nonzerox, nonzeroy):

        margin = LaneMarking.searchpoly_margin

        lane_inds = ((nonzerox > (searchpoly[0] * (nonzeroy ** 2) + searchpoly[1] * nonzeroy +
                                  searchpoly[2] - margin)) & (
                             nonzerox < (searchpoly[0] * (nonzeroy ** 2) +
                                         searchpoly[1] * nonzeroy + searchpoly[
                                             2] + margin)))

        xs = nonzerox[lane_inds]
        ys = nonzeroy[lane_inds]

        detection_count = len(xs)

        if detection_count > LaneMarking.min_detection:
            # Fit new polynomials
            poly = np.polyfit(ys, xs, 2)
            valid = True
        else:
            poly = []
            valid = False

        return LaneMarking(poly, side, xs, ys, detection_count, frame_number, valid, searchpoly=searchpoly)

    @staticmethod
    def peaks(topdown, lr=(True, True)):
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
        return peaks

    @staticmethod
    def find_by_sliding_windows(side, frame_number, y_height, peaks, nonzerox, nonzeroy):

        nwindows = LaneMarking.window_count
        # Set the width of the windows +/- margin
        margin = LaneMarking.window_margin
        # Set minimum number of pixels found to recenter window
        minpix = LaneMarking.window_recenter_minpix

        window_height = np.int(y_height // nwindows)

        x_current = peaks[side]
        # Step through the windows one by one
        lane_inds = []
        window_rectangles = []
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = y_height - (window + 1) * window_height
            win_y_high = y_height - window * window_height
            win_x_low = x_current - margin
            win_x_high = x_current + margin
            window_rectangles.append(((win_x_low, win_y_low), (win_x_high, win_y_high)))

            # Identify the nonzero pixels in x and y within the window #
            good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                         (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]

            # Append these indices to the lists
            lane_inds.append(good_inds)

            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_inds) > minpix:
                x_current = np.int(np.mean(nonzerox[good_inds]))

        lane_inds = np.concatenate(lane_inds)

        # Extract left and right line pixel positions
        xs = nonzerox[lane_inds]
        ys = nonzeroy[lane_inds]
        detected = len(xs)

        if detected > LaneMarking.min_detection:
            # Fit a second order polynomial
            lane_poly = np.polyfit(ys, xs, 2)
            valid = True
        else:
            lane_poly = []
            valid = False

        return LaneMarking(lane_poly, side, xs, ys, detected, frame_number, valid,
                           window_rectangles=window_rectangles)

    @staticmethod
    def average_valid_copy(side, lanes):
        fn = []
        avgpoly = [[], [], []]
        for l in lanes:
            if l.valid:
                fn.append(l.frame_number)
                for i in [0, 1, 2]:
                    avgpoly[i].append(l.poly[i])
        for i in [0, 1, 2]:
            avgpoly[i] = np.average(avgpoly[i])

        return LaneMarking(avgpoly, side, [], [], 0, -1, True, averaged_frames=fn)

    @staticmethod
    def intersect_in_range(lane_a, lane_b, y):
        y.sort()
        yr = np.linspace(y[0], y[1] - 1, y[1])
        xsa = lane_a.fitted_x(yr)
        xsb = lane_b.fitted_x(yr)
        b_larger = xsb[0] > xsa[0]
        for xa, xb in zip(xsa, xsb):
            if xa == xb:
                return True
            elif (xb < xa) == b_larger:
                return True
        return False

    def fitted_x(self, y):
        if len(self.poly) == 3:
            return self.poly[0] * y ** 2 + self.poly[1] * y + self.poly[2]
        return y

    def searchpoly_x(self, y):
        if len(self.searchpoly) == 3:
            return self.searchpoly[0] * y ** 2 + self.searchpoly[1] * y + self.searchpoly[2]
        return y

    def put_visuals(self, rgb_vis, search_area=True, found_pixels=True, poly=True):
        ploty = np.linspace(0, rgb_vis.shape[0] - 1, rgb_vis.shape[0])

        if search_area:
            if len(self.searchpoly) == 3:
                spx = self.searchpoly_x(ploty)
                # Generate a polygon to illustrate the search window area
                # And recast the x and y points into usable format for cv2.fillPoly()
                line_window1 = np.array([np.transpose(np.vstack([spx - LaneMarking.searchpoly_margin, ploty]))])
                line_window2 = np.array([np.flipud(np.transpose(np.vstack([spx + LaneMarking.searchpoly_margin,
                                                                           ploty])))])
                line_pts = np.hstack((line_window1, line_window2))

                # Draw the lane onto the warped blank image
                cv2.fillPoly(rgb_vis, np.int_([line_pts]), LaneMarking.searchcolor[self.side])

            elif len(self.window_rectangles) > 0:
                for bt in self.window_rectangles:
                    cv2.rectangle(rgb_vis, bt[0], bt[1], LaneMarking.searchcolor[self.side], 5)

        if found_pixels:
            rgb_vis[self.detected_y, self.detected_x] = LaneMarking.sidecolor[self.side]

        if poly and self.valid:
            fitx = self.fitted_x(ploty)
            # Plot the polynomial lines onto the image
            line_window1 = np.array([np.transpose(np.vstack([fitx - LaneMarking.lanewidth, ploty]))])
            line_window2 = np.array([np.flipud(np.transpose(np.vstack([fitx + LaneMarking.lanewidth,
                                                                       ploty])))])

            line_pts = np.hstack((line_window1, line_window2))
            cv2.fillPoly(rgb_vis, np.int_([line_pts]), LaneMarking.linecolor[self.side])

        return rgb_vis

    def put_lanearea(self, rgb_vis, other_lane):

        lr_sorter = [(self.side, self), (other_lane.side, other_lane)]
        lr_sorter.sort()
        if lr_sorter[0][1].valid and lr_sorter[1][1].valid:
            ploty = np.linspace(0, rgb_vis.shape[0] - 1, rgb_vis.shape[0])
            lx = lr_sorter[0][1].fitted_x(ploty)
            rx = lr_sorter[1][1].fitted_x(ploty)
            # Generate a polygon to illustrate the search window area
            # And recast the x and y points into usable format for cv2.fillPoly()
            line_window1 = np.array([np.transpose(np.vstack([lx, ploty]))])
            line_window2 = np.array([np.flipud(np.transpose(np.vstack([rx,
                                                                       ploty])))])
            line_pts = np.hstack((line_window1, line_window2))

            # Draw the lane onto the warped blank image
            cv2.fillPoly(rgb_vis, np.int_(line_pts), LaneMarking.lanearea_color)

    def put_curvatures_and_position(self, inimg, other_lane=False):

        # left curvature, right curvature, position in lane
        curvature_text_centers = [inimg.shape[1] // 4, (inimg.shape[1] // 4) * 3, ]
        position_text_center = inimg.shape[1] // 2

        lr_sorter = [(self.side, self), (other_lane.side, other_lane)]
        lr_sorter.sort()
        if lr_sorter[0][1].valid and lr_sorter[1][1].valid:
            l = [lr_sorter[0][1], lr_sorter[1][1]]
            c = [0, 0]
            x = [0, 0]
            for i in [0, 1]:
                c[i] = ((1 + (2 * l[i].poly[0] * (inimg.shape[0] - 1) * self.ym_per_pix + l[i].poly[
                    1]) ** 2) ** 1.5) / np.absolute(2 * l[i].poly[0])
                x[i] = l[i].fitted_x(inimg.shape[0] - 1)

            # put the curvatures
            for center, curv in zip(curvature_text_centers, c):
                if curv:
                    txt = "{:.1f}km".format(round(curv / 100) / 10)
                    # get boundary of this text
                    textsize = cv2.getTextSize(txt, LaneMarking.curvature_font, LaneMarking.curvature_fontscale, 2)[
                        0]
                    cv2.putText(inimg, txt, (center - textsize[0] // 2, textsize[1] * 2),
                                LaneMarking.curvature_font,
                                LaneMarking.curvature_fontscale, (0, 0, 0), 2)

            x_middle = (x[0] + x[1]) / 2

            if x_middle > inimg.shape[1] / 2:
                postext = "{:.2f}m left of center".format(
                    (x_middle - inimg.shape[1] / 2) * LaneMarking.xm_per_pix)
            else:
                postext = "{:.2f}m right of center".format(
                    (inimg.shape[1] / 2 - x_middle) * LaneMarking.xm_per_pix)

            # get boundary of this text
            postextsize = cv2.getTextSize(postext, LaneMarking.curvature_font, LaneMarking.curvature_fontscale, 2)[
                0]
            cv2.putText(inimg, postext, (position_text_center - postextsize[0] // 2, postextsize[1] * 2),
                        LaneMarking.curvature_font,
                        LaneMarking.curvature_fontscale, (0, 0, 0), 2)

    def quality_ok_for_find_by_poly(self):
        return self.valid and self.quality > 0.35


class Pipeline:
    def __init__(self):

        self.poly_summary = []
        self.intersecting_lanes_frames = []

        self.camera = Camera()
        self.frame_number = False

        self.lanemarkings = False

        # gimp coordinates, clockwise, bottom left first, taken from thresholded image
        # this is the trapezoid to later use for perspective transform
        # point[1] and point[2] are projected to y = 0 and the x of point[0] and point[3]
        self.trapezoid = [(194, 719), (581, 460), (699, 460), (1086, 719)]
        # set up the perspective transform
        src = np.float32(self.trapezoid)

        dst = np.float32([self.trapezoid[0], [self.trapezoid[0][0], 0],
                          [self.trapezoid[3][0], 0], self.trapezoid[3]])

        self.transform_matrix = cv2.getPerspectiveTransform(src, dst)

        self.trapezoid_color = [161, 16, 112]

        self.region_of_interest_offsets = ((100, 0), (30, 30))

        self.lanemarkings_memory = 10

        self.intersecting_lanes_window_margin_top = 100
        self.intersecting_lanes_window_margin_bottom = 0

        self.output = {}

        self.image_export_stages = ['distorted', 'undistorted', 'trapezoid', 'bin_thresh', 'perspective_transform',
                                    'find_lane', 'inverse_perspective_transform', 'all_vis', 'lanearea']
        # this only affects exported single frames, see the last tuple of each videopath
        self.video_export_stages = ['distorted', 'undistorted', 'trapezoid', 'bin_thresh', 'perspective_transform',
                                    'find_lane', 'inverse_perspective_transform', 'all_vis', 'lanearea']
        self.process_result = 'lanearea'
        self.image_paths = []
        for p in glob.glob("test_images/*"):
            self.image_paths.append((p, os.path.join("output_images", os.path.basename(p))))

        # pipeline_config["image_paths"] = imgpaths
        # (input, output, (start,stop),(export_frames)
        self.video_paths = [
            ("project_video.mp4", "output_videos/project_video_out.mp4", False, (100, 500, 1047, 1500, 2000)),
            ("challenge_video.mp4", "output_videos/challenge_video_out.mp4", False, (100, 500, 1047, 1500, 2000)),
            ("harder_challenge_video.mp4", "output_videos/harder_challenge_video.mp4", False,
             (100, 500, 1047, 1500, 2000))
        ]

        self.exportframes = []



    def start(self):

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

        self.lanemarkings = [[], []]

        self.intersecting_lanes_frames = []

        self.input_file = input_file
        self.output_file = output_file
        self.frame_number = 0
        self.mode = mode
        self.exportframes = exportframes
        return self

    def put_trapezoid(self, img_RGB):
        color = self.trapezoid_color
        thickness = 10
        cv2.line(img_RGB, tuple(self.trapezoid[0]),
                 tuple(self.trapezoid[1]), color, thickness)
        cv2.line(img_RGB, tuple(self.trapezoid[1]),
                 tuple(self.trapezoid[2]), color, thickness)
        cv2.line(img_RGB, tuple(self.trapezoid[2]),
                 tuple(self.trapezoid[3]), color, thickness)
        cv2.line(img_RGB, tuple(self.trapezoid[3]),
                 tuple(self.trapezoid[0]), color, thickness)

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

        B = img_RGB[:, :, 2]

        HLS = cv2.cvtColor(img_RGB, cv2.COLOR_BGR2HLS)
        H = HLS[:, :, 0]
        L = HLS[:, :, 1]
        S = HLS[:, :, 2]

        Sthresh = (100, 255)
        Lthresh = (200, 255)
        Hthresh = (0, 100)
        BThresh = (220, 255)
        binary = np.zeros_like(S)
        binary[(((S > Sthresh[0]) & (S <= Sthresh[1])) | (L > Lthresh[0]) & (L <= Lthresh[1])) & (
                (H > Hthresh[0]) & (H <= Hthresh[1])) | (B > BThresh[0]) & (B < BThresh[1])] = 255

        mt = self.trapezoid
        offsets = self.region_of_interest_offsets

        to_mask = ((mt[0][0] - offsets[0][0], mt[0][1] + offsets[0][1]),
                   (mt[1][0] - offsets[1][0], mt[1][1] - offsets[1][1]),
                   (mt[2][0] + offsets[1][0], mt[2][1] - offsets[1][1]),
                   (mt[3][0] + offsets[0][0], mt[3][1] + offsets[0][1]))

        return self.region_of_interest(binary, to_mask)

    def perspective_transform(self, img_in):
        return cv2.warpPerspective(img_in.copy(), self.transform_matrix, (img_in.shape[1], img_in.shape[0]))

    def inverse_perspective_transform(self, img_in):
        return cv2.warpPerspective(img_in.copy(), self.transform_matrix, (img_in.shape[1], img_in.shape[0]),
                                   flags=cv2.WARP_INVERSE_MAP)

    def visualize_lanearea(self, dist, undist, lane_l, lane_r):

        warp_zero = np.zeros_like(dist).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        lane_l.put_lanearea(color_warp, lane_r)

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = self.inverse_perspective_transform(color_warp)
        # Combine the result with the original image
        return cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    def process(self, inimg):
        inimg = cv2.cvtColor(inimg, cv2.COLOR_BGR2RGB)
        stage = {}
        stage['distorted'] = inimg
        stage['undistorted'] = self.camera.undistort(inimg)
        stage['trapezoid'] = np.copy(stage['undistorted'])
        self.put_trapezoid(stage['trapezoid'])
        stage["bin_thresh"] = self.binary_threshold(stage['undistorted'])
        stage["perspective_transform"] = self.perspective_transform(stage['bin_thresh'])

        # decide for each side which algorithm to use for this frame

        nonzero = stage["perspective_transform"].nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        peaks = False
        for i in [0, 1]:
            if len(self.lanemarkings[i]) > 0 and self.lanemarkings[i][0].quality_ok_for_find_by_poly():

                self.lanemarkings[i] = [LaneMarking.find_by_poly(i, self.frame_number, self.lanemarkings[i][0].poly,
                                                                 nonzerox,
                                                                 nonzeroy)] + self.lanemarkings[i]
            else:
                if peaks == False:
                    peaks = LaneMarking.peaks(stage["perspective_transform"])
                self.lanemarkings[i] = [LaneMarking.find_by_sliding_windows(i, self.frame_number,
                                                                            stage["perspective_transform"].shape[0],
                                                                            peaks,
                                                                            nonzerox, nonzeroy)] + self.lanemarkings[i]

        stage["find_lane"] = cv2.cvtColor(stage["perspective_transform"], cv2.COLOR_GRAY2RGB)

        poly_source = [False, False]
        for i in [0, 1]:
            al = LaneMarking.average_valid_copy(i, self.lanemarkings[i])
            if len(al.averaged_frames) > 1:
                self.lanemarkings[i][0].put_visuals(stage["find_lane"], search_area=True, found_pixels=True, poly=False)
                al.put_visuals(stage["find_lane"], search_area=False, found_pixels=False, poly=True)
                poly_source[i] = al
            else:
                self.lanemarkings[i][0].put_visuals(stage["find_lane"], search_area=True, found_pixels=True, poly=True)
                poly_source[i] = self.lanemarkings[i][0]

        if LaneMarking.intersect_in_range(poly_source[0], poly_source[1], [-self.intersecting_lanes_window_margin_top,
                                                                           stage["find_lane"].shape[
                                                                               0] + self.intersecting_lanes_window_margin_bottom]):
            self.intersecting_lanes_frames.append(self.frame_number)
            for i in [0, 1]:
                self.lanemarkings[i][0].valid = False

        stage["inverse_perspective_transform"] = self.inverse_perspective_transform(stage["find_lane"])

        stage["all_vis"] = stage['undistorted'].copy()
        stage["all_vis"][stage["inverse_perspective_transform"] > 0] = 0
        stage["all_vis"] = stage["all_vis"] + stage["inverse_perspective_transform"]
        stage["all_vis"] = self.visualize_lanearea(stage["perspective_transform"], stage["all_vis"],
                                                   poly_source[0],
                                                   poly_source[1])
        poly_source[0].put_curvatures_and_position(stage["all_vis"], poly_source[1])
        stage["lanearea"] = self.visualize_lanearea(stage["perspective_transform"], stage["undistorted"],
                                                    poly_source[0],
                                                    poly_source[1])
        poly_source[0].put_curvatures_and_position(stage["lanearea"], poly_source[1])

        if (self.mode == "image"):
            for st in self.image_export_stages:
                out_img = stage[st]
                if len(out_img.shape) == 2:
                    mpimg.imsave(os.path.join(os.path.splitext(self.output_file)[0]
                                              + '.' + st + os.path.splitext(self.output_file)[1]), out_img,
                                 None, None, "gray")
                else:
                    mpimg.imsave(os.path.join(os.path.splitext(self.output_file)[0]
                                              + '.' + st + os.path.splitext(self.output_file)[1]),
                                 cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB))
        elif (self.exportframes and self.frame_number in self.exportframes):
            for st in self.video_export_stages:
                out_img = stage[st]
                if len(out_img.shape) == 2:
                    mpimg.imsave(os.path.join(os.path.splitext(self.output_file)[0]
                                              + '.' + str(self.frame_number) + '.' + st + ".jpg"), out_img,
                                 None, None, "gray")
                else:
                    mpimg.imsave(os.path.join(os.path.splitext(self.output_file)[0]
                                              + '.' + str(self.frame_number) + '.' + st + ".jpg"),
                                 cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB))

        self.frame_number = self.frame_number + 1

        for i in [0, 1]:
            self.lanemarkings[i] = self.lanemarkings[i][0:self.lanemarkings_memory - 1]

        return cv2.cvtColor(stage[self.process_result], cv2.COLOR_BGR2RGB)


Camera().undistort()
pl = Pipeline()
pl.start()
