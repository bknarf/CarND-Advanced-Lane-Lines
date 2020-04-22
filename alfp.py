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

        self.trapezoid = np.array(config["perspective"]["trapezoid"])

        src = np.float32(self.trapezoid - np.array([self.trapezoid[2][0],self.trapezoid[0][1]]))

        trapezoid_bottom_width = self.trapezoid[3][0] - self.trapezoid[2][0]
        warp_height = self.trapezoid[2][1] - self.trapezoid[0][1]
        dst = np.float32([[0, 0], [trapezoid_bottom_width, 0],
                          [0, warp_height*2], [trapezoid_bottom_width, warp_height*2]])

        self.transform_matrix = cv2.getPerspectiveTransform(src, dst)
        self.transform_roi = (self.trapezoid[2][0], self.trapezoid[3][0],
                              self.trapezoid[0][1], self.trapezoid[2][1])

        if (("image_paths" in config) and config["image_paths"]):
            # pipeline is in self-controlled image mode
            # each entry in image_paths is a tuple (in,out) of filepaths
            self.image_paths = config["image_paths"]
        else:
            self.image_paths = False

        if (("video_paths" in config) and config["video_paths"]):
            # pipeline is in self-controlled video mode
            # each entry in videopaths is a tuple (in,out) of filepaths
            self.video_paths = config["video_paths"]
        else:
            self.video_paths = False

    def start(self):
        if (self.image_paths):
            for ip_in, ip_out in self.image_paths:
                # create output directory
                Path(Path(ip_out).parent).mkdir(parents=True, exist_ok=True)
                self.init_for_file(ip_in, ip_out, "image")
                img_in = mpimg.imread(ip_in)
                img_out = self.process(img_in)
                if len(img_out.shape) == 2:
                    mpimg.imsave(ip_out, img_out, None, None, "gray")
                else:
                    mpimg.imsave(ip_out, img_out)

        if (self.video_paths):
            for vp_in, vp_out in self.video_paths:
                # create output directory
                Path(Path(vp_out).parent).mkdir(parents=True, exist_ok=True)
                self.init_for_file(vp_in, vp_out, "video").process()
                in_clip = VideoFileClip(v_in)
                out_clip = in_clip.fl_image(self.process())
                out_clip.write_videofile(v_out, audio=False)

    def init_for_file(self, input_file, output_file, mode):
        self.input_file = input_file
        self.output_file = output_file
        self.frame_number = -1
        self.mode = mode
        self.stage_output = self.config["output"][mode]["stages"]
        return self

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


    def perspective_transform(self, img_1C, img_shape):
        #y comes first! also [ ... : ... , ... : ...] syntax
        roi = img_1C[self.transform_roi[2]:self.transform_roi[3],self.transform_roi[0]:self.transform_roi[1]]
        print(img_1C.shape)
        print(f"img_1C[{self.transform_roi[2]}:{self.transform_roi[3]}][{self.transform_roi[0]}:{self.transform_roi[1]}]")
        roi_shape = (self.transform_roi[1]-self.transform_roi[0],self.transform_roi[3]-self.transform_roi[2])

        return cv2.warpPerspective(roi, self.transform_matrix, (roi_shape[0],roi_shape[1]*2))


    def process(self, inimg):
        stage = {}
        stage['undistorted'] = self.camera.undistort(inimg)
        stage["bin_thresh"] = self.binary_threshold(stage['undistorted'])
        stage["perspective_transform"] = self.perspective_transform(stage['bin_thresh'], inimg.shape)

        """
        """

        if (self.mode == "image"):
            for st in self.stage_output:
                out_img = stage[st]
                if len(out_img.shape) == 2:
                    mpimg.imsave(os.path.join(os.path.splitext(self.output_file)[0]
                                              + st + os.path.splitext(self.output_file)[1]), out_img,
                                 None, None, "gray")
                else:
                    mpimg.imsave(os.path.join(os.path.splitext(self.output_file)[0]
                                              + st + os.path.splitext(self.output_file)[1]), out_img)

        return stage["perspective_transform"]

def configure():
    pipeline_config = {}
    pipeline_config["camera"] = {}
    pipeline_config["camera"]["calibration_paths"] = glob.glob("camera_cal/*")
    pipeline_config["camera"]["calibration_chessboard_path"] = False
    pipeline_config["camera"]["chessboard_dimensions"] = [9, 6]
    pipeline_config["camera"]["do_selftest"] = False
    # first number initial calibration, second number calibration failures on undistorted images
    pipeline_config["camera"]["chessboard_accepted_failures"] = [3, 4]



    xmax = 1280
    ymax = 720

    #inkscape coordinates, y from botton --> ymax - val
    measured_trapezoids = [
        [
            (586,ymax-270), (641,ymax-270),
            (258,ymax-50), (994, ymax-50)
        ],
        [
            (584, ymax-270), (647, ymax-270),
            (269, ymax-50), (998, ymax-50)
        ]
    ]

    avg_trz = np.array([(0,0),(0,0),(0,0),(0,0)])
    for mt in measured_trapezoids:
        avg_trz = np.add(avg_trz,mt)
    avg_trz = (avg_trz / len(measured_trapezoids)).astype(int)
    print(avg_trz)

    # left to right, top to bottom, lt, rt, lb, rb
    # left to right, top to bottom, lt, rt, lb, rb
    #pipeline_config["topview_trapezoid"] = (
    #   (xoffs_top, yoffs_top), (xmax - xoffs_top, yoffs_top), (xoffs_bottom, yoffs_bottom),
    #    (xmax - xoffs_bottom, yoffs_bottom))

    pipeline_config["perspective"] = {}
    pipeline_config["perspective"]["trapezoid"] = avg_trz
    #the trapezoid is the outer line of straight centered lanes, we need some space for curves and such
    pipeline_config["perspective"]["padding"] = (100,0)

    #when projecting the trapezoif upright, how high should the resulting rectangle be
    pipeline_config["perspective"]["height"] = avg_trz[2][1] - avg_trz[0][1]

    pipeline_config["output"] = {}
    pipeline_config["output"]["image"] = {}
    # possible stages are: ['undistorted','bin_thresh']
    pipeline_config["output"]["image"]["stages"] = ['undistorted']

    pipeline_config["video_paths"] = False

    imgpaths = []
    for p in glob.glob("test_images/*"):
        imgpaths.append((p, os.path.join("test_images_output", os.path.basename(p))))

    pipeline_config["image_paths"] = imgpaths
    return pipeline_config

pl = Pipeline(configure())
pl.start()

"""
c = Camera(**pipeline_config["camera"])
print(c)

if not os.path.exists("camera_cal_undist"):
        os.makedirs("camera_cal_undist")
for ip in pipeline_config["camera"]["calibration_paths"]:
    img = mpimg.imread(ip)
    dst = c.undistort(img)
    mpimg.imsave(os.path.join("camera_cal_undist", os.path.splitext(os.path.basename(ip))[0] + f".undist.jpg"), dst)
"""
