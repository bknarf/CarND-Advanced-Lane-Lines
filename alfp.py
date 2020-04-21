import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import os


class Camera:
    def __init__(self, calibration_paths, chessboard_dimensions, chessboard_accepted_failures, do_selftest, calibration_chessboard_path):
        self.calibration_paths = calibration_paths
        self.chessboard_accepted_failures = chessboard_accepted_failures
        self.chessboard_dimensions = chessboard_dimensions
        self.calibration_chessboard_path=calibration_chessboard_path
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

        if calibration_chessboard_path :
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
                    icbc = cv2.drawChessboardCorners(calimg, (chessboard_dimensions[0], chessboard_dimensions[1]), corners, ret)
                    mpimg.imsave(
                        os.path.join(calibration_chessboard_path, os.path.splitext(os.path.basename(ip))[0] + f".corners.jpg"),
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
        print("selftest")
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
                print(f"{self.distort_coeff} --> {st_distort_coeff}")
                if len(st_chessboard_notfound) > self.chessboard_accepted_failures[1]:
                    self.selftest_result = (
                        False,
                        f"Chessboard detection failed on {len(st_chessboard_notfound)} "
                        + f"undistorted calibration images. "
                        + f"Accepted failures {self.chessboard_accepted_failures [1]} "
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


pipeline_config = {}
pipeline_config["camera"] = {}
pipeline_config["camera"]["calibration_paths"] = glob.glob("camera_cal/*")
pipeline_config["camera"]["calibration_chessboard_path"] = "camera_cal_corners"
pipeline_config["camera"]["chessboard_dimensions"] = [9, 6]
pipeline_config["camera"]["do_selftest"] = True
#first number initial calibration, second number calibration failures on undistorted images
pipeline_config["camera"]["chessboard_accepted_failures"] = [3,4]


c = Camera(**pipeline_config["camera"])
print(c)

if not os.path.exists("camera_cal_undist"):
        os.makedirs("camera_cal_undist")
for ip in pipeline_config["camera"]["calibration_paths"]:
    img = mpimg.imread(ip)
    dst = c.undistort(img)
    mpimg.imsave(os.path.join("camera_cal_undist", os.path.splitext(os.path.basename(ip))[0] + f".undist.jpg"), dst)
