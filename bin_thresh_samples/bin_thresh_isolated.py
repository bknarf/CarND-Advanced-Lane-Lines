import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import os

from moviepy.editor import VideoFileClip
from pathlib import Path


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
        print(f"min: {np.amin(absgraddir)} max:{np.amax(absgraddir)}")
        binary_output = np.zeros_like(absgraddir)
        binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

        # Return the binary image
        return binary_output


def region_of_interest(img, vertices):
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

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def binary_threshold(img_RGB):
    # measured_trapezoid
    mt = [
        (194, 719), (581, 460),
        (702, 460), (1115, 719)
    ]

    offsets = ((100, 0), (10, 20))

    to_mask = ((mt[0][0] - offsets[0][0], mt[0][1] + offsets[0][1]),
               (mt[1][0] - offsets[1][0], mt[1][1] - offsets[1][1]),
               (mt[2][0] + offsets[1][0], mt[2][1] - offsets[1][1]),
               (mt[3][0] + offsets[0][0], mt[3][1] + offsets[0][1]))

    HLS = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2HLS)
    gray = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2GRAY)
    R = img_RGB[:, :, 0]
    G = img_RGB[:, :, 1]
    B = img_RGB[:, :, 2]
    H = HLS[:, :, 0]
    L = HLS[:, :, 1]
    S = HLS[:, :, 2]

    sobelR = Sobel.mag_thresh(R, sobel_kernel=3, mag_thresh=(80, 255))
    sobelH = Sobel.dir_threshold(H, sobel_kernel=5, thresh=((np.pi / 2) * (1 / 3), (np.pi / 2) * (2 / 3)))

    Sthresh = (100, 255)
    Lthresh = (200, 255)
    Hthresh = (0, 100)
    BThresh = (220,255)
    binary = np.zeros_like(S)
    binary[(((S > Sthresh[0]) & (S <= Sthresh[1])) | (L > Lthresh[0]) & (L <= Lthresh[1])) & (
                (H > Hthresh[0]) & (H <= Hthresh[1])) | (B>BThresh[0]) & (B<BThresh[1])] = 255
    #sobelMagLS = Sobel.mag_thresh(binaryLS, sobel_kernel=3, mag_thresh=(80, 255))
    #sobelDirLS = Sobel.dir_threshold(binaryLS, sobel_kernel=5, thresh=((np.pi / 2) * (1 / 3), (np.pi / 2) * (2 / 3)))

    return region_of_interest(binary, np.array([[to_mask]], dtype=np.int32))


Path("bin_thresh_out/").mkdir(parents=True, exist_ok=True)
for fn in glob.glob("*.undistorted.jpg"):
    inimg = mpimg.imread(fn)
    outimg = binary_threshold(inimg)
    if len(outimg.shape) == 2:
        mpimg.imsave("bin_thresh_out/" + fn, outimg, None, None, "gray")
    else:
        mpimg.imsave("bin_thresh_out/" + fn, outimg)
