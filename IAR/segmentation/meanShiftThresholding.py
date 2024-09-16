import cv2
import numpy as np


def meanShiftThresholding(input_img, thresh_delta=0.01):

    # - Initial Thresh Point
    thresh_val = input_img.mean()

    while True:
        # - Thresholding
        _, thresh_img = cv2.threshold(input_img, thresh_val,
        255, cv2.THRESH_BINARY)
        # - Mean of Intensity (Pass)
        y_pass_idx, x_pass_idx = np.where(thresh_img == 255)
        pass_mean = input_img[y_pass_idx,x_pass_idx].mean()

        # - Mean of Intensity (Not Pass)
        y_npass_idx, x_npass_idx = np.where(thresh_img == 0)
        npass_mean = input_img[y_npass_idx,x_npass_idx].mean()

        # - Average of both Mean
        new_thresh_val = (pass_mean + npass_mean) / 2

        # - Different of Thresh Point |New - Old|
        if abs(new_thresh_val-thresh_val) < thresh_delta:
            break
        else:
            thres_val = new_thresh_val

        return thresh_img, thres_val