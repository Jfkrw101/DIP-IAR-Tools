import cv2
import numpy as np


def kmeans(input_img, k):

    # -> Record Transformation
    y, x, c = input_img.shape
    input_rec = input_img.reshape(y*x, c)
    input_rec = input_rec.astype(np.float32)

    ### -> K-means Clustering
    criteria = (cv2.TERM_CRITERIA_MAX_ITER +
    cv2.TERM_CRITERIA_EPS, 100, 0.2) \
    _, label_rec, (centers) = cv2.kmeans(input_rec,k, None, criteria, 10,cv2.KMEANS_RANDOM_CENTERS)
    # -> Assign Label with Center Color
    centers = centers.astype(np.uint8)
    output_rec = centers[label_rec.flatten()]

    # -> Reshape into image
    output_img = output_rec.reshape((y, x, c))

    return output_img, centers