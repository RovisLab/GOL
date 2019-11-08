import cv2
from skimage import feature
import numpy as np
import math


def lbp_feature(image, num_points, radius, resize=(0, 0)):
    """
    Compute the Local Binary Pattern representation of the image and build the histogram of patterns
    :param image: image
    :param num_points: number of points for LBP
    :param radius: radius of LBP
    :param resize: resize images to specified value
    :return: histogram of LBP features
    """
    if resize[0] != 0 and resize[1] != 0:
        processed_image = cv2.resize(image, resize)
    else:
        processed_image = image
    if len(image.shape) == 3:
        num_chn = image.shape[2]
        if num_chn == 3:
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
        elif num_chn == 4:
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGRA2GRAY)

    lbp = feature.local_binary_pattern(processed_image, num_points, radius, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(),
                             bins=np.arange(0, num_points + 3),
                             range=(0, num_points + 2))
    hist = hist.astype("float")
    hist /= hist.sum()
    return hist


def bhatt_coeff(u, v):
    """
    B_C = sum(sqrt(u,v))
    :param u: first vector
    :param v: second vector
    :return: Bhattacharyya Distance Coefficient
    """
    if len(u) != len(v):
        raise ValueError
    return sum(math.sqrt(u_i * v_i) for u_i, v_i in zip(u, v))


def bhatt(u, v):
    """
    B_D = -ln(B_C)
    :param u: first vector
    :param v: second vector
    :return: Bhattacharyya Distance
    """
    if len(u) != len(v):
        raise ValueError
    # return -math.log(bhatt_coeff(u, v), math.e)
    return bhatt_coeff(u, v)