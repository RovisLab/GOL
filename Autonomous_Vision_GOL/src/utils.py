import cv2
import numpy as np
import ctypes
from math import sqrt, log

colors_distributions = {
    "black_b_mean": 40,
    "black_b_dev": 28,
    "black_g_mean": 39,
    "black_g_dev": 26,
    "black_r_mean": 39,
    "black_r_dev": 26,
    "white_b_mean": 151,
    "white_b_dev": 48,
    "white_g_mean": 154,
    "white_g_dev": 50,
    "white_r_mean": 148,
    "white_r_dev": 54,
    "red_b_mean": 56,
    "red_b_dev": 27,
    "red_g_mean": 58,
    "red_g_dev": 30,
    "red_r_mean": 126,
    "red_r_dev": 55
}


def error_pop_up():
    ctypes.windll.user32.MessageBoxW(0, "Invalid coefficient", "ERROR", 1)


def transform_perspective(img, origin_points, warped_points, img_width, img_height):
    from Artificial_Samples_Generator import ASG
    perspective_transform = cv2.getPerspectiveTransform(origin_points, warped_points)
    pts = cv2.perspectiveTransform(np.float32([[34, 7], [61, 89]]).reshape(-1, 1, 2), perspective_transform)
    # print(pts)
    ASG.annot_top_left = [int(pts[0, 0, 0]), int(pts[0, 0, 1])]
    ASG.annot_bot_right = [int(pts[1, 0, 0]), int(pts[1, 0, 1])]
    return cv2.warpPerspective(img, perspective_transform, (img_width, img_height))


def get_img_properties(img):
    img_height = img.shape[0]
    img_width = img.shape[1]
    if len(img) > 2:
        img_channels = img.shape[2]
    else:
        img_channels = 1
    return img_height, img_width, img_channels


def remap_image(src_img, x_offset, y_offset, src_img_height, src_img_width):
    map_x = np.zeros((src_img_height, src_img_width), np.float32)
    map_y = np.zeros((src_img_height, src_img_width), np.float32)
    for j in range(src_img_height):
        for i in range(src_img_width):
            if ((i - x_offset >= 0) and (j - y_offset >= 0)) and (
                    (i - x_offset < src_img_width) and (j - y_offset) < src_img_height):
                map_x[j, i] = i - x_offset
                map_y[j, i] = j - y_offset
    src_img = cv2.remap(src_img, map_x, map_y, cv2.INTER_LINEAR)
    return src_img


def bgr_channels(b_val, g_val, r_val):
    color = np.empty(3)
    if b_val < 0:
        b_val = 0
    elif b_val > 255:
        b_val = 255
    if g_val < 0:
        g_val = 0
    elif g_val > 255:
        g_val = 255
    if r_val < 0:
        r_val = 0
    elif r_val > 255:
        r_val = 255
    color[0] = b_val
    color[1] = g_val
    color[2] = r_val
    return color


def change_color(dst_img, color, row, col):
    dst_img[row, col, 0] = color[0]
    dst_img[row, col, 1] = color[1]
    dst_img[row, col, 2] = color[2]


def gaussian_random():
    next_gaussian = 0
    saved_gaussian_value = 0
    fac = 0
    rsq = 0
    v1 = 0
    v2 = 0
    if next_gaussian == 0:
        while True:
            v1 = 2 * np.random.uniform() - 1
            v2 = 2 * np.random.uniform() - 1
            rsq = v1 * v1 + v2 * v2
            if rsq >=1 or rsq == 0:
                break
        fac = sqrt(-2 * log(rsq) / rsq)
        saved_gaussian_value = v1 * fac
        next_gaussian = 1
        return v2 * fac
    else:
        next_gaussian = 0
        return saved_gaussian_value
