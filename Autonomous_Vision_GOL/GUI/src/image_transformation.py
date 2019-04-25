from utils import *
from random import randint
from math import sqrt


def perspective_transform_horizontal(src_img, coefficient):
    if coefficient == 0:
        return src_img
    elif coefficient < -1 or coefficient > 1:
        error_pop_up()
        return src_img
    src_img_height, src_img_width, src_img_channels = get_img_properties(src_img)
    origin_points = np.float32(
        [[0, 0], [src_img_width - 1, 0], [0, src_img_height - 1], [src_img_width - 1, src_img_height - 1]])
    if coefficient > 0:
        warped_points = np.float32([[coefficient * src_img_width / 2, coefficient * src_img_height / 2],
                                    [src_img_width - coefficient * src_img_width / 2,
                                     -(coefficient * src_img_height / 2)],
                                    [coefficient * src_img_width / 2,
                                     src_img_height - coefficient * src_img_height / 2],
                                    [src_img_width - coefficient * src_img_width / 2,
                                     src_img_height + coefficient * src_img_height / 2]])
        dst_img = transform_perspective(src_img, origin_points, warped_points, src_img_width, src_img_height)
    else:
        warped_points = np.float32([[abs(coefficient) * src_img_width / 2, -(abs(coefficient) * src_img_height / 2)],
                                    [src_img_width - (abs(coefficient) * src_img_width / 2),
                                     abs(coefficient) * src_img_height / 2],
                                    [abs(coefficient) * src_img_width / 2,
                                     src_img_height + abs(coefficient) * src_img_height / 2],
                                    [src_img_width - (abs(coefficient) * src_img_width / 2),
                                     src_img_height - (abs(coefficient) * src_img_height / 2)]])
        dst_img = transform_perspective(src_img, origin_points, warped_points, src_img_width, src_img_height)
    return dst_img


def perspective_transform_vertical(src_img, coefficient):
    if coefficient == 0:
        return src_img
    elif coefficient < -1 or coefficient > 1:
        error_pop_up()
        return src_img
    src_img_height, src_img_width, src_img_channels = get_img_properties(src_img)
    origin_points = np.float32(
        [[0, 0], [src_img_width - 1, 0], [0, src_img_height - 1], [src_img_width - 1, src_img_height - 1]])
    if coefficient > 0:
        warped_points = np.float32([[-(coefficient * src_img_width / 2), coefficient * src_img_height / 2],
                                    [src_img_width + coefficient * src_img_width / 2, coefficient * src_img_height / 2],
                                    [coefficient * src_img_width / 2,
                                     src_img_height - coefficient * src_img_height / 2],
                                    [src_img_width - coefficient * src_img_width / 2,
                                     src_img_height - coefficient * src_img_height / 2]])
        dst_img = transform_perspective(src_img, origin_points, warped_points, src_img_width, src_img_height)
    else:
        warped_points = np.float32([[abs(coefficient) * src_img_width / 2, abs(coefficient) * src_img_height / 2],
                                    [src_img_width - abs(coefficient) * src_img_width / 2,
                                     abs(coefficient) * src_img_height / 2],
                                    [-(abs(coefficient) * src_img_width / 2),
                                     src_img_height - abs(coefficient) * src_img_height / 2],
                                    [src_img_width + abs(coefficient) * src_img_width / 2,
                                     src_img_height - abs(coefficient) * src_img_height / 2]])
        dst_img = transform_perspective(src_img, origin_points, warped_points, src_img_width, src_img_height)
    return dst_img


def motion_blur_filter(src_img, amplitude):
    anchor = (-1, -1)
    depth = -1
    kernel_size = (amplitude, amplitude)
    dst_img = cv2.boxFilter(src=src_img, ddepth=depth, ksize=kernel_size, anchor=anchor)
    return dst_img


def modify_contrast_and_brightness(src_img, alpha, beta):
    src_img_height, src_img_width, src_img_channels = get_img_properties(src_img)
    dst_img = np.zeros((src_img_height, src_img_width, src_img_channels), np.uint8)
    for y in range(src_img_height):
        for x in range(src_img_width):
            for c in range(src_img_channels):
                dst_img[y, x, c] = np.clip(alpha * src_img[y, x, c] + beta, 0, 255)
    return dst_img


def overlay_image(background, foreground, x_offset=0, y_offset=0):
    foreground_height, foreground_width, foreground_channels = get_img_properties(foreground)
    alpha = foreground[:, :, 3] / 255
    bkg_copy = background.copy()
    bkg_copy[:, :, 0] = (1 - alpha) * bkg_copy[:, :, 0] + alpha * foreground[:, :, 0]
    bkg_copy[:, :, 1] = (1 - alpha) * bkg_copy[:, :, 1] + alpha * foreground[:, :, 1]
    bkg_copy[:, :, 2] = (1 - alpha) * bkg_copy[:, :, 2] + alpha * foreground[:, :, 2]
    return bkg_copy


    # foreground_height, foreground_width, foreground_channels = get_img_properties(foreground)
    # bkg_copy = background.copy()
    # y1, y2 = y_offset, y_offset + foreground_height
    # x1, x2 = x_offset, x_offset + foreground_width
    # foreground = cv2.cvtColor(foreground, cv2.COLOR_RGB2RGBA)
    #
    # alpha_f = foreground[:, :, 3] / 510.0
    # alpha_b = 1.0 - alpha_f
    #
    # for c in range(0, 3):
    #     bkg_copy[y1:y2, x1:x2, c] = (alpha_f * foreground[y1:y2, x1:x2, c] + alpha_b * foreground[y1:y2, x1:x2, c])
    # return bkg_copy

    # src_img_height, src_img_width, src_img_channels = get_img_properties(src_img)
    # loc_x = location[0]
    # loc_y = location[1]
    # dst_img = background.copy()
    # for y in range(max(0, loc_y), src_img_height):
    #     fy = y - loc_y
    #     if fy > src_img_height:
    #         break
    #     for x in range(max(0, loc_x), src_img_width):
    #         fx = x - loc_x
    #         if fx > src_img_width:
    #             break
    #         # TODO determine the opacity of the foreground


def add_shadow(src_img, show_equalized, min_color, max_color, min_area=0.3, max_area=0.9, shapes_nr=3):
    src_img_height, src_img_width, src_img_channels = get_img_properties(src_img)
    shadow_img = np.zeros((src_img_height, src_img_width, src_img_channels), np.uint8)
    working_img = src_img.copy()
    black_lightness = np.random.uniform(min_color, max_color)
    shadow_img.fill(black_lightness)
    center = (randint(int(min_area * src_img_width), int(max_area * src_img_width)),
              randint(int(min_area * src_img_height), int(max_area * src_img_height)))
    shape_code = randint(0, shapes_nr)
    if shape_code == 1:
        # generate ellipse
        ellipse_axis_a = randint(int(min_area * (min(src_img_height, src_img_width))),
                                 int(max_area * (min(src_img_height, src_img_width))))
        ellipse_axis_b = randint(int(min_area * (min(src_img_height, src_img_width))),
                                 int(max_area * (min(src_img_height, src_img_width))))
        ellipse_angle = randint(0, 360)
        cv2.ellipse(shadow_img, center, (ellipse_axis_a, ellipse_axis_b), ellipse_angle, 0, 360, 0)
    elif shape_code == 2:
        # generate parallelogram
        rect_side_a = randint(int(min_area * (min(src_img_height, src_img_width))),
                              int(max_area * (min(src_img_height, src_img_width))))
        rect_side_b = randint(int(min_area * (min(src_img_height, src_img_width))),
                              int(max_area * (min(src_img_height, src_img_width))))
        cv2.rectangle(shadow_img, center, (rect_side_a, rect_side_b), 0)
    else:
        # generate circle
        radius = randint(int(min_area * (min(src_img_height, src_img_width))),
                         int(max_area * (min(src_img_height, src_img_width))))
        cv2.circle(shadow_img, center, radius, 0)

    blur_radius_x = randint(int(min_area * src_img_width), int(max_area * src_img_width))
    if blur_radius_x % 2 == 0:
        blur_radius_x -= 1
    blur_radius_y = randint(int(min_area * src_img_height), int(max_area * src_img_height))
    if blur_radius_y % 2 == 0:
        blur_radius_y -= 1
    shadow_img = cv2.GaussianBlur(shadow_img, (blur_radius_x, blur_radius_y), sigmaX=blur_radius_x,
                                  sigmaY=blur_radius_y)
    dst_img = cv2.subtract(working_img, shadow_img)
    if show_equalized:
        channel_b, channel_g, channel_r = cv2.split(dst_img)
        cv2.equalizeHist(channel_b, channel_b)
        cv2.equalizeHist(channel_g, channel_g)
        cv2.equalizeHist(channel_r, channel_r)
        dst_img = cv2.merge([channel_b, channel_g, channel_r])
    return dst_img


def apply_grain_effect(src_img, min_noise, max_noise):
    src_img_height, src_img_width, src_img_channels = get_img_properties(src_img)
    grain_mask = np.zeros((src_img_height, src_img_width, src_img_channels), np.uint8)
    if src_img_channels == 1:
        working_img = src_img.copy()
        cv2.randn(grain_mask, min_noise, max_noise)
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        # cv2.dilate(grain_mask, grain_mask, element)
        grain_mask = cv2.GaussianBlur(grain_mask, (21, 21), 21, 21)
        cv2.subtract(working_img, grain_mask, working_img)
    elif src_img_channels == 3:
        working_img = src_img.copy()
        grain_channel_r, grain_channel_g, grain_channel_b = cv2.split(working_img)
        cv2.randn(grain_mask, np.full(3, min_noise), np.full(3, max_noise))
        # cv2.randn(grain_mask, min_noise, max_noise)
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5), (0, 0))
        grain_mask = cv2.dilate(grain_mask, element)
        grain_mask = cv2.GaussianBlur(grain_mask, (21, 21), 21, 21)
        grain_mask_channel_r, grain_mask_channel_g, grain_mask_channel_b = cv2.split(grain_mask)
        grain_channel_r = cv2.subtract(grain_channel_r, grain_mask_channel_r)
        grain_channel_g = cv2.subtract(grain_channel_g, grain_mask_channel_g)
        grain_channel_b = cv2.subtract(grain_channel_b, grain_mask_channel_b)
        cv2.merge([grain_channel_r, grain_channel_g, grain_channel_b], working_img)
    elif src_img_channels == 4:
        working_img = src_img.copy()
        grain_channel_r, grain_channel_g, grain_channel_b, grain_channel_a = cv2.split(working_img)
        cv2.randn(grain_mask, np.full(4, min_noise), np.full(4, max_noise))
        # cv2.randn(grain_mask, min_noise, max_noise)
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5), (0, 0))
        grain_mask = cv2.dilate(grain_mask, element)
        grain_mask = cv2.GaussianBlur(grain_mask, (21, 21), 21, 21)
        grain_mask_channel_r, grain_mask_channel_g, grain_mask_channel_b, grain_mask_channel_a = cv2.split(grain_mask)
        grain_channel_r = cv2.subtract(grain_channel_r, grain_mask_channel_r)
        grain_channel_g = cv2.subtract(grain_channel_g, grain_mask_channel_g)
        grain_channel_b = cv2.subtract(grain_channel_b, grain_mask_channel_b)
        grain_channel_a = cv2.subtract(grain_channel_a, grain_mask_channel_a)
        cv2.merge([grain_channel_r, grain_channel_g, grain_channel_b, grain_channel_a], working_img)
    else:
        error_pop_up()
        return src_img
    return working_img


def add_salt_and_pepper_noise(src_img):
    src_img_height, src_img_width, src_img_channels = get_img_properties(src_img)
    src_img_copy = src_img.copy()
    noise = np.zeros((src_img_height, src_img_width), np.uint8)
    cv2.randu(noise, 0, 255)
    black = noise < 1
    white = noise > 250
    white_offset = 255 - abs(np.random.normal() * 5)
    black_offset = abs(np.random.normal() * 5)
    for y in range(src_img_height):
        for x in range(src_img_width):
            if white[y, x]:
                src_img_copy[y, x] = white_offset
            elif black[y, x]:
                src_img_copy[y, x] = black_offset
    # src_img.setTo(white_offset, white)
    # src_img.setTo(black_offset, black)

    element = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2), (0, 0))
    src_img_copy = cv2.dilate(src_img_copy, element)
    dst_img = cv2.GaussianBlur(src_img_copy, (3, 3), 1, 1, cv2.BORDER_CONSTANT)
    return dst_img


# TODO
def apply_fish_eye_effect(src_img, k, center_x, center_y):
    src_img_height, src_img_width, src_img_channels = get_img_properties(src_img)
    if src_img_channels < 4:
        error_pop_up()
        return src_img
    working_img = np.zeros(src_img.shape, np.uint8)


def apply_chromatic_abberation(src_img, x_shift, y_shift):
    src_img_height, src_img_width, src_img_channels = get_img_properties(src_img)
    if src_img_channels < 3:
        error_pop_up()
        return src_img
    if src_img_channels == 3:
        channel_b, channel_g, channel_r = cv2.split(src_img)
        channel = np.random.uniform(0, 4)
        n_shift_x = np.random.uniform(-x_shift, x_shift) if x_shift > 0 else np.random.uniform(x_shift, -x_shift)
        n_shift_y = np.random.uniform(-y_shift, y_shift) if y_shift > 0 else np.random.uniform(y_shift, -y_shift)
        if channel == 0:
            channel_b = remap_image(channel_b, n_shift_x, n_shift_y, channel_b.shape[0], channel_b.shape[1])
        elif channel == 1:
            channel_g = remap_image(channel_g, n_shift_x, n_shift_y, channel_g.shape[0], channel_g.shape[1])
        else:
            channel_r = remap_image(channel_r, n_shift_x, n_shift_y, channel_r.shape[0], channel_r.shape[1])
        dst_img = cv2.merge([channel_b, channel_g, channel_r])
    elif src_img_channels == 4:
        channel_b, channel_g, channel_r, channel_a = cv2.split(src_img)
        channel = np.random.uniform(0, 4)
        n_shift_x = np.random.uniform(-x_shift, x_shift) if x_shift > 0 else np.random.uniform(x_shift, -x_shift)
        n_shift_y = np.random.uniform(-y_shift, y_shift) if y_shift > 0 else np.random.uniform(y_shift, -y_shift)
        if channel == 0:
            channel_b = remap_image(channel_b, n_shift_x, n_shift_y, channel_b.shape[0], channel_b.shape[1])
        elif channel == 1:
            channel_g = remap_image(channel_g, n_shift_x, n_shift_y, channel_g.shape[0], channel_g.shape[1])
        else:
            channel_r = remap_image(channel_r, n_shift_x, n_shift_y, channel_r.shape[0], channel_r.shape[1])
        dst_img = cv2.merge([channel_b, channel_g, channel_r, channel_a])
    return dst_img


def apply_halo(src_img, amount):
    if amount % 2 == 0:
        amount += 1
    working_img = cv2.GaussianBlur(src_img, (amount, amount), amount, amount)
    working_img = cv2.subtract(src_img, working_img)
    dst_img = cv2.add(working_img, src_img)
    return dst_img


def segment_red(src_img, thr1, thr2, thl, rg_dif, gb_dif, br_dif):
    src_img_height, src_img_width, src_img_channels = get_img_properties(src_img)
    dst_mask = np.zeros((src_img_height, src_img_width), np.uint8)
    I = 0
    for i in range(src_img_width):
        for j in range(src_img_height):
            if int(src_img[j, i, 2]) > 200 and int(src_img[j, i, 1]) < 100 and int(src_img[j, i, 0] < 100):
                dst_mask[j, i] = 255
            rg = abs(int(src_img[j, i, 0]) - int(src_img[j, i, 1]))
            gb = abs(int(src_img[j, i, 1]) - int(src_img[j, i, 2]))
            br = abs(int(src_img[j, i, 2]) - int(src_img[j, i, 0]))
            # print(rg, gb, br)
            I = int(src_img[j, i, 0]) + int(src_img[j, i, 1]) + int(src_img[j, i, 2])
            if I < thl:
                dst_mask[j, i] = 0
            if rg <= rg_dif and gb <= gb_dif and br <= br_dif:
                dst_mask[j, i] = 0
            if I == 0:
                I += 1
            p1 = (1 / sqrt(2)) * (int(src_img[j, i, 0]) - int(src_img[j, i, 2])) / I
            p2 = (1 / sqrt(6)) * (2 * int(src_img[j, i, 1]) - int(src_img[j, i, 2]) - int(src_img[j, i, 0])) / I
            if (p1 >= thr1) and (p2 <= thr2):
                dst_mask[j, i] = 255
            else:
                dst_mask[j, i] = 0
    # element = cv2.dilate(dst_mask, dst_mask)
    return dst_mask


def non_color_segment(src_img, thl, rg_dif, gb_dif, br_dif):
    src_img_height, src_img_width, src_img_channels = get_img_properties(src_img)
    dst_mask = np.zeros((src_img_height, src_img_width), np.uint8)
    for i in range(src_img_width):
        for j in range(src_img_height):
            rg = abs(int(src_img[j, i, 0]) - int(src_img[j, i, 1]))
            gb = abs(int(src_img[j, i, 1]) - int(src_img[j, i, 2]))
            br = abs(int(src_img[j, i, 2]) - int(src_img[j, i, 0]))
            # print(src_img[j, i, 0], src_img[j, i, 1], src_img[j, i, 2], rg, gb, br)
            I = src_img[j, i, 2] + src_img[j, i, 1] + src_img[j, i, 0]
            if I > thl and rg <= rg_dif and gb <= gb_dif and br <= br_dif:
                dst_mask[j, i] = 255
            else:
                dst_mask[j, i] = 0
    return dst_mask


def segment_white(src_img, thl, rg_dif, gb_dif, br_dif):
    return non_color_segment(src_img, thl, rg_dif, gb_dif, br_dif)


def segment_black(src_img, thl, rg_dif, gb_dif, br_dif):
    return non_color_segment(src_img, thl, rg_dif, gb_dif, br_dif)


def randomise_colors_for_templates(src_img, colors_distributions, thr1, thr2, thl, rg_dif, gb_dif, br_dif):
    src_img_height, src_img_width, src_img_channels = get_img_properties(src_img)
    if src_img_channels != 4:
        error_pop_up()
        return src_img
    background_img = np.zeros((src_img_height, src_img_width, 3), np.uint8)
    background_img[:, :, 1] = 255
    bgra_with_bg_img = overlay_image(background_img, src_img, 0, 0)
    cv2.imshow("bgr", bgra_with_bg_img)
    cv2.waitKey()
    bgr_img = cv2.cvtColor(bgra_with_bg_img, cv2.COLOR_BGRA2BGR)
    cv2.imshow("bgr", bgr_img)
    cv2.waitKey()
    dst_img = src_img.copy()
    # rand_g_noise = np.random.normal()
    # rand_g_noise = gaussian_random()
    rand_g_noise = 0.52871
    # print(rand_g_noise)
    # quit()
    blue_val = colors_distributions["red_b_mean"] + int(colors_distributions["red_b_dev"] / 2 * rand_g_noise)
    green_val = colors_distributions["red_g_mean"] + int(colors_distributions["red_g_dev"] / 2 * rand_g_noise)
    red_val = colors_distributions["red_r_mean"] + int(colors_distributions["red_r_dev"] / 2 * rand_g_noise)
    # print(blue_val, green_val, red_val)
    # quit()
    red = bgr_channels(blue_val, green_val, red_val)
    blue_val = colors_distributions["white_b_mean"] + int(colors_distributions["white_b_dev"] / 2 * rand_g_noise)
    green_val = colors_distributions["white_g_mean"] + int(colors_distributions["white_g_dev"] / 2 * rand_g_noise)
    red_val = colors_distributions["white_r_mean"] + int(colors_distributions["white_r_dev"] / 2 * rand_g_noise)
    white = bgr_channels(blue_val, green_val, red_val)
    blue_val = colors_distributions["black_b_mean"] + int(colors_distributions["black_b_dev"] / 2 * rand_g_noise)
    green_val = colors_distributions["black_g_mean"] + int(colors_distributions["black_g_dev"] / 2 * rand_g_noise)
    red_val = colors_distributions["black_r_mean"] + int(colors_distributions["black_r_dev"] / 2 * rand_g_noise)
    black = bgr_channels(blue_val, green_val, red_val)
    red_mask = segment_red(bgr_img, thr1, thr2, thl, rg_dif, gb_dif, br_dif)
    black_mask = segment_black(bgr_img, thl, rg_dif, gb_dif, br_dif)
    white_mask = segment_white(bgr_img, thl, rg_dif, gb_dif, br_dif)
    for i in range(src_img_width):
        for j in range(src_img_height):
            if red_mask[j, i] != 0:
                change_color(dst_img, red, j, i)
            if black_mask[j, i] != 0:
                change_color(dst_img, black, j, i)
            if white_mask[j, i] != 0:
                change_color(dst_img, white, j, i)
    return dst_img


if __name__ == "__main__":
    src_img = cv2.imread("bkg.png", cv2.IMREAD_UNCHANGED)
    # dst_img = randomise_colors_for_templates(src_img, colors_distributions, 0.024, -0.027, 100, 25, 25, 25)
    # # dst_img = perspective_transform_vertical(src_img, coefficient=-0.3)
    # # dst_img = motion_blur_filter(src_img, 3)
    # # dst_img = modify_contrast_and_brightness(src_img, 1, 20)
    # # dst_img = add_salt_and_pepper_noise(src_img)
    # # dst_img = apply_chromatic_abberation(src_img, 1000, 1000)
    # # dst_img = apply_halo(src_img, 33)
    # dst_img = apply_grain_effect(src_img, 0, 30)
    foreground = cv2.imread("lightning.png", cv2.IMREAD_UNCHANGED)
    dst_img = overlay_image(src_img, foreground, 50, 50)
    # # dst_img = segment_red(src_img, 0.024, -0.027, 100, 25, 25, 25)
    # dst_img = add_shadow(src_img, True, 0, 0, 0.3, 0.9, 3)
    # # dst_img = non_color_segment(src_img, 100, 25, 25, 25)
    cv2.imshow("initial", src_img)
    cv2.imshow("test", dst_img)
    cv2.waitKey()
