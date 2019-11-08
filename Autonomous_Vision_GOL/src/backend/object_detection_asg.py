from artificial_samples_generator import ASG, ASGObjectDetectionParameters, NUMBER_OF_ILLUMINATIONS, NUMBER_OF_PERSPECTIVES
from random import randint
import shutil
import os.path
import yaml
import glob
import numpy as np
import cv2
from config_file import config_file_path
from image_transformation import ImageTransformer


# Class handles the Artificial Samples Generator call
class ObjectDetectionASG(ASG):
    annot_top_left = [0, 0]
    annot_bot_right = [0, 0]

    def __init__(self):
        self.transformer = ImageTransformer()

    def is_valid_config_file(self):
        if os.path.exists(config_file_path) == False:
            print("Generation config file does not exist!")
            return False
        if config_file_path.split('.')[-1] != "yml":
            print("Invalid config file!")
            return False
        return True

    def read_config_file(self):
        config_file = yaml.safe_load(open(config_file_path))

        # get paths from config file
        ASGObjectDetectionParameters.fields['template_path'] = config_file['Path to templates']
        ASGObjectDetectionParameters.fields['template_num'] = config_file['Nr of templates in folder']
        ASGObjectDetectionParameters.fields['output_path'] = config_file['Output path']

        # get parameter values from config file
        ASGObjectDetectionParameters.fields['min_resize'] = config_file['Min resize']
        ASGObjectDetectionParameters.fields['max_resize'] = config_file['Max resize']
        ASGObjectDetectionParameters.fields['min_h_persp'] = config_file['Min H perspective']
        ASGObjectDetectionParameters.fields['max_h_persp'] = config_file['Max H perspective']
        ASGObjectDetectionParameters.fields['min_v_persp'] = config_file['Min V perspective']
        ASGObjectDetectionParameters.fields['max_v_persp'] = config_file['Max V perspective']
        ASGObjectDetectionParameters.fields['min_brightness'] = config_file['Min light']
        ASGObjectDetectionParameters.fields['max_brightness'] = config_file['Max light']
        ASGObjectDetectionParameters.fields['min_noise'] = config_file['Min noise value']
        ASGObjectDetectionParameters.fields['max_noise'] = config_file['Max noise value']
        ASGObjectDetectionParameters.fields['min_blur'] = config_file['Min blur amplitude']
        ASGObjectDetectionParameters.fields['max_blur'] = config_file['Max blur amplitude']
        ASGObjectDetectionParameters.fields['max_aberration_v'] = config_file['Vertical max aberration']
        ASGObjectDetectionParameters.fields['max_aberration_h'] = config_file['Horizontal max aberration']
        ASGObjectDetectionParameters.fields['radial_distorsion'] = config_file['Radial distortion']
        ASGObjectDetectionParameters.fields['halo_amount'] = config_file['Halo amount']
        ASGObjectDetectionParameters.fields['max_enlargement_bg_v'] = config_file['Max enlarge background vertical']
        ASGObjectDetectionParameters.fields['max_enlargement_bg_h'] = config_file['Max enlarge background horizontal']

        # reading otha parameters from config file
        otha_values = config_file['Otha']
        ASGObjectDetectionParameters.otha_params['thr1'] = otha_values[0]
        ASGObjectDetectionParameters.otha_params['thr2'] = otha_values[1]
        ASGObjectDetectionParameters.otha_params['thl'] = otha_values[2]
        ASGObjectDetectionParameters.otha_params['rg_dif'] = otha_values[3]
        ASGObjectDetectionParameters.otha_params['gb_dif'] = otha_values[4]
        ASGObjectDetectionParameters.otha_params['br_dif'] = otha_values[5]

        # reading black distributions
        black_distr_values = config_file['Black']
        ASGObjectDetectionParameters.black_distributions['blue_mean'] = black_distr_values[0]
        ASGObjectDetectionParameters.black_distributions['blue_deviation'] = black_distr_values[1]
        ASGObjectDetectionParameters.black_distributions['green_mean'] = black_distr_values[2]
        ASGObjectDetectionParameters.black_distributions['green_deviation'] = black_distr_values[3]
        ASGObjectDetectionParameters.black_distributions['red_mean'] = black_distr_values[4]
        ASGObjectDetectionParameters.black_distributions['red_deviation'] = black_distr_values[5]

        # reading white distributions
        white_distr_values = config_file['White']
        ASGObjectDetectionParameters.white_distributions['blue_mean'] = white_distr_values[0]
        ASGObjectDetectionParameters.white_distributions['blue_deviation'] = white_distr_values[1]
        ASGObjectDetectionParameters.white_distributions['green_mean'] = white_distr_values[2]
        ASGObjectDetectionParameters.white_distributions['green_deviation'] = white_distr_values[3]
        ASGObjectDetectionParameters.white_distributions['red_mean'] = white_distr_values[4]
        ASGObjectDetectionParameters.white_distributions['red_deviation'] = white_distr_values[5]

        # reading red distributions
        red_distr_values = config_file['Red']
        ASGObjectDetectionParameters.red_distributions['blue_mean'] = red_distr_values[0]
        ASGObjectDetectionParameters.red_distributions['blue_deviation'] = red_distr_values[1]
        ASGObjectDetectionParameters.red_distributions['green_mean'] = red_distr_values[2]
        ASGObjectDetectionParameters.red_distributions['green_deviation'] = red_distr_values[3]
        ASGObjectDetectionParameters.red_distributions['red_mean'] = red_distr_values[4]
        ASGObjectDetectionParameters.red_distributions['red_deviation'] = red_distr_values[5]

        # choose optionals
        ASGObjectDetectionParameters.background_status = config_file['Use background']
        ASGObjectDetectionParameters.fish_eye_status = config_file['Use fish eye effect']
        ASGObjectDetectionParameters.randomise_colors_status = config_file['Use randomise colors']
        ASGObjectDetectionParameters.salt_and_pepper_status = config_file['Use salt and pepper noise']
        ASGObjectDetectionParameters.crop_status = config_file['Crop image']
        if ASGObjectDetectionParameters.crop_status:
            aspect_ratio = config_file['Aspect ratio']
            ASGObjectDetectionParameters.aspect_ratio_width = aspect_ratio[0]
            ASGObjectDetectionParameters.aspect_ratio_height = aspect_ratio[1]
        ASGObjectDetectionParameters.label_status = config_file['Use labels']

    def generate(self, nr_of_generations):
        templates_path = ASGObjectDetectionParameters.fields['template_path']
        output_path = ASGObjectDetectionParameters.fields['output_path']
        if ASGObjectDetectionParameters.fields['max_h_persp'] == 0 or ASGObjectDetectionParameters.fields['min_h_persp'] == 0:
            step_perspective_h = 1
        else:
            step_perspective_h = abs((ASGObjectDetectionParameters.fields['max_h_persp'] - ASGObjectDetectionParameters.fields['min_h_persp']) / (
                        NUMBER_OF_PERSPECTIVES - 1))
        if ASGObjectDetectionParameters.fields['max_v_persp'] == 0 or ASGObjectDetectionParameters.fields['min_v_persp'] == 0:
            step_perspective_v = 1
        else:
            step_perspective_v = abs((ASGObjectDetectionParameters.fields['max_v_persp'] - ASGObjectDetectionParameters.fields['min_v_persp']) / (
                        NUMBER_OF_PERSPECTIVES - 1))
        number_of_images = ASGObjectDetectionParameters.fields['template_num']
        img_files = [f for f in glob.glob(templates_path + "/*.png")]
        for pos in range(number_of_images):
            i = 0
            img_template = cv2.imread(img_files[pos], cv2.IMREAD_UNCHANGED)
            if img_template is None:
                print("Warning: image not loaded.")
                quit()
            for generations in np.arange(nr_of_generations):
                for illumination in np.arange(NUMBER_OF_ILLUMINATIONS):
                    for perspective_angle_h in np.arange(ASGObjectDetectionParameters.fields['min_h_persp'],
                                                         ASGObjectDetectionParameters.fields['max_h_persp'] + step_perspective_h,
                                                         step_perspective_h):
                        # print("Doing persp transf h")
                        img_perspective_transform_h = self.transformer.perspective_transform_horizontal(img_template,
                                                                                       perspective_angle_h)
                        for perspective_angle_v in np.arange(ASGObjectDetectionParameters.fields['min_v_persp'],
                                                             ASGObjectDetectionParameters.fields['max_v_persp'] + step_perspective_v,
                                                             step_perspective_v):
                            img_perspective_transform_v = self.transformer.perspective_transform_vertical(img_perspective_transform_h,
                                                                                         perspective_angle_v)
                            # TODO apply aspect ratio
                            shadowed_img = self.transformer.add_shadow(img_perspective_transform_v, False,
                                                      ASGObjectDetectionParameters.fields['min_brightness'],
                                                      ASGObjectDetectionParameters.fields['max_brightness'])
                            if ASGObjectDetectionParameters.salt_and_pepper_status:
                                salt_and_pepper_img = self.transformer.add_salt_and_pepper_noise(shadowed_img)
                            for blur_amplitude in np.arange(ASGObjectDetectionParameters.fields['min_blur'],
                                                            ASGObjectDetectionParameters.fields['max_blur'] + 1):
                                if 0 < ASGObjectDetectionParameters.fields['min_blur'] < ASGObjectDetectionParameters.fields['max_blur']:
                                    img_blurred = self.transformer.motion_blur_filter(shadowed_img, blur_amplitude)
                                else:
                                    img_blurred = shadowed_img.copy()
                                img_grain_effect = self.transformer.apply_grain_effect(img_blurred, ASGObjectDetectionParameters.fields['min_noise'],
                                                                      ASGObjectDetectionParameters.fields['max_noise'])
                                img_chromatic_abberation = self.transformer.apply_chromatic_abberation(img_grain_effect,
                                                                                      ASGObjectDetectionParameters.fields[
                                                                                          'max_aberration_h'],
                                                                                      ASGObjectDetectionParameters.fields[
                                                                                          'max_aberration_v'])
                                img_halo = self.transformer.apply_halo(img_chromatic_abberation, ASGObjectDetectionParameters.fields['halo_amount'])
                                img_brightness_contrast = self.transformer.modify_contrast_and_brightness(img_halo, 1, randint(0, 10))
                                if 0 < ASGObjectDetectionParameters.fields['min_resize'] < ASGObjectDetectionParameters.fields['max_resize']:
                                    img_size = np.random.uniform(ASGObjectDetectionParameters.fields['min_resize'],
                                                                 ASGObjectDetectionParameters.fields['max_resize'])
                                    aspect_ratio_factor = max(img_brightness_contrast.shape[0],
                                                              img_brightness_contrast.shape[1]) / min(
                                        img_brightness_contrast.shape[0],
                                        img_brightness_contrast.shape[1])
                                    if img_brightness_contrast.shape[1] > img_brightness_contrast.shape[0]:
                                        new_size = (img_size, img_size // aspect_ratio_factor)
                                    else:
                                        new_size = (img_size // aspect_ratio_factor, img_size)
                                    cv2.resize(img_brightness_contrast, new_size, dst=img_brightness_contrast)
                                cv2.imwrite(output_path + "/%s.png" % i, img_brightness_contrast)
                                i += 1

    def callback(self, nr_of_generations):
        self.read_config_file()
        self.generate(nr_of_generations)

    def run_generator_from_parameters(self, parameters, epoch_nr, templates_path, output_path):
        self.get_params(parameters)
        self.generate_without_config(epoch_nr, templates_path, output_path)

    def get_params(self, parameters):
        # get parameter values from Pareto optimizer
        ASGObjectDetectionParameters.fields['min_resize'] = parameters[0]
        ASGObjectDetectionParameters.fields['max_resize'] = parameters[1]
        ASGObjectDetectionParameters.fields['min_h_persp'] = parameters[2]
        ASGObjectDetectionParameters.fields['max_h_persp'] = parameters[3]
        ASGObjectDetectionParameters.fields['min_v_persp'] = parameters[4]
        ASGObjectDetectionParameters.fields['max_v_persp'] = parameters[5]
        ASGObjectDetectionParameters.fields['min_brightness'] = parameters[6]
        ASGObjectDetectionParameters.fields['max_brightness'] = parameters[7]
        ASGObjectDetectionParameters.fields['min_noise'] = parameters[8]
        ASGObjectDetectionParameters.fields['max_noise'] = parameters[9]
        ASGObjectDetectionParameters.fields['min_blur'] = parameters[10]
        ASGObjectDetectionParameters.fields['max_blur'] = parameters[11]
        ASGObjectDetectionParameters.fields['max_aberration_v'] = parameters[12]
        ASGObjectDetectionParameters.fields['max_aberration_h'] = parameters[13]
        ASGObjectDetectionParameters.fields['radial_distorsion'] = parameters[14]
        ASGObjectDetectionParameters.fields['halo_amount'] = parameters[15]
        ASGObjectDetectionParameters.fields['max_enlargement_bg_v'] = parameters[16]
        ASGObjectDetectionParameters.fields['max_enlargement_bg_h'] = parameters[17]

        # set default values for the remaining parameters
        self.default_otha()
        self.default_aspect_ratio()
        self.default_color_distribution()

    def default_otha(self):
        ASGObjectDetectionParameters.otha_params['thr1'] = 0.024
        ASGObjectDetectionParameters.otha_params['thr2'] = -0.027
        ASGObjectDetectionParameters.otha_params['thl'] = 25
        ASGObjectDetectionParameters.otha_params['br_dif'] = 100
        ASGObjectDetectionParameters.otha_params['gb_dif'] = 25
        ASGObjectDetectionParameters.otha_params['rg_dif'] = 25

    def default_aspect_ratio(self):
        ASGObjectDetectionParameters.crop_status = True
        ASGObjectDetectionParameters.aspect_ratio_height = 1
        ASGObjectDetectionParameters.aspect_ratio_width = 1

    def default_color_distribution(self):
        # set default values for black distributions
        ASGObjectDetectionParameters.black_distributions['blue_mean'] = 40
        ASGObjectDetectionParameters.black_distributions['blue_deviation'] = 28
        ASGObjectDetectionParameters.black_distributions['green_mean'] = 39
        ASGObjectDetectionParameters.black_distributions['green_deviations'] = 26
        ASGObjectDetectionParameters.black_distributions['red_mean'] = 33
        ASGObjectDetectionParameters.black_distributions['red_deviation'] = 26

        # set default values for white distributions
        ASGObjectDetectionParameters.white_distributions['blue_mean'] = 151
        ASGObjectDetectionParameters.white_distributions['blue_deviation'] = 48
        ASGObjectDetectionParameters.white_distributions['green_mean'] = 154
        ASGObjectDetectionParameters.white_distributions['green_deviations'] = 50
        ASGObjectDetectionParameters.white_distributions['red_mean'] = 148
        ASGObjectDetectionParameters.white_distributions['red_deviation'] = 54

        # set default values for red distributions
        ASGObjectDetectionParameters.red_distributions['blue_mean'] = 56
        ASGObjectDetectionParameters.red_distributions['blue_deviation'] = 27
        ASGObjectDetectionParameters.red_distributions['green_mean'] = 58
        ASGObjectDetectionParameters.red_distributions['green_deviations'] = 30
        ASGObjectDetectionParameters.red_distributions['red_mean'] = 126
        ASGObjectDetectionParameters.red_distributions['red_deviation'] = 55

    # Generates images using the parameters obtained after optimization
    def generate_without_config(self, epoch_nr, templates_path, output_path):
        if ASGObjectDetectionParameters.fields['max_h_persp'] == 0 or ASGObjectDetectionParameters.fields['min_h_persp'] == 0:
            step_perspective_h = 1
        else:
            step_perspective_h = abs((ASGObjectDetectionParameters.fields['max_h_persp'] - ASGObjectDetectionParameters.fields['min_h_persp']) / (
                    NUMBER_OF_PERSPECTIVES - 1))
        if ASGObjectDetectionParameters.fields['max_v_persp'] == 0 or ASGObjectDetectionParameters.fields['min_v_persp'] == 0:
            step_perspective_v = 1
        else:
            step_perspective_v = abs((ASGObjectDetectionParameters.fields['max_v_persp'] - ASGObjectDetectionParameters.fields['min_v_persp']) / (
                    NUMBER_OF_PERSPECTIVES - 1))
        templates_classes_dirs = [name for name in os.listdir(templates_path) if os.path.isdir(os.path.join(templates_path, name))]
        bkg_path = os.path.join(templates_path, "..", "backgrounds")
        bkgs = [f for f in glob.glob(bkg_path + "/*.png")]
        for folder in templates_classes_dirs:
            img_output_path = os.path.join(output_path, folder)
            if os.path.exists(img_output_path):
                shutil.rmtree(img_output_path)
            os.mkdir(img_output_path)
            crt_img_idx = 0

            img_files = [f for f in glob.glob(os.path.join(templates_path, folder) + "/*.png")]
            number_of_images = len(img_files)
            # print(img_files[0].split('.')[-2].split("\\")[-1])
            for pos in range(number_of_images):
                # create a directory with the label name
                # img_output_path = os.path.join(output_path, img_files[pos].split('.')[-2].split("\\")[-1])
                # if os.path.exists(img_output_path):
                #     shutil.rmtree(img_output_path)
                # os.mkdir(img_output_path)
                # print(img_output_path)
                # quit()
                # crt_img_idx = 0
                img_template = cv2.imread(img_files[pos], cv2.IMREAD_UNCHANGED)
                if img_template is None:
                    print("Warning: image not loaded.")
                    quit()
                for illumination in np.arange(NUMBER_OF_ILLUMINATIONS):
                    for perspective_angle_h in np.arange(ASGObjectDetectionParameters.fields['min_h_persp'],
                                                         ASGObjectDetectionParameters.fields['max_h_persp'] + step_perspective_h,
                                                         step_perspective_h):
                        # print("Doing persp transf h")
                        ASG.annot_top_left = [34, 7]
                        ASG.annot_bot_right = [61, 89]
                        img_perspective_transform_h = self.transformer.perspective_transform_horizontal(img_template, perspective_angle_h)
                        for perspective_angle_v in np.arange(ASGObjectDetectionParameters.fields['min_v_persp'],
                                                             ASGObjectDetectionParameters.fields['max_v_persp'] + step_perspective_v,
                                                             step_perspective_v):
                            img_perspective_transform_v = self.transformer.perspective_transform_vertical(img_perspective_transform_h,
                                                                                         perspective_angle_v)
                            shadowed_img = self.transformer.add_shadow(img_perspective_transform_v, False,
                                                      ASGObjectDetectionParameters.fields['min_brightness'],
                                                      ASGObjectDetectionParameters.fields['max_brightness'])
                            if ASGObjectDetectionParameters.salt_and_pepper_status:
                                salt_and_pepper_img = self.transformer.add_salt_and_pepper_noise(shadowed_img)
                            for blur_amplitude in np.arange(ASGObjectDetectionParameters.fields['min_blur'],
                                                            ASGObjectDetectionParameters.fields['max_blur'] + 1):
                                if 0 < ASGObjectDetectionParameters.fields['min_blur'] < ASGObjectDetectionParameters.fields['max_blur']:
                                    img_blurred = self.transformer.motion_blur_filter(shadowed_img, blur_amplitude)
                                else:
                                    img_blurred = shadowed_img.copy()
                                img_grain_effect = self.transformer.apply_grain_effect(img_blurred, ASGObjectDetectionParameters.fields['min_noise'],
                                                                      ASGObjectDetectionParameters.fields['max_noise'])
                                img_chromatic_abberation = self.transformer.apply_chromatic_abberation(img_grain_effect,
                                                                                      ASGObjectDetectionParameters.fields[
                                                                                          'max_aberration_h'],
                                                                                      ASGObjectDetectionParameters.fields[
                                                                                          'max_aberration_v'])
                                img_halo = self.transformer.apply_halo(img_chromatic_abberation, ASGObjectDetectionParameters.fields['halo_amount'])
                                img_brightness_contrast = self.transformer.modify_contrast_and_brightness(img_halo, 1, randint(0, 10))
                                if 0 < ASGObjectDetectionParameters.fields['min_resize'] < ASGObjectDetectionParameters.fields['max_resize']:
                                    img_size = np.random.uniform(ASGObjectDetectionParameters.fields['min_resize'],
                                                                 ASGObjectDetectionParameters.fields['max_resize'])
                                    aspect_ratio_factor = max(img_brightness_contrast.shape[0],
                                                              img_brightness_contrast.shape[1]) / min(
                                        img_brightness_contrast.shape[0],
                                        img_brightness_contrast.shape[1])
                                    if img_brightness_contrast.shape[1] > img_brightness_contrast.shape[0]:
                                        new_size = (int(img_size), int(img_size // aspect_ratio_factor))
                                    else:
                                        new_size = (int(img_size // aspect_ratio_factor), int(img_size))
                                    cv2.resize(img_brightness_contrast, new_size, dst=img_brightness_contrast)
                                bkg_idx = randint(0, len(bkgs) - 1)
                                bkg_img = cv2.imread(bkgs[bkg_idx], cv2.IMREAD_UNCHANGED)
                                # final_img = overlay_image(bkg_img, img_brightness_contrast)
                                final_img = img_brightness_contrast
                                cv2.imwrite(img_output_path + "/%s.png" % crt_img_idx, final_img)
                                crt_img_idx += 1

        epochs_output_path = os.path.join(output_path, "..\\", "epochs_output", "epochs_output{0}".format(epoch_nr))
        shutil.copytree(output_path, epochs_output_path)