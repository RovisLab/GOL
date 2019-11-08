import random
import shutil
import os.path
import yaml
import glob
from config_file import config_file_path
from image_transformation import *

NUMBER_OF_PERSPECTIVES = 9
# NUMBER_OF_ILLUMINATIONS = 12
NUMBER_OF_ILLUMINATIONS = 1


class ASGObjectDetectionParameters(object):
    background_status = False
    fish_eye_status = False
    randomise_colors_status = False
    salt_and_pepper_status = False
    crop_status = False
    aspect_ratio_width = 1
    aspect_ratio_height = 1
    label_status = False

    black_distributions = {
        'blue_mean': 'N/A',
        'blue_deviation': 'N/A',
        'green_mean': 'N/A',
        'green_deviations': 'N/A',
        'red_mean': 'N/A',
        'red_deviation': 'N/A'
    }

    white_distributions = {
        'blue_mean': 'N/A',
        'blue_deviation': 'N/A',
        'green_mean': 'N/A',
        'green_deviations': 'N/A',
        'red_mean': 'N/A',
        'red_deviation': 'N/A'
    }

    red_distributions = {
        'blue_mean': 'N/A',
        'blue_deviation': 'N/A',
        'green_mean': 'N/A',
        'green_deviations': 'N/A',
        'red_mean': 'N/A',
        'red_deviation': 'N/A'
    }

    otha_params = {
        'thr1': 'N/A',
        'thr2': 'N/A',
        'thl': 'N/A',
        'rg_dif': 'N/A',
        'gb_dif': 'N/A',
        'br_dif': 'N/A'
    }

    fields = {
        'output_path': 'N/A',
        'template_path': 'N/A',
        'template_num': 'N/A',
        'background_path': 'N/A',
        'min_resize': 'N/A',
        'max_resize': 'N/A',
        'min_h_persp': 'N/A',
        'max_h_persp': 'N/A',
        'min_v_persp': 'N/A',
        'max_v_persp': 'N/A',
        'min_brightness': 'N/A',
        'max_brightness': 'N/A',
        'min_noise': 'N/A',
        'max_noise': 'N/A',
        'min_blur': 'N/A',
        'max_blur': 'N/A',
        'max_aberration_v': 'N/A',
        'max_aberration_h': 'N/A',
        'radial_distorsion': 'N/A',
        'halo_amount': 'N/A',
        'max_enlargement_bg_h': 'N/A',
        'max_enlargement_bg_v': 'N/A'
    }


# Class handles the Artificial Samples Generator call
class ASG(object):

    @staticmethod
    def is_valid_config_file():
        if os.path.exists(config_file_path) == False:
            print("Generation config file does not exist!")
            return False
        if config_file_path.split('.')[-1] != "yml":
            print("Invalid config file!")
            return False
        return True

    def read_config_file(self):
        pass

    def generate(self, nr_of_generations):
        pass

    def callback(self, nr_of_generations):
        self.read_config_file()
        self.generate(nr_of_generations)

    def run_generator_from_parameters(self, parameters, epoch_nr, templates_path, output_path):
        self.get_params(parameters)
        self.generate_without_config(epoch_nr, templates_path, output_path)

    # Get parameters from Pareto Optimization
    def get_params(self, parameters):
        pass

    # Generates artificial samples using the parameters obtained after optimization
    def generate_without_config(self, epoch_nr, templates_path, output_path):
        pass
