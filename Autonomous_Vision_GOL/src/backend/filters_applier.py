import ctypes
import yaml
import cv2
from image_transformation import *
from img_proc_utils import get_img_properties

# Filters applier DLL object

class FiltersParameters(object):
    paths = {

    }

    which = 0
    counter = 0

    minimum_values = {

    }

    maximum_values = {

    }

    other = {

    }


# Class handles parameters sampling
class FiltersApplier(object):
    def __init__(self, config_path):
        print("Entered filters applier")
        self.obj = lib.apply_filters_new()
        self._config_path = config_path

    def read_sample_config(self):
        config_file = yaml.safe_load(open(self._config_path))
        FiltersParameters.paths = config_file['Paths']
        FiltersParameters.which = config_file['Which']
        FiltersParameters.counter = config_file['Counter']
        FiltersParameters.minimum_values = config_file['Minimum']
        FiltersParameters.maximum_values = config_file['Maximum']
        FiltersParameters.other = config_file['Other']

    def __add_shadow(self, src_img, intensity):
        src_img_height, src_img_width, src_img_channels = get_img_properties(src_img)
        shadow_img = np.zeros((src_img_height, src_img_width, src_img_channels), np.uint8)
        shadow_img.fill(intensity)
        dst_img = cv2.subtract(src_img, shadow_img)
        return dst_img

    def apply_min(self, image):
        if FiltersParameters.minimum_values['H perspective'] != 0:
            image = perspective_transform_horizontal(image, FiltersParameters.minimum_values['H perspective'])
        if FiltersParameters.minimum_values['V perspective'] != 0:
            image = perspective_transform_vertical(image, FiltersParameters.minimum_values['V perspective'])
        if FiltersParameters.minimum_values['Brightness'] != 0:
            image = self.__add_shadow(image, FiltersParameters.minimum_values['Brightness'])
        if FiltersParameters.minimum_values['Blur'] != 0:
            image = motion_blur_filter(image, FiltersParameters.minimum_values['Blur'])
        if FiltersParameters.minimum_values['Noise'] != 0:
            image = apply_grain_effect(image, FiltersParameters.minimum_values['Noise'], FiltersParameters.minimum_values['Noise'])
        if FiltersParameters.other['H aberration'] != 0 or FiltersParameters.other['V aberration'] != 0:
            image = apply_chromatic_abberation(image, FiltersParameters.other['H aberration'], FiltersParameters.other['V aberration'])
        if FiltersParameters.other['Halo'] != 0:
            image = apply_halo(image, FiltersParameters.other['Halo'])
        if FiltersParameters.minimum_values['Resize'] != 0:
            image = cv2.resize(image, (FiltersParameters.minimum_values['Resize'], FiltersParameters.minimum_values['Resize']))
        return image

    def apply_max(self, image):
        if FiltersParameters.maximum_values['H perspective'] != 0:
            image = perspective_transform_horizontal(image, FiltersParameters.maximum_values['H perspective'])
        if FiltersParameters.maximum_values['V perspective'] != 0:
            image = perspective_transform_vertical(image, FiltersParameters.maximum_values['V perspective'])
        if FiltersParameters.maximum_values['Brightness'] != 0:
            image = self.__add_shadow(image, FiltersParameters.maximum_values['Brightness'])
        if FiltersParameters.maximum_values['Blur'] != 0:
            image = motion_blur_filter(image, FiltersParameters.maximum_values['Blur'])
        if FiltersParameters.maximum_values['Noise'] != 0:
            image = apply_grain_effect(image, FiltersParameters.maximum_values['Noise'], FiltersParameters.maximum_values['Noise'])
        if FiltersParameters.other['H aberration'] != 0 or FiltersParameters.other['V aberration'] != 0:
            image = apply_chromatic_abberation(image, FiltersParameters.other['H aberration'], FiltersParameters.other['V aberration'])
        if FiltersParameters.other['Halo'] != 0:
            image = apply_halo(image, FiltersParameters.other['Halo'])
        if FiltersParameters.maximum_values['Resize'] != 0:
            image = cv2.resize(image, (FiltersParameters.maximum_values['Resize'], FiltersParameters.maximum_values['Resize']))
        return image

    def apply_filters(self):
        template_img = cv2.imread(FiltersParameters.paths['Template path'], cv2.IMREAD_UNCHANGED)
        min_img = template_img.copy()
        max_img = template_img.copy()

        if FiltersParameters.counter == 1:
            cv2.imwrite(FiltersParameters.paths['Output path'] + '/Minimum.png', min_img)
            cv2.imwrite(FiltersParameters.paths['Output path'] + '/Maximum.png', max_img)
        if FiltersParameters.which == -1:
            min_img = self.apply_min(min_img)
            cv2.imwrite(FiltersParameters.paths['Output path'] + '/Minimum.png', min_img)
        if FiltersParameters.which == 1:
            max_img = self.apply_max(max_img)
            cv2.imwrite(FiltersParameters.paths['Output path'] + '/Maximum.png', max_img)
        if FiltersParameters.which == 0:
            min_img = self.apply_min(min_img)
            cv2.imwrite(FiltersParameters.paths['Output path'] + '/Minimum.png', min_img)
            max_img = self.apply_max(max_img)
            cv2.imwrite(FiltersParameters.paths['Output path'] + '/Maximum.png', max_img)

    # Function calls the filters applier
    def callback(self):
        self.read_sample_config()
        self.apply_filters()
        # c_path = self._config_path.encode('utf-8')
        # lib.apply_filters_callback(self.obj, ctypes.c_char_p(c_path))
