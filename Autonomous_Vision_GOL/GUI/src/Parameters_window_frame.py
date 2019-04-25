from tkinter import TOP, LEFT, RIGHT, BOTTOM, S, W, CENTER, Y, VERTICAL
from tkinter import Frame, Scale, Label, Button
from tkinter.ttk import Separator
from Apply_filters_frame import ApplyFiltersFrame
from PIL import Image
import glob
import random

# Values used for calculating number of samples that will be generated
NUMBER_OF_ILLUMINATIONS = 12
NUMBER_OF_PERSPECTIVES = 9


# Class handles second window frame, asking for parameters setting
class ParametersWindow(Frame):
    def __init__(self, master=None, name=None, param_dict=None):
        Frame.__init__(self, master=master, name=name)
        self._master = master
        self._param_dict = param_dict
        self._name = name

        # Variable for keeping track regarding whether the body() was called before or not
        self._body_pack_checker = False

        self.misc_dir_path = "../bin/misc"
        self.config_file_path = self.misc_dir_path + "/sampling_config.yml"

        self.win_width = 1500
        self.win_height = 750
        self.sampling_width = 450
        self.padding_x = 160
        self.separator_pad = 50
        self.param_width = self.win_width - self.sampling_width - self.separator_pad
        self.x_offset = int((self.winfo_screenwidth() - self.win_width) / 2)
        self.y_offset = int((self.winfo_screenheight() - self.win_height - 50) / 2)
        self._geometry = "{:d}x{:d}+{:d}+{:d}".format(self.win_width, self.win_height, self.x_offset, self.y_offset)

        self.frame_width = self.param_width - self.padding_x
        self.frame_height = self.win_height / 13

        self.config(width=self.win_width-self.padding_x, height=self.win_height-self.frame_height)
        self.pack_propagate(False)

        # For establishing which sample image will be affected by a parameter there will be three states as follows:
        # which = -1    - for parameters that affect only minimum
        # which = 0     - for parameters that affect both
        # which = 1     - for parameters that affect only maximum
        self.sampling_dict = {'Template path': '',
                              'Output path': self.misc_dir_path,
                              'Which': 0,
                              'Counter': 0}

        # Configuring sampling frame
        self.apply_filters = ApplyFiltersFrame(master=self, width=self.sampling_width, height=self.win_height)

        # Configuring title frame
        self.title_font = ('Segoe UI', 12)
        self.title_frame = Frame(self, width=self.frame_width, height=self.frame_height)
        self.title = "Parameters"

        self.W_geometry = {'orient': 'horizontal', 'length': 200}

        # Configuring main frame
        self.param_set_frame = Frame(self, name='param_set', width=self.frame_width, height=self.frame_height * 11)
        self.param_set_frame.pack_propagate(False)

        # Configuring radial distortion and halo frame
        self.distortion_n_halo_frame = Frame(self.param_set_frame, width=self.frame_width, height=self.frame_height)
        self.distortion_n_halo_frame.pack_propagate(False)
        self.radial_distortion_scale = Scale(self.distortion_n_halo_frame, self.W_geometry, name='dist', from_=0,
                                             to=0.0001, resolution=0.00001, command=lambda x: self.both_callback())
        self.distortion_center_x = 0
        self.distortion_center_y = 0
        self.halo_scale = Scale(self.distortion_n_halo_frame, self.W_geometry, name='halo', from_=0, to=15,
                                resolution=1, command=lambda x: self.both_callback())

        # Configuring vertical perspective frame
        self.V_perspective_frame = Frame(self.param_set_frame, name='frame v_persp',
                                         width=self.frame_width, height=self.frame_height)
        self.V_perspective_frame.pack_propagate(False)
        self.min_V_perspective_scale = Scale(self.V_perspective_frame, self.W_geometry, name='min v_persp', from_=-0.5,
                                             to=0.5, resolution=0.1,
                                             command=lambda value: self.min_callback(value, 'min v_persp'))
        self.max_V_perspective_scale = Scale(self.V_perspective_frame, self.W_geometry, name='max v_persp', from_=-0.5,
                                             to=0.5, resolution=0.1,
                                             command=lambda value: self.max_callback(value, 'max v_persp'))

        # Configuring horizontal perspective frame
        self.H_perspective_frame = Frame(self.param_set_frame, name='frame h_persp',
                                         width=self.frame_width, height=self.frame_height)
        self.H_perspective_frame.pack_propagate(False)
        self.min_H_perspective_scale = Scale(self.H_perspective_frame, self.W_geometry, name='min h_persp', from_=-0.5,
                                             to=0.5, resolution=0.1,
                                             command=lambda value: self.min_callback(value, 'min h_persp'))
        self.max_H_perspective_scale = Scale(self.H_perspective_frame, self.W_geometry, name='max h_persp', from_=-0.5,
                                             to=0.5, resolution=0.1,
                                             command=lambda value: self.max_callback(value, 'max h_persp'))

        # Configuring brightness frame
        self.brightness_frame = Frame(self.param_set_frame, name='frame bright',
                                      width=self.frame_width, height=self.frame_height)
        self.brightness_frame.pack_propagate(False)
        self.min_brightness_scale = Scale(self.brightness_frame, self.W_geometry, name='min bright', from_=0, to=60,
                                          resolution=1, command=lambda value: self.min_callback(value, 'min bright'))
        self.max_brightness_scale = Scale(self.brightness_frame, self.W_geometry, name='max bright', from_=0, to=60,
                                          resolution=1, command=lambda value: self.max_callback(value, 'max bright'))

        # Configuring noise frame
        self.noise_frame = Frame(self.param_set_frame, name='frame noise',
                                 width=self.frame_width, height=self.frame_height)
        self.noise_frame.pack_propagate(False)
        self.min_noise_scale = Scale(self.noise_frame, self.W_geometry, name='min noise', from_=0, to=20, resolution=1,
                                     command=lambda value: self.min_callback(value, 'min noise'))
        self.max_noise_scale = Scale(self.noise_frame, self.W_geometry, name='max noise', from_=0, to=20, resolution=1,
                                     command=lambda value: self.max_callback(value, 'max noise'))

        # Configuring blur frame
        self.blur_frame = Frame(self.param_set_frame, name='frame blur',
                                width=self.frame_width, height=self.frame_height)
        self.blur_frame.pack_propagate(False)
        self.min_blur_scale = Scale(self.blur_frame, self.W_geometry, name='min blur', from_=0, to=20, resolution=1,
                                    command=lambda value: self.min_callback(value, 'min blur'))
        self.max_blur_scale = Scale(self.blur_frame, self.W_geometry, name='max blur', from_=0, to=20, resolution=1,
                                    command=lambda value: self.max_callback(value, 'max blur'))

        # Configuring aberration frame
        self.aberration_frame = Frame(self.param_set_frame, width=self.frame_width, height=self.frame_height)
        self.aberration_frame.pack_propagate(False)
        self.V_aberration_scale = Scale(self.aberration_frame, self.W_geometry, name='v_aberration', from_=0, to=10,
                                        resolution=1, command=lambda x: self.both_callback())
        self.H_aberration_scale = Scale(self.aberration_frame, self.W_geometry, name='h_aberration', from_=0, to=10,
                                        resolution=1, command=lambda x: self.both_callback())

        # Configuring resize frame
        self.resize_frame = Frame(self.param_set_frame, name='frame resize',
                                  width=self.frame_width, height=self.frame_height)
        self.resize_frame.pack_propagate(False)
        self.min_resize_scale = Scale(self.resize_frame, self.W_geometry, name='min resize', from_=0, to=150,
                                      resolution=1, command=lambda value: self.min_callback(value, 'min resize'))
        self.max_resize_scale = Scale(self.resize_frame, self.W_geometry, name='max resize', from_=0, to=500,
                                      resolution=1, command=lambda value: self.max_callback(value, 'max resize'))

        # Configuring background enlargement frame
        self.enlarge_bg_frame = Frame(self.param_set_frame, width=self.frame_width, height=self.frame_height)
        self.enlarge_bg_frame.pack_propagate(False)
        self.V_max_enlargement_scale = Scale(self.enlarge_bg_frame, self.W_geometry, name='v_bg', from_=0, to=50,
                                             resolution=1, command=lambda x: self.calc_nr_of_samples_generated())
        self.H_max_enlargement_scale = Scale(self.enlarge_bg_frame, self.W_geometry, name='h_bg', from_=0, to=50,
                                             resolution=1, command=lambda x: self.calc_nr_of_samples_generated())

        # Configuring notes frame
        self.notes_frame = Frame(self.param_set_frame, width=self.frame_width, height=self.frame_height)
        self.notes_frame.pack_propagate(False)
        self.notes = "Note: Parameters that are set to 0 (at both minimum and maximum, where applicable) will not be " \
                     "applied."

        # Configuring buttons frame (SET DEFAULT, RESET and samples scale)
        self.buttons_frame = Frame(self.param_set_frame, width=self.frame_width, height=self.frame_height)
        self.buttons_frame.pack_propagate(False)
        self.generated_samples = 0
        self.nr_of_samples_scale = Scale(self.buttons_frame, self.W_geometry)
        self.reset_button = Button(self.buttons_frame, text="Reset", width=10, command=self.reset_callback)
        self.default_button = Button(self.buttons_frame, text="Set default", width=15, command=self.default_callback)

    # Function sets geometry of master
    def set_geometry(self):
        self._master.geometry(self._geometry)

    # Function sets the body of the window
    def body(self):
        template_path = glob.glob(self._param_dict["Path to templates"] + '/*.png')[0].replace('\\', '/')
        self.sampling_dict["Template path"] = template_path

        # Setting title of the frame
        Label(self.title_frame, text=self.title, font=self.title_font).pack(side=TOP, anchor=CENTER)
        self.title_frame.pack(side=TOP, pady=10)

        # Setting sampling frame
        self.apply_filters.body_pack(side=LEFT, image_path=self.sampling_dict['Template path'])

        Separator(self, orient=VERTICAL).pack(side=LEFT, fill=Y, padx=self.separator_pad/2)

        # Setting radial distortion and halo frame
        Label(self.distortion_n_halo_frame, text="Radial distortion:", width=13, anchor=W).pack(side=LEFT, anchor=S)
        self.radial_distortion_scale.pack(side=LEFT, anchor=S)
        self.halo_scale.pack(side=RIGHT, anchor=S)
        Label(self.distortion_n_halo_frame, text="Halo:", width=5, anchor=W).pack(side=RIGHT, anchor=S)
        self.distortion_n_halo_frame.pack(side=TOP)

        # Setting vertical perspective frame
        Label(self.V_perspective_frame, text="Vertical perspective:", width=30, anchor=W).pack(side=LEFT, anchor=S)
        Label(self.V_perspective_frame, text="Minimum:", width=10, anchor=W).pack(side=LEFT, anchor=S)
        self.min_V_perspective_scale.pack(side=LEFT, anchor=S)
        self.max_V_perspective_scale.pack(side=RIGHT, anchor=S)
        Label(self.V_perspective_frame, text="Maximum:", width=10, anchor=W).pack(side=RIGHT, anchor=S)
        self.V_perspective_frame.pack(side=TOP)

        # Setting horizontal perspective frame
        Label(self.H_perspective_frame, text="Horizontal perspective:", width=30, anchor=W).pack(side=LEFT, anchor=S)
        Label(self.H_perspective_frame, text="Minimum:", width=10, anchor=W).pack(side=LEFT, anchor=S)
        self.min_H_perspective_scale.pack(side=LEFT, anchor=S)
        self.max_H_perspective_scale.pack(side=RIGHT, anchor=S)
        Label(self.H_perspective_frame, text="Maximum:", width=10, anchor=W).pack(side=RIGHT, anchor=S)
        self.H_perspective_frame.pack(side=TOP)

        # Setting brightness frame
        Label(self.brightness_frame, text="Brightness:", width=30, anchor=W).pack(side=LEFT, anchor=S)
        Label(self.brightness_frame, text="Minimum:", width=10, anchor=W).pack(side=LEFT, anchor=S)
        self.min_brightness_scale.pack(side=LEFT, anchor=S)
        self.max_brightness_scale.pack(side=RIGHT, anchor=S)
        Label(self.brightness_frame, text="Maximum:", width=10, anchor=W).pack(side=RIGHT, anchor=S)
        self.brightness_frame.pack(side=TOP)

        # Setting noise frame
        Label(self.noise_frame, text="Noise:", width=30, anchor=W).pack(side=LEFT, anchor=S)
        Label(self.noise_frame, text="Minimum:", width=10, anchor=W).pack(side=LEFT, anchor=S)
        self.min_noise_scale.pack(side=LEFT, anchor=S)
        self.max_noise_scale.pack(side=RIGHT, anchor=S)
        Label(self.noise_frame, text="Maximum:", width=10, anchor=W).pack(side=RIGHT, anchor=S)
        self.noise_frame.pack(side=TOP)

        # setting blur frame
        Label(self.blur_frame, text="Blur:", width=30, anchor=W).pack(side=LEFT, anchor=S)
        Label(self.blur_frame, text="Minimum:", width=10, anchor=W).pack(side=LEFT, anchor=S)
        self.min_blur_scale.pack(side=LEFT, anchor=S)
        self.max_blur_scale.pack(side=RIGHT, anchor=S)
        Label(self.blur_frame, text="Maximum:", width=10, anchor=W).pack(side=RIGHT, anchor=S)
        self.blur_frame.pack(side=TOP)

        # Setting aberration frame
        Label(self.aberration_frame, text="Aberration:", width=30, anchor=W).pack(side=LEFT, anchor=S)
        Label(self.aberration_frame, text="Vertical:", width=10, anchor=W).pack(side=LEFT, anchor=S)
        self.V_aberration_scale.pack(side=LEFT, anchor=S)
        self.H_aberration_scale.pack(side=RIGHT, anchor=S)
        Label(self.aberration_frame, text="Horizontal:", width=10, anchor=W).pack(side=RIGHT, anchor=S)
        self.aberration_frame.pack(side=TOP)

        # Setting resize frame
        Label(self.resize_frame, text="Resize:", width=30, anchor=W).pack(side=LEFT, anchor=S)
        Label(self.resize_frame, text="Minimum:", width=10, anchor=W).pack(side=LEFT, anchor=S)
        self.min_resize_scale.pack(side=LEFT, anchor=S)
        self.max_resize_scale.pack(side=RIGHT, anchor=S)
        Label(self.resize_frame, text="Maximum", width=10, anchor=W).pack(side=RIGHT, anchor=S)
        self.resize_frame.pack(side=TOP)

        # Setting background enlargement frame
        Label(self.enlarge_bg_frame, text="Maximum background enlargement:", width=30, anchor=W).pack(side=LEFT,
                                                                                                      anchor=S)
        Label(self.enlarge_bg_frame, text="Vertical:", width=10, anchor=W).pack(side=LEFT, anchor=S)
        self.V_max_enlargement_scale.pack(side=LEFT, anchor=S)
        self.H_max_enlargement_scale.pack(side=RIGHT, anchor=S)
        Label(self.enlarge_bg_frame, text="Horizontal:", width=10, anchor=W).pack(side=RIGHT, anchor=S)
        self.enlarge_bg_frame.pack(side=TOP)

        # Setting notes frame
        Label(self.notes_frame, text=self.notes, fg='gray').pack(side=BOTTOM, anchor=W)
        self.notes_frame.pack(side=TOP)

        # Setting SET DEFAULT, RESET and samples scale buttons frame
        self.default_button.pack(side=LEFT, anchor=S)
        self.reset_button.pack(side=LEFT, anchor=S, padx=35)
        self.calc_nr_of_samples_generated()
        self.nr_of_samples_scale.pack(side=RIGHT, anchor=S)
        Label(self.buttons_frame, text="Samples generated per class:", anchor=W).pack(side=RIGHT, anchor=S)
        self.buttons_frame.pack(side=TOP)

        self.param_set_frame.pack(side=LEFT, fill=Y)

        self._body_pack_checker = True

    # Function verifies if body() was called before and packs up the frame
    def body_pack(self):
        self.set_geometry()

        if self._body_pack_checker is False:
            self.body()

        self.pack(side=TOP)

    # Callback function for the RESET button, setting all scales to 0(zero)
    def reset_callback(self):
        self.min_resize_scale.set(0)
        self.max_resize_scale.set(0)
        self.min_V_perspective_scale.set(0)
        self.max_V_perspective_scale.set(0)
        self.min_H_perspective_scale.set(0)
        self.max_H_perspective_scale.set(0)
        self.min_brightness_scale.set(0)
        self.max_brightness_scale.set(0)
        self.min_noise_scale.set(0)
        self.max_noise_scale.set(0)
        self.min_blur_scale.set(0)
        self.max_blur_scale.set(0)
        self.V_aberration_scale.set(0)
        self.H_aberration_scale.set(0)
        self.V_max_enlargement_scale.set(0)
        self.H_max_enlargement_scale.set(0)
        self.radial_distortion_scale.set(0.0)
        self.halo_scale.set(0)

    # Callback function for the SET DEFAULT button, setting all scales to values defined in default configuration file
    def default_callback(self):
        if self._param_dict:
            self.min_resize_scale.set(self._param_dict["Min resize"])
            self.max_resize_scale.set(self._param_dict["Max resize"])
            self.min_V_perspective_scale.set(self._param_dict["Min V perspective"])
            self.max_V_perspective_scale.set(self._param_dict["Max V perspective"])
            self.min_H_perspective_scale.set(self._param_dict["Min H perspective"])
            self.max_H_perspective_scale.set(self._param_dict["Max H perspective"])
            self.min_brightness_scale.set(self._param_dict["Min light"])
            self.max_brightness_scale.set(self._param_dict["Max light"])
            self.min_noise_scale.set(self._param_dict["Min noise value"])
            self.max_noise_scale.set(self._param_dict["Max noise value"])
            self.min_blur_scale.set(self._param_dict["Min blur amplitude"])
            self.max_blur_scale.set(self._param_dict["Max blur amplitude"])
            self.V_aberration_scale.set(self._param_dict["Vertical max aberration"])
            self.H_aberration_scale.set(self._param_dict["Horizontal max aberration"])
            self.radial_distortion_scale.set(self._param_dict["Radial distortion"])
            self.halo_scale.set(self._param_dict["Halo amount"])
            self.V_max_enlargement_scale.set(self._param_dict["Max enlarge background vertical"])
            self.H_max_enlargement_scale.set(self._param_dict["Max enlarge background horizontal"])

    # Function returns the distortion center used for the fish eye effect. Based on dims of template images
    def calc_distortion_center(self):
        # List containing template images
        images_path_list = glob.glob(self._param_dict["Path to templates"] + '/*.png')

        # Lists to collect dims of the images
        widths = []
        heights = []
        for image_path in images_path_list:
            image = Image.open(image_path)
            widths.append(image.size[0])
            heights.append(image.size[1])

        x_center_distortion = random.randint(1, min(widths))
        y_center_distortion = random.randint(1, min(heights))
        return x_center_distortion, y_center_distortion

    # Callback function for parameters that apply only to minimum sampling images
    def min_callback(self, value=None, name=None):
        self.correlate_buttons_by_min(value=value, name=name)
        self.sampling_dict['Which'] = -1
        self.write_sampling_configuration()
        self.apply_filters.sampling_filters(self.config_file_path)
        self.calc_nr_of_samples_generated()

    # Function correlates a max button to a min value when min is higher than max
    def correlate_buttons_by_min(self, value=None, name=None):
        max_button_name = name.replace('min', 'max')
        frame_name = name.replace('min', 'frame')
        path_to_button = '.' + self.winfo_name() + '.param_set.' + frame_name + '.' + max_button_name

        if value.find('.'):
            value = float(value)
        else:
            value = int(value)

        if value > self.nametowidget(path_to_button).get():
            self.nametowidget(path_to_button).set(value)

    # Callback function for parameters that apply only to maximum images
    def max_callback(self, value=None, name=None):
        self.correlate_buttons_by_max(value=value, name=name)
        self.sampling_dict['Which'] = 1
        self.write_sampling_configuration()
        self.apply_filters.sampling_filters(self.config_file_path)
        self.calc_nr_of_samples_generated()

    # Function correlates a min button to a max value when max is lower than min
    def correlate_buttons_by_max(self, value=None, name=None):
        max_button_name = name.replace('max', 'min')
        frame_name = name.replace('max', 'frame')
        path_to_button = '.' + self.winfo_name() + '.param_set.' + frame_name + '.' + max_button_name

        if value.find('.'):
            value = float(value)
        else:
            value = int(value)

        if value < self.nametowidget(path_to_button).get():
            self.nametowidget(path_to_button).set(value)

    # Callback function for parameters that apply to both sampling images
    def both_callback(self):
        self.sampling_dict['Which'] = 0
        self.write_sampling_configuration()
        self.apply_filters.sampling_filters(self.config_file_path)
        self.calc_nr_of_samples_generated()

    # Function overwrites the sampling configuration file
    def write_sampling_configuration(self):
        import os

        if not os.path.exists(self.misc_dir_path):
            os.makedirs(self.misc_dir_path)

        yaml_file = open(self.config_file_path, 'w')

        yaml_file.write("%YAML 1.0\n")
        yaml_file.write("---")

        yaml_file.write("\nPaths:\n")
        yaml_file.write("    Template path: \"{:}\"\n".format(self.sampling_dict["Template path"]))
        yaml_file.write("    Output path: \"{:}\"\n".format(self.sampling_dict["Output path"]))

        yaml_file.write("\nWhich: {:}\n".format(self.sampling_dict["Which"]))
        self.sampling_dict["Counter"] = self.sampling_dict["Counter"] + 1
        yaml_file.write("\nCounter: {:}\n".format(self.sampling_dict["Counter"]))

        yaml_file.write("\nMinimum:\n")
        yaml_file.write("    Resize: {:}\n".format(self.min_resize_scale.get()))
        yaml_file.write("    V perspective: {:}\n".format(self.min_V_perspective_scale.get()))
        yaml_file.write("    H perspective: {:}\n".format(self.min_H_perspective_scale.get()))
        yaml_file.write("    Brightness: {:}\n".format(self.min_brightness_scale.get()))
        yaml_file.write("    Noise: {:}\n".format(self.min_noise_scale.get()))
        yaml_file.write("    Blur: {:}\n".format(self.min_blur_scale.get()))

        yaml_file.write("\nMaximum:\n")
        yaml_file.write("    Resize: {:}\n".format(self.max_resize_scale.get()))
        yaml_file.write("    V perspective: {:}\n".format(self.max_V_perspective_scale.get()))
        yaml_file.write("    H perspective: {:}\n".format(self.max_H_perspective_scale.get()))
        yaml_file.write("    Brightness: {:}\n".format(self.max_brightness_scale.get()))
        yaml_file.write("    Noise: {:}\n".format(self.max_noise_scale.get()))
        yaml_file.write("    Blur: {:}\n".format(self.max_blur_scale.get()))

        if self.radial_distortion_scale.get() is not 0:
            if self.distortion_center_y is 0 and self.distortion_center_y is 0:
                self.distortion_center_x, self.distortion_center_y = self.calc_distortion_center()
        else:
            self.distortion_center_x = 0
            self.distortion_center_y = 0
        yaml_file.write("\nOther:\n")
        yaml_file.write("    V aberration: {:}\n".format(self.V_aberration_scale.get()))
        yaml_file.write("    H aberration: {:}\n".format(self.H_aberration_scale.get()))
        yaml_file.write("    Radial distortion: {:}\n".format(self.radial_distortion_scale.get()))
        yaml_file.write("    X center distortion: {:}\n".format(self.distortion_center_x))
        yaml_file.write("    Y center distortion: {:}\n".format(self.distortion_center_y))
        yaml_file.write("    Halo: {:}\n".format(self.halo_scale.get()))

        yaml_file.close()

    # Function calculates the number of samples that will be generated
    def calc_nr_of_samples_generated(self):
        if self.min_H_perspective_scale.get() == 0.0 and self.max_H_perspective_scale.get() == 0.0:
            h_perspective_cycles = 1
        else:
            h_perspective_cycles = NUMBER_OF_PERSPECTIVES

        if self.min_V_perspective_scale.get() == 0.0 and self.max_V_perspective_scale.get() == 0.0:
            v_perspective_cycles = 1
        else:
            v_perspective_cycles = NUMBER_OF_PERSPECTIVES

        if self.min_blur_scale.get() == 0 and self.max_blur_scale.get() == 0:
            blur_cycles = 1
        else:
            blur_cycles = self.max_blur_scale.get() - self.min_blur_scale.get() + 1

        self.generated_samples = NUMBER_OF_ILLUMINATIONS * h_perspective_cycles * v_perspective_cycles * blur_cycles

        if self.generated_samples < 0:
            self.nr_of_samples_scale.config(label='N/A', from_=0, to=0)

        else:
            self.nr_of_samples_scale.config(from_=self.generated_samples, to=(self.generated_samples * 5), label='',
                                            resolution=self.generated_samples)
            self.nr_of_samples_scale.set(self.generated_samples)

    # Functions passes configuration into parameter _dict
    def get_parameters(self, _dict):
        _dict["Min resize"] = self.min_resize_scale.get()
        _dict["Max resize"] = self.max_resize_scale.get()
        _dict["Min H perspective"] = self.min_H_perspective_scale.get()
        _dict["Max H perspective"] = self.max_H_perspective_scale.get()
        _dict["Min V perspective"] = self.min_V_perspective_scale.get()
        _dict["Max V perspective"] = self.max_V_perspective_scale.get()
        _dict["Min light"] = self.min_brightness_scale.get()
        _dict["Max light"] = self.max_brightness_scale.get()
        _dict["Min noise value"] = self.min_noise_scale.get()
        _dict["Max noise value"] = self.max_noise_scale.get()
        _dict["Min blur amplitude"] = self.min_blur_scale.get()
        _dict["Max blur amplitude"] = self.max_blur_scale.get()
        _dict["Vertical max aberration"] = self.V_aberration_scale.get()
        _dict["Horizontal max aberration"] = self.H_aberration_scale.get()
        _dict["Radial distortion"] = self.radial_distortion_scale.get()
        if _dict["Radial distortion"] == 0:
            _dict["X center distortion"] = 0
            _dict["Y center distortion"] = 0
        else:
            _dict["X center distortion"], _dict["Y center distortion"] = self.calc_distortion_center()
        _dict["Halo amount"] = self.halo_scale.get()
        _dict["Max enlarge background vertical"] = self.V_max_enlargement_scale.get()
        _dict["Max enlarge background horizontal"] = self.H_max_enlargement_scale.get()
        _dict["Number of samples per class"] = self.generated_samples
        _dict["Number of generations"] = self.nr_of_samples_scale.get() / self.generated_samples

    # Function calls Apply_filters_frame 'delete_misc_content' function, passing its on files to delete
    def delete_misc(self):
        self.apply_filters.delete_misc_content(other_files=self.config_file_path)


if __name__ == "__main__":
    from tkinter import Tk

    root = Tk()
    _dict = {"Path to templates": "D:/dev/data/ASG/Input/Templates/Unique_signs",
             "Min resize": 0,
             "Max resize": 0,
             "Min V perspective": 0,
             "Max V perspective": 0,
             "Min H perspective": 0,
             "Max H perspective": 0,
             "Min light": 0,
             "Max light": 0,
             "Min noise value": 0,
             "Max noise value": 0,
             "Min blur amplitude": 0,
             "Max blur amplitude": 0,
             "Vertical max aberration": 0,
             "Horizontal max aberration": 0,
             "Radial distortion": 0,
             "X center distortion": 0,
             "Y center distortion": 0,
             "Halo amount": 0,
             "Max enlarge background horizontal": 0,
             "Max enlarge background vertical": 0}

    param_frame = ParametersWindow(master=root, param_dict=_dict)
    param_frame.body_pack()

    root.mainloop()
