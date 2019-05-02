import os
from tkinter import Frame, StringVar, Entry, Label, Button, messagebox
from tkinter import filedialog
from tkinter import TOP, LEFT, RIGHT, S, W, CENTER, NORMAL, DISABLED
import glob
from PIL import Image


# Class handles first window frame, asking for needed paths
class PathsWindow(Frame):
    def __init__(self, master=None, name=None, button=None):
        Frame.__init__(self, master=master, name=name)

        self._master = master
        self._button = button
        self._name = name

        # Variable keeps the track regarding whether the body() was called before or not
        self._body_pack_checker = False

        self.etc_dir_path = '../etc'
        self.backgrounds_name_txt_file_path = '/backgrounds.txt'

        width = 750
        height = 500
        x_offset = int((self._master.winfo_screenwidth() - width) / 2)
        y_offset = int((self._master.winfo_screenheight() - height - 50) / 2)
        self._geometry = "{:d}x{:d}+{:d}+{:d}".format(width, height, x_offset, y_offset)

        self.frame_width = width - 110
        self.frame_height = height / 4

        self.error_font = ('Segoe UI', 12)

        # Configuring error messages space
        self.error_frame = Frame(self, width=self.frame_width, height=self.frame_height)
        self.error_frame.pack_propagate(False)
        self.templates_error = StringVar()
        self.outdir_error = StringVar()
        self.base_path_error = StringVar()
        # self.backgrounds_error = StringVar()  # Not used for now. May be useful in future development

        # Configuring entries space
        self.input_frame = Frame(self, width=self.frame_width, height=self.frame_height)

        # base path entry row
        self.in_base_row = Frame(self.input_frame, width=self.frame_width, height=self.frame_height / 5)
        self.in_base_row.pack_propagate(False)
        self.base_entry = Entry(self.in_base_row, width=70, validatecommand=self.validate_regularization_path)

        # berkley path entry row
        self.in_berkley_row = Frame(self.input_frame, width=self.frame_width, height=self.frame_height / 5)
        self.in_berkley_row.pack_propagate(False)
        self.berkley_entry = Entry(self.in_berkley_row, width=70, validatecommand=self.validate_regularization_path)

        self.opt_row = Frame(self.input_frame, width=self.frame_width, height=self.frame_height/5)
        self.opt_row.pack_propagate(False)

        # Configuring warning messages space
        self.warning_frame = Frame(self, width=self.frame_width, height=self.frame_height/2)
        self.warning_frame.pack_propagate(False)
        self.templates_warning = StringVar()
        self.backgrounds_warning = StringVar()
        # self.outdir_warning = StringVar()  # Not used for now. May be useful in future development

        # Configuring output info space
        self.output_frame = Frame(self, width=self.frame_width, height=self.frame_height)

        # templates info output
        self.out_templates_row = Frame(self.output_frame, width=self.frame_width, height=self.frame_height/4)
        self.out_templates_row.pack_propagate(False)
        self.templates_found = StringVar()

        # classes info output
        self.classes_row = Frame(self.output_frame, width=self.frame_width, height=self.frame_height/4)
        self.classes_row.pack_propagate(False)
        self.classes_found = StringVar()

        # background info output
        self.out_backgrounds_row = Frame(self.output_frame, width=self.frame_width, height=self.frame_height/4)
        self.out_backgrounds_row.pack_propagate(False)
        self.backgrounds_found = StringVar()

        # Configuring other info output space
        self.info_row = Frame(self.output_frame, width=self.frame_width, height=self.frame_height/4)
        self.info_row.pack_propagate(False)
        self.notes = "Note: If path to backgrounds is not given, samples will be generated without.\n" + \
                     "If path to regularization samples is not given, Pareto optimization cannot be performed"

        # Status variables for activating NEXT button
        # self.backgrounds_status = False   # May be useful in future development
        self.templates_status = False
        self.outdir_status = False
        self.base_status = False

    # Function sets geometry of master
    def set_geometry(self):
        self._master.geometry(self._geometry)

    # Function sets the body of the window
    def body(self):
        # Setting error messages space
        self.templates_error.set('')
        Label(self.error_frame, textvariable=self.templates_error, justify=LEFT, wraplength=self.frame_width, fg='red',
              font=self.error_font).pack(side=LEFT, anchor=CENTER)
        self.outdir_error.set('')
        Label(self.error_frame, textvariable=self.outdir_error, justify=LEFT, wraplength=self.frame_width, fg='red',
              font=self.error_font).pack(side=LEFT, anchor=CENTER)
        Label(self.error_frame, textvariable=self.base_path_error, justify=LEFT, wraplength=self.frame_width, fg='red',
              font=self.error_font).pack(side=LEFT, anchor=CENTER)
        self.error_frame.pack(side=TOP)
        # self.backgrounds_error.set('')    # May be useful in future development
        # Label(self.error_frame, textvariable=self.backgrounds_error, justify=LEFT, wraplength=self.frame_width,
        #       fg='red', font=self.error_font).pack(side=LEFT, anchor=CENTER)

        # Setting entries space

        # base path entry
        Label(self.in_base_row, text="Base path:", width=15, anchor=W).pack(side=LEFT, anchor=CENTER)
        self.base_entry.pack(side=LEFT, anchor=CENTER, padx=20)
        Button(self.in_base_row, text='Browse', padx=5, command=self.browse_base_callback) \
            .pack(side=RIGHT, anchor=CENTER)
        self.in_base_row.pack(side=TOP)

        # berkley path entry
        Label(self.in_berkley_row, text="Berkley path:", width=15, anchor=W).pack(side=LEFT, anchor=CENTER)
        self.berkley_entry.pack(side=LEFT, anchor=CENTER, padx=20)
        Button(self.in_berkley_row, text='Browse', padx=5, command=self.browse_berkley_callback) \
            .pack(side=RIGHT, anchor=CENTER)
        self.in_berkley_row.pack(side=TOP)

        # optimization
        Button(self.opt_row, text='Optimize', padx=5, command=self.optimize_callback).pack(side=LEFT, anchor=CENTER)
        self.opt_row.pack(side=TOP)

        self.input_frame.pack(side=TOP)

        # Setting warning messages space
        self.templates_warning.set('')
        Label(self.warning_frame, textvariable=self.templates_warning, justify=LEFT, wraplength=self.frame_width,
              fg='gray').pack(side=LEFT, anchor=CENTER)
        self.backgrounds_warning.set('')
        Label(self.warning_frame, textvariable=self.backgrounds_warning, justify=LEFT, wraplength=self.frame_width,
              fg='gray').pack(side=LEFT, anchor=CENTER)
        # self.outdir_warning.set('')   # May be useful in future development
        # Label(self.warning_frame, textvariable=self.outdir_warning, justify=LEFT, wraplength=self.frame_width,
        #       fg='gray').pack(side=LEFT, anchor=CENTER)
        self.warning_frame.pack(side=TOP)

        # Setting output info space
        # templates info
        self.out_templates_row.pack(side=TOP)
        Label(self.out_templates_row, text="Template images found:", width=20, anchor=W).pack(side=LEFT, anchor=CENTER)
        self.templates_found.set('')
        Label(self.out_templates_row, textvariable=self.templates_found, anchor=W)\
            .pack(side=LEFT, anchor=CENTER, padx=10)

        # classes info
        self.classes_row.pack(side=TOP)
        Label(self.classes_row, text="Classes found:", width=20, anchor=W).pack(side=LEFT, anchor=CENTER)
        self.classes_found.set('')
        Label(self.classes_row, textvariable=self.classes_found, wraplength=(self.frame_width-20), justify=LEFT,
              anchor=W).pack(side=LEFT, anchor=CENTER, padx=10)

        # backgrounds info
        self.out_backgrounds_row.pack(side=TOP)
        Label(self.out_backgrounds_row, text="Background images found:", width=20, anchor=W)\
            .pack(side=LEFT, anchor=CENTER)
        self.backgrounds_found.set('')
        Label(self.out_backgrounds_row, textvariable=self.backgrounds_found, anchor=W)\
            .pack(side=LEFT, anchor=CENTER, padx=10)

        # other info
        self.info_row.pack(side=TOP)
        Label(self.info_row, text=self.notes, fg='gray', anchor=W).pack(side=LEFT, anchor=S)

        self.output_frame.pack(side=TOP)

        self._body_pack_checker = True

    # Function verifies if body() was called before and packs up the frame
    def body_pack(self):
        self.set_geometry()
        if self._body_pack_checker is False:
            self.body()
        self.pack(side=TOP)

    def browse_regularization_callback(self):
        reg_path = filedialog.askdirectory()
        if reg_path:
            self.reg_entry.delete(0, 'end')
        self.reg_entry.insert('end', reg_path)
        self.validate_regularization_path()

    @staticmethod
    def error_messagebox(error_message):
        messagebox.showerror("Error", error_message)

    @staticmethod
    def get_file_name(file):
        return os.path.splitext(os.path.basename(file))[0]

    @staticmethod
    def get_folder_name(folder):
        return os.path.basename(folder)

    # Checks if every evaluation sample has a corresponding ground truth annotations file
    def check_gt_and_eval_data_correspondence(self, gt_data, eval_data):
        if len(gt_data) != len(eval_data):
            return False
        for i in range(0, len(gt_data)):
            if self.get_file_name(gt_data[i]) != self.get_file_name(eval_data[i]):
                return False
        return True

    def check_templates_and_reg_folders(self, base_path):
        templates_folder = os.path.join(base_path, "templates")
        regularization_folder = os.path.join(base_path, "regularization")
        templates_classes_dirs = [name for name in os.listdir(templates_folder) if
                                  os.path.isdir(os.path.join(templates_folder, name))]
        if len(templates_classes_dirs) == 0:
            self.error_messagebox("There are no templates!")
            return False
        for folder in templates_classes_dirs:
            templates = [f for f in glob.glob(os.path.join(templates_folder, folder) + "/*.png")]
            if len(templates) == 0:
                self.error_messagebox("No templates for class " + folder + "!")
                return False
            label = self.get_folder_name(folder)
            label_path = os.path.join(regularization_folder, label)
            regularization_templates = [f for f in glob.glob(label_path + "/*.png")]
            if not os.path.isdir(label_path) or len(regularization_templates) == 0:
                self.error_messagebox("There are no corresponding regularization images for class" + folder + "!")
                return False
        return True

    def check_model_data_folder(self, base_path):
        model_data_folder = os.path.join(base_path, "model_data")
        if not os.path.isfile(os.path.join(model_data_folder, "yolov3-tiny.weights")):
            self.error_messagebox("Tiny YOLO weights are missing!")
            return False
        if not os.path.isfile(os.path.join(model_data_folder, "tiny_yolo_anchors.txt")):
            self.error_messagebox("Tiny YOLO anchors are missing!")
            return False
        return True

    def check_evaluation_folder(self, base_path):
        eval_folder = os.path.join(base_path, "evaluation")
        if not os.path.isfile(os.path.join(eval_folder, "classes.txt")):
            self.error_messagebox("Classes file is missing!")
            return False
        if not os.path.isfile(os.path.join(eval_folder, "annotation_file.txt")):
            self.error_messagebox("Annotation file is missing!")
            return False
        eval_data_folder = os.path.join(eval_folder, "data")
        eval_data = [f for f in glob.glob(eval_data_folder + "/*.png")]
        if not eval_data_folder or len(eval_data) == 0:
            self.error_messagebox("Evaluation data is missing!")
            return False
        gt_folder = os.path.join(eval_folder, "ground-truth")
        gt_data = [f for f in glob.glob(gt_folder + "/*.txt")]
        if not gt_folder or len(gt_data) == 0:
            self.error_messagebox("Ground truth data is missing!")
            return False
        if not os.path.join(eval_folder, "predicted"):
            self.error_messagebox("Prediction folder is missing!")
            return False
        if not self.check_gt_and_eval_data_correspondence(gt_data, eval_data):
            self.error_messagebox("Invalid ground truth - data correspondence!")
            return False
        return True

    def check_base_folder_structure(self):
        base_path = self.base_entry.get()
        templates_folder = os.path.isdir(os.path.join(base_path, "templates"))
        regularization_folder = os.path.isdir(os.path.join(base_path, "regularization"))
        backgrounds_folder = os.path.isdir(os.path.join(base_path, "backgrounds"))
        output_folder = os.path.isdir(os.path.join(base_path, "output"))
        model_data_folder = os.path.isdir(os.path.join(base_path, "model_data"))
        eval_folder = os.path.isdir(os.path.join(base_path, "evaluation"))

        if not templates_folder:
            self.error_messagebox("Templates folder is missing!")
            return False
        if not regularization_folder:
            self.error_messagebox("Regularization folder is missing!")
            return False
        if not backgrounds_folder:
            self.error_messagebox("Backgrounds folder is missing!")
            return False
        if not output_folder:
            self.error_messagebox("Output folder is missing!")
            return False
        if not model_data_folder:
            self.error_messagebox("Model data folder is missing!")
            return False
        if not eval_folder:
            self.error_messagebox("Evaluation folder is missing!")
            return False
        if not self.check_model_data_folder(base_path):
            return False
        if not self.check_templates_and_reg_folders(base_path):
            return False
        # if not self.check_evaluation_folder(base_path):
        #     return False
        return True

    def optimize_callback(self):
        if self.check_base_folder_structure():
            self._master.show_pareto_window()

    # Callback function asking for a folder as a path to templates
    def browse_templates_callback(self):
        templates_path = filedialog.askdirectory()
        if templates_path:
            self.templates_entry.delete(0, 'end')
        self.templates_entry.insert('end', templates_path)
        self.validate_templates_path()

    def validate_regularization_path(self):
        pass

    # Function verifies templates path in order to contain at least one usable image. If none, sets an error message
    def validate_templates_path(self):
        del self.class_names[:]
        self.classes_string = ''

        if self.templates_entry.get() == '':
            self.templates_error.set('')
            self.templates_warning.set('')
            self.templates_status = False
        else:
            folder_list = glob.glob(self.templates_entry.get() + '/*.png')

            for item in folder_list:
                start = item.rfind('\\') + 1
                end = item.rfind('.png')
                self.class_names.append(item[start: end])
                self.classes_string += item[start: end] + ', '

            self.classes_string = self.classes_string[0: len(self.classes_string) - 2]
            self.templates_found.set(len(self.class_names))
            self.classes_found.set(self.classes_string)

            if not self.class_names:
                self.templates_warning.set('')
                self.templates_error.set("Error: Selected templates folder contains no usable images!")
                self.templates_status = False
            elif len(self.class_names) is 1:
                self.templates_error.set('')
                self.templates_warning.set("Warning: Selected templates folder contains a single image.")
                self.validate_template_images()
            else:
                self.templates_error.set('')
                self.templates_warning.set('')
                self.validate_template_images()

            self.activate_button()

    # Function verifies if contained images are of the required type
    def validate_template_images(self):
        error_message = "Error: Following images don't respect type requirements!\n"

        names_counter = 0
        for name in self.class_names:
            img = Image.open(self.templates_entry.get() + '/' + name + '.png')
            if img.mode != 'RGBA' and img.mode != 'RGB' and img.mode != 'P':
                error_message += '{}, '.format(name)

                names_counter += 1
                if names_counter is len(self.class_names):
                    end = len(error_message) - 2
                    error_message = error_message[0: end]

                self.templates_error.set(error_message)
                self.templates_status = False
            else:
                self.templates_status = True

    # Callback function asking for a folder as a path to backgrounds
    def browse_bg_callback(self):
        backgrounds_path = filedialog.askdirectory()
        if backgrounds_path:
            self.backgrounds_entry.delete(0, 'end')
        self.backgrounds_entry.insert('end', backgrounds_path)
        self.validate_backgrounds_path()
        self.activate_button()

    # Function verifies backgrounds path in order to contain usable images. If none, sets a warning message
    def validate_backgrounds_path(self):
        folder_list = glob.glob(self.backgrounds_entry.get() + '/*.png')

        self.backgrounds_found.set(len(folder_list))

        if int(self.backgrounds_found.get()) == 0:
            # self.backgrounds_error.set('')    # May be useful in future development
            self.backgrounds_warning.set('')
            self.backgrounds_warning.set("Warning: Selected background folder contains no usable images. "
                                         "Generated samples will have no background. "
                                         "If that's not what you intend, please provide another folder")
            # self.backgrounds_status = False    # May be useful in future development

        else:
            self.write_txt_bg_names(folder_list)
            # self.backgrounds_status = True    # May be useful in future development

    # Function passes names of the background images into a text file from which generator will choose one
    def write_txt_bg_names(self, images_list):
        file = open(self.etc_dir_path + self.backgrounds_name_txt_file_path, 'w')

        for image_name in images_list:
            image_name = image_name[image_name.find('\\'):]
            file.write("{:}\n".format(image_name.replace('\\', '/')))

    # Callback function asking for a folder as base path
    def browse_base_callback(self):
        base_path = filedialog.askdirectory()
        if base_path:
            self.base_entry.delete(0, 'end')
        self.base_entry.insert('end', base_path)

    def browse_berkley_callback(self):
        berkley_path = filedialog.askdirectory()
        if berkley_path:
            self.berkley_entry.delete(0, 'end')
        self.berkley_entry.insert('end', berkley_path)

    def validate_base_path(self):
        folder_path = self.base_entry.get()
        end = folder_path.rfind('/')
        folder_name = folder_path[end + 1:]
        folder_parent = folder_path[:end]
        if self.base_entry.get() is '':
            self.base_status = False
        else:
            parent = glob.glob(folder_parent + '/*')
            folder_path = os.path.join(folder_parent, folder_name)
            for item in parent:
                if folder_path == item:
                    self.base_status = True
                    break
            if not self.base_status:
                # TODO show error message in frame
                self.base_path_error.set('')
                self.base_path_error.set("Error: Folder '{}' doesn't exist! Provide a valid path!".format(folder_name))
                # print("Error: Folder '{}' doesn't exist! Provide a valid path!".format(folder_name))

    # Callback function asking for a folder as an output path
    def browse_outdir_callback(self):
        outdir_path = filedialog.askdirectory()
        if outdir_path:
            self.outdir_entry.delete(0, 'end')
        self.outdir_entry.insert('end', outdir_path)
        self.validate_outdir_path()
        self.activate_button()

    # Function verifies existence of output directory. If none, sets an error message
    def validate_outdir_path(self):
        folder_path = self.outdir_entry.get()
        end = folder_path.rfind('/')
        folder_name = folder_path[end + 1:]
        folder_parent = folder_path[:end]

        if self.outdir_entry.get() is '':
            self.outdir_status = False
        else:
            parent = glob.glob(folder_parent + '/*')
            folder_path = folder_parent + '\\' + folder_name
            for item in parent:
                if folder_path == item:
                    self.outdir_status = True
                    break
            if not self.outdir_status:
                # self.outdir_warning.set('')   # May be useful in future development
                self.outdir_error.set('')
                self.outdir_error.set("Error: Folder '{}' doesn't exist! Provide a valid path!".format(folder_name))

    # Function verifies the validation statuses of the entries and, if all are valid, activates NEXT button
    def activate_button(self):
        if self._button is not None:
            if self.templates_status and self.outdir_status:
                self._button.config(state=NORMAL)
            else:
                self._button.config(state=DISABLED)

    # Pass base path
    def get_base_path(self):
        return self.base_entry.get()

    def get_berkley_path(self):
        return self.berkley_entry.get()

    # Functions passes paths into parameter _dict
    def get_paths(self, _dict):
        _dict["Path to templates"] = self.templates_entry.get()
        _dict["Nr of templates in folder"] = self.templates_found.get()
        _dict["Path to backgrounds"] = self.backgrounds_entry.get()
        _dict["Nr of backgrounds in folder"] = self.backgrounds_found.get()
        _dict["Background names file"] = self.etc_dir_path + self.backgrounds_name_txt_file_path
        _dict["Output path"] = self.outdir_entry.get()
        _dict["Regularization path"] = self.reg_entry.get()
        _dict["Signs"] = self.class_names


if __name__ == '__main__':
    from tkinter import Tk

    root = Tk()

    paths_window = PathsWindow(root)
    paths_window.body_pack()

    root.mainloop()
