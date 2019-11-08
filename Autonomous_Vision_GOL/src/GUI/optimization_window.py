import _thread
import os
from tkinter import TOP, LEFT, RIGHT, BOTTOM, S, W, E, CENTER, Y, VERTICAL
from tkinter import Frame, Scale, Label, Button, Entry, Canvas
from generalization_energy import ParetoOptimizer, run_optimization
from plot_functions import ParetoPlotter
import shutil


class OptimizationWindow(Frame):
    def __init__(self, master=None, name=None):
        Frame.__init__(self, master=master, name=name)
        self._master = master
        self._name = name
        # Variable for keeping track regarding whether the body() was called before or not
        self._body_pack_checker = False

        self.misc_dir_path = "../bin/misc"
        self.config_file_path = self.misc_dir_path + "/sampling_config.yml"
        self.win_width = 1500
        self.win_height = 950
        self.sampling_width = 450
        self.padding_x = 160
        self.separator_pad = 50
        self.param_width = self.win_width - self.sampling_width - self.separator_pad
        self.x_offset = int((self.winfo_screenwidth() - self.win_width) / 2)
        self.y_offset = int((self.winfo_screenheight() - self.win_height - 50) / 2)
        self._geometry = "{:d}x{:d}+{:d}+{:d}".format(self.win_width, self.win_height, self.x_offset, self.y_offset)

        self.frame_width = self.param_width - self.padding_x
        self.frame_height = self.win_height / 13

        self.config(width=self.win_width - self.padding_x, height=self.win_height - self.frame_height)
        self.pack_propagate(False)
        self.title = "Pareto Optimization"
        self.title_font = ('Segoe UI', 12)
        self.title_frame = Frame(self, width=self.frame_width, height=self.frame_height)
        self.image_step_frame = Frame(self, width=self.frame_width, height=self.frame_height)
        self.epochs_frame = Frame(self, width=self.frame_width, height=self.frame_height)
        self.optimize_parameters_frame = Frame(self, width=1200, height=600)

        self.pareto_plot_frame = Frame(self, width=1200, height=800)
        self.drawing_canvas = Canvas(self.pareto_plot_frame, width=1200, height=800)

        self.W_geometry = {'orient': 'horizontal', 'length': 200}

        self.image_step = 1
        self.num_epochs = 1

        self.base_path = ""
        self.berkley_path = ""

        self.template_path = ""
        self.output_path = ""
        self.regularization_path = ""
        self.background_path = ""

        self.image_step_scale = Scale(self.image_step_frame, self.W_geometry, name='img_step', from_=1, to=100,
                                      resolution=1, command=lambda value: self.adjust_image_step(value))
        self.epochs_scale = Scale(self.epochs_frame, self.W_geometry, name="epochs", from_=1, to=10000,
                                  resolution=10, command=lambda value: self.adjust_num_epochs(value))
        self.optimize_button = Button(self.optimize_parameters_frame,
                                      text="Optimize",
                                      width=10,
                                      command=self.optimizer_thread)
        self.download_model_frame = Frame(self, width=self.frame_width, height=self.frame_height)
        self.download_model_button = Button(self.download_model_frame,
                                            text="Download model",
                                            width=10,
                                            command=self.download_model,
                                            padx=50)

    def adjust_image_step(self, value):
        self.image_step = value

    def adjust_num_epochs(self, value):
        self.num_epochs = value

    def body_pack(self):
        self.set_geometry()
        if self._body_pack_checker is False:
            self.body()
        self.pack(side=TOP)

    # Function sets the body of the window
    def body(self):
        # Setting title of the frame
        Label(self.title_frame, text=self.title, font=self.title_font).pack(side=TOP, anchor=CENTER)
        self.title_frame.pack(side=TOP, pady=10)

        Label(self.image_step_frame, text="Image step:", width=10, anchor=W).pack(side=LEFT, anchor=S)

        self.image_step_scale.pack(side=LEFT, anchor=E)
        self.image_step_frame.pack(side=TOP, pady=10)

        Label(self.epochs_frame, text="Num epochs: ", width=10, anchor=W).pack(side=LEFT, anchor=S)
        self.epochs_scale.pack(side=LEFT, anchor=E)
        self.epochs_frame.pack(side=TOP, pady=10)

        self.optimize_button.pack(side=LEFT, anchor=E)
        self.optimize_parameters_frame.pack(side=TOP, pady=10)

        self.download_model_button.pack(side=BOTTOM, anchor=E)
        self.download_model_frame.pack(side=BOTTOM, pady=10, padx=10)

        self.drawing_canvas.pack(side=TOP, pady=10)
        self.pareto_plot_frame.pack(side=TOP, pady=10)

    # Function sets geometry of master
    def set_geometry(self):
        self._master.geometry(self._geometry)

    def fetch_base_path(self):
        self.base_path = self._master.base_path

    def fetch_berkley_path(self):
        self.berkley_path = self._master.berkley_path

    def get_paths(self):
        self.template_path = os.path.join(self.base_path, "templates")
        self.output_path = os.path.join(self.base_path, "output")
        self.regularization_path = os.path.join(self.base_path, "regularization")
        self.background_path = os.path.join(self.base_path, "backgrounds")

    def fetch_parameters(self):
        self.template_path = self._master.param_dict["Path to templates"]
        self.output_path = self._master.param_dict["Output path"]
        self.regularization_path = self._master.param_dict["Regularization path"]
        self.background_path = self._master.param_dict["Path to backgrounds"]

    def optimizer_thread(self):
        _thread.start_new_thread(self.optimize, ())

    def download_model(self):
        base_path = os.path.join(self.output_path, "../")
        model_path = os.path.join(base_path, "model_data", "trained_models")
        optimal_models_path = os.path.join(base_path, "optimal_models")
        if os.path.isdir(optimal_models_path):
            shutil.rmtree(optimal_models_path)
        os.mkdir(optimal_models_path)
        for epoch in ParetoPlotter.optimal_models_epochs:
            optimal_epoch = os.path.join(model_path, "output_network_epoch{0}.h5".format(epoch))
            if os.path.isfile(optimal_epoch):
                epoch_dst = os.path.join(optimal_models_path, str(epoch))
                if not os.path.exists(epoch_dst):
                    os.mkdir(epoch_dst)
                shutil.copy(optimal_epoch, epoch_dst)

    def optimize(self):
        self.fetch_base_path()
        self.fetch_berkley_path()
        self.get_paths()
        trained_models_path = os.path.join(self.base_path, "model_data", "trained_models")
        if os.path.isdir(trained_models_path):
            shutil.rmtree(trained_models_path)
        os.mkdir(trained_models_path)

        # self.fetch_parameters()
        optimizer = ParetoOptimizer(template_path=self.template_path,
                                    output_path=self.output_path,
                                    bg_path=self.background_path,
                                    regularization_path=self.regularization_path,
                                    berkley_path=self.berkley_path,
                                    image_step=int(self.image_step))
                                    # frame=self.drawing_canvas)
        result = run_optimization(optimizer, int(self.num_epochs))
        costs, sol_vars = optimizer.decode_results(result)
        for i in range(0, len(costs)):
            print("Cost[{0}]={1}".format(i, costs[i]))
            print("Variables[{0}] = {1}".format(i, sol_vars[i]))
        _thread.exit_thread()
