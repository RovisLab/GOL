from config_file import ConfigFile
from paths_window_frame import PathsWindow
from parameters_window_frame import ParametersWindow
from optimization_window import OptimizationWindow
from while_generation_window_frame import GenerationWindow
from finish_window_frame import FinishWindow
from tkinter import Tk, Frame, Button
from tkinter import BOTTOM, TOP, E, W, DISABLED, LEFT, RIGHT
from object_detection_asg import ObjectDetectionASG
import _thread


# Class handles GUI application
class Application(Tk):
    def __init__(self):
        Tk.__init__(self)
        self.title("Generative One-Shot learning")

        self.base_path = ''
        self.berkley_path = ''

        # Parameters dictionary
        self.param_dict = ConfigFile.read_default_config()

        # Generator object
        self.generator = ObjectDetectionASG()

        # Objects corresponding to first frame - asking for directories
        self.next_button = Button(self,
                                  text='Next',
                                  width=10,
                                  state=DISABLED,
                                  command=self.next_callback)
        self.paths_window = PathsWindow(self, button=self.next_button)

        # Objects corresponding to second frame - parameters setting
        self.buttons_frame = Frame(self)
        self.back_button = Button(self.buttons_frame, text="Back", width=10, command=self.back_to_start_callback)
        self.generate_button = Button(self.buttons_frame, text='Generate', width=10, command=self.generate_callback)
        self.parameters_window = ParametersWindow(self, param_dict=self.param_dict)

        # Pareto Optimization Window
        self.optimization_window = OptimizationWindow(self)

        # Object corresponding to third frame - while generation
        self.generation_window = GenerationWindow(self)

        # Objects corresponding to forth frame - finish process
        self.close_button = Button(self, text="Close", width=10, command=self.quit)
        self.finish_window = FinishWindow(self)

    # Function sets first frame
    def start(self):
        self.paths_window.body_pack()
        self.next_button.pack(side=BOTTOM, anchor=E, padx=15, pady=10)

    def show_pareto_window(self):
        self.base_path = self.paths_window.get_base_path()
        self.berkley_path = self.paths_window.get_berkley_path()
        # print(self.base_path)
        # quit()
        # self.paths_window.get_paths(self.param_dict)
        self.paths_window.pack_forget()
        self.next_button.pack_forget()
        self.back_button.pack(side=LEFT, padx=15, pady=10)
        self.buttons_frame.pack(side=BOTTOM, anchor=E)
        self.optimization_window.body_pack()

    # Callback function setting second frame
    def next_callback(self):
        self.base_path = self.paths_window.get_base_path()

        # self.paths_window.get_paths(self.param_dict)

        self.paths_window.pack_forget()
        self.next_button.pack_forget()

        self.parameters_window.body_pack()

        self.back_button.pack(side=LEFT, padx=15, pady=10)
        self.generate_button.pack(side=RIGHT, padx=15, pady=10)
        self.buttons_frame.pack(side=BOTTOM, anchor=E)

    # Callback function restoring first frame
    def back_to_start_callback(self):
        self.parameters_window.pack_forget()
        self.optimization_window.pack_forget()
        self.back_button.pack_forget()
        self.generate_button.pack_forget()
        self.buttons_frame.pack_forget()

        self.paths_window.body_pack()
        self.next_button.pack(side=BOTTOM, anchor=E, padx=15, pady=10)

    # Callback function setting third frame, creates a second thread for generation process and calls the generator
    def generate_callback(self):
        self.parameters_window.get_parameters(self.param_dict)

        self.parameters_window.delete_misc()

        # Writing configuration file
        ConfigFile.write_config_file(self.param_dict)

        self.parameters_window.pack_forget()
        self.buttons_frame.pack_forget()

        self.generation_window.body_pack()

        # Calling generator in second thread
        _thread.start_new_thread(self.generation_handler, ())

    def generation_handler(self):
        nr_of_generations = self.param_dict["Number of generations"]

        self.generator.callback(nr_of_generations)
        self.finish()

        _thread.exit_thread()

    # Function sets forth frame
    def finish(self):
        self.generation_window.pack_forget()

        self.finish_window.body_pack()

        self.close_button.pack(side=BOTTOM, anchor=E, padx=5, pady=5)


# Call for the application interface
if __name__ == "__main__":
    app = Application()
    app.start()
    app.mainloop()
