from qt_environment import Ui_MainWindow
from PyQt5 import QtWidgets
from PyQt5.QtCore import QThreadPool
from qt_optimization_worker import OptimizationWorker
import sys
from generalization_energy import ParetoOptimizer, run_optimization
import os
import shutil
from plot_functions import ParetoPlotter


# This class handles the actions that take place in the optimization page
class OptimizationPageHelper(object):

    def __init__(self, ui):
        self.ui = ui
        self.threadpool = QThreadPool()
        self.dataset_path = ""
        # self.template_path = ""
        # self.output_path = ""
        # self.regularization_path = ""
        # self.background_path = ""

    # Downloads only the Pareto optimal models
    def download_models(self):
        model_path = os.path.join(self.dataset_path, "model_data", "trained_models")
        optimal_models_path = os.path.join(self.dataset_path, "optimal_models")
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

    # Starts a thread where the whole process takes place
    def generate(self):
        image_step_coefficient = self.ui.img_step_label.text()
        nr_of_epochs = self.ui.nr_of_epochs_label.text()
        widget = self.ui.plot_canvas
        worker = OptimizationWorker(self.dataset_path, image_step_coefficient, nr_of_epochs, widget)
        self.threadpool.start(worker)
        # print(self.dataset_path)
        # image_step_coefficient = self.ui.img_step_label.text()
        # nr_of_epochs = self.ui.nr_of_epochs_label.text()
        # self.set_paths()
        # trained_models_path = os.path.join(self.dataset_path, "model_data", "trained_models")
        # if os.path.isdir(trained_models_path):
        #     shutil.rmtree(trained_models_path)
        # os.mkdir(trained_models_path)
        # optimizer = ParetoOptimizer(template_path=self.template_path,
        #                             output_path=self.output_path,
        #                             bg_path=self.background_path,
        #                             regularization_path=self.regularization_path,
        #                             berkley_path='',
        #                             image_step=int(image_step_coefficient))
        # result = run_optimization(optimizer, int(nr_of_epochs))
        # costs, sol_vars = optimizer.decode_results(result)
        # for i in range(0, len(costs)):
        #     print("Cost[{0}]={1}".format(i, costs[i]))
        #     print("Variables[{0}] = {1}".format(i, sol_vars[i]))

    # def set_paths(self):
    #     self.template_path = os.path.join(self.dataset_path, "templates")
    #     self.output_path = os.path.join(self.dataset_path, "output")
    #     self.regularization_path = os.path.join(self.dataset_path, "regularization")
    #     self.background_path = os.path.join(self.dataset_path, "backgrounds")



