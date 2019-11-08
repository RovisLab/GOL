from PyQt5.QtCore import QRunnable
import os
import shutil
from generalization_energy import ParetoOptimizer, run_optimization


# This class inherits from QRunnable to start the whole process on a seperate thread
class OptimizationWorker(QRunnable):

    def __init__(self, dataset_path, image_step_coefficient, nr_of_epochs, widget):
        super().__init__()
        self.image_step_coefficient = image_step_coefficient
        self.nr_of_epochs = nr_of_epochs
        self.dataset_path = dataset_path
        self.template_path = ""
        self.output_path = ""
        self.regularization_path = ""
        self.background_path = ""
        self.widget = widget

    # Starts the whole process
    def run(self):
        print(self.dataset_path)
        image_step_coefficient = self.image_step_coefficient
        nr_of_epochs = self.nr_of_epochs
        self.set_paths()
        trained_models_path = os.path.join(self.dataset_path, "model_data", "trained_models")
        if os.path.isdir(trained_models_path):
            shutil.rmtree(trained_models_path)
        os.mkdir(trained_models_path)
        optimizer = ParetoOptimizer(template_path=self.template_path,
                                    output_path=self.output_path,
                                    bg_path=self.background_path,
                                    regularization_path=self.regularization_path,
                                    berkley_path='',
                                    image_step=int(image_step_coefficient),
                                    widget=self.widget)
        result = run_optimization(optimizer, int(nr_of_epochs))
        costs, sol_vars = optimizer.decode_results(result)
        for i in range(0, len(costs)):
            print("Cost[{0}]={1}".format(i, costs[i]))
            print("Variables[{0}] = {1}".format(i, sol_vars[i]))

    def set_paths(self):
        self.template_path = os.path.join(self.dataset_path, "templates")
        self.output_path = os.path.join(self.dataset_path, "output")
        self.regularization_path = os.path.join(self.dataset_path, "regularization")
        self.background_path = os.path.join(self.dataset_path, "backgrounds")
