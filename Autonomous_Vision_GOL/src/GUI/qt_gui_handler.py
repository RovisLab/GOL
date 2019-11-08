from qt_environment import Ui_MainWindow
from PyQt5 import QtWidgets
from PyQt5.QtCore import QThreadPool
from PyQt5.QtWidgets import QMessageBox
import sys
from qt_setup_page import SetupPageHelper
from qt_optimization_page import OptimizationPageHelper
from path_checker import PathChecker
from path_error_status import ErrorStatus


# This class handles the whole GUI
class GUIHandler(object):
    def __init__(self, ui):
        self.dataset_path = ''
        self.setup_page_helper = SetupPageHelper(ui)
        self.optimization_page_helper = OptimizationPageHelper(ui)
        self.ui = ui
        self.btn_action_handler()
        self.view_setup_page()

    # Handles the actions for every button
    def btn_action_handler(self):
        self.setup_btn_action_handler()
        self.optimization_btn_action_handler()

    # Handles the actions for the buttons from the setup page
    def setup_btn_action_handler(self):
        self.ui.browse_btn.clicked.connect(self.setup_page_helper.browse_dataset_path)
        self.ui.optimize_btn.clicked.connect(self.view_optimization_page)

    # Handles the action for the buttons from the optimization page
    def optimization_btn_action_handler(self):
        self.ui.back_btn.clicked.connect(self.view_setup_page)
        self.ui.generate_btn.clicked.connect(self.optimization_page_helper.generate)
        self.ui.download_btn.clicked.connect(self.optimization_page_helper.download_models)

    # Switch view to the optimization page
    def view_optimization_page(self):
        self.dataset_path = self.ui.dataset_path_textbox.toPlainText()
        if self.path_validation():
            self.optimization_page_helper.dataset_path = self.dataset_path
            self.ui.stackedWidget.setCurrentIndex(1)

    # Switch view to the setup page
    def view_setup_page(self):
        self.ui.stackedWidget.setCurrentIndex(0)

    # Checks if the path provided is valid
    def path_validation(self):
        path_checker = PathChecker(self.dataset_path)
        status = path_checker.check_base_folder_structure()
        if status != ErrorStatus.NEGATIVE:
            if status == ErrorStatus.TEMPLATES:
                QMessageBox.question(self.ui.page, 'Templates error', 'Templates folder is missing!', QMessageBox.Ok,
                                     QMessageBox.Ok)
                return False
            elif status == ErrorStatus.BACKGROUNDS:
                QMessageBox.question(self.ui.page, 'Backgrounds error', 'Backgrounds folder is missing!',
                                     QMessageBox.Ok,
                                     QMessageBox.Ok)
                return False
            elif status == ErrorStatus.REGULARIZATION:
                QMessageBox.question(self.ui.page, 'Regularization error', 'Regularization folder is missing!',
                                     QMessageBox.Ok,
                                     QMessageBox.Ok)
                return False
            elif status == ErrorStatus.OUTPUT:
                QMessageBox.question(self.ui.page, 'Output error', 'Output folder is missing!',
                                     QMessageBox.Ok,
                                     QMessageBox.Ok)
                return False
            elif status == ErrorStatus.MODEL:
                QMessageBox.question(self.ui.page, 'Model error', 'Model folder is missing!',
                                     QMessageBox.Ok,
                                     QMessageBox.Ok)
                return False
            elif status == ErrorStatus.EVALUATION:
                QMessageBox.question(self.ui.page, 'Evaluation error', 'Evaluation folder is missing!',
                                     QMessageBox.Ok,
                                     QMessageBox.Ok)
                return False
            elif status == ErrorStatus.MODEL_FOLDER:
                QMessageBox.question(self.ui.page, 'Model folder error', 'Neural network files are missing!',
                                     QMessageBox.Ok,
                                     QMessageBox.Ok)
                return False
        status = path_checker.check_templates_and_reg_folders()
        if status != ErrorStatus.NEGATIVE:
            if status == ErrorStatus.NO_TEMPLATES:
                QMessageBox.question(self.ui.page, 'No templates error',
                                     'Templates for at least one class are missing!',
                                     QMessageBox.Ok,
                                     QMessageBox.Ok)
                return False
            elif status == ErrorStatus.NO_REGULARIZATION:
                QMessageBox.question(self.ui.page, 'No regularization samples error',
                                     'Regularization samples for at least one class are missing!',
                                     QMessageBox.Ok,
                                     QMessageBox.Ok)
                return False
        return True


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    gui_handler = GUIHandler(ui)
    sys.exit(app.exec_())
