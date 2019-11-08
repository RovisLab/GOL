from qt_environment import Ui_MainWindow
from PyQt5 import QtWidgets
import sys


class SetupPageHelper(object):

    def __init__(self, ui):
        self.ui = ui

    def browse_dataset_path(self):
        dataset_path = QtWidgets.QFileDialog.getExistingDirectory()
        self.ui.dataset_path_textbox.setText(dataset_path)
