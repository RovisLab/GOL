import os
import glob
from path_error_status import ErrorStatus


class PathChecker(object):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    # Checks if the base folder structure is valid
    def check_base_folder_structure(self):
        templates_folder = os.path.isdir(os.path.join(self.dataset_path, "templates"))
        regularization_folder = os.path.isdir(os.path.join(self.dataset_path, "regularization"))
        backgrounds_folder = os.path.isdir(os.path.join(self.dataset_path, "backgrounds"))
        output_folder = os.path.isdir(os.path.join(self.dataset_path, "output"))
        model_data_folder = os.path.isdir(os.path.join(self.dataset_path, "model_data"))
        eval_folder = os.path.isdir(os.path.join(self.dataset_path, "evaluation"))

        if not templates_folder:
            return ErrorStatus.TEMPLATES
        if not regularization_folder:
            return ErrorStatus.REGULARIZATION
        if not backgrounds_folder:
            return ErrorStatus.BACKGROUNDS
        if not output_folder:
            return ErrorStatus.OUTPUT
        if not model_data_folder:
            return ErrorStatus.MODEL
        if not eval_folder:
            return ErrorStatus.EVALUATION
        if not self.check_model_data_folder():
            return ErrorStatus.MODEL_FOLDER
        return ErrorStatus.NEGATIVE

    # Checks if the model files are present in the model folder
    def check_model_data_folder(self):
        model_data_folder = os.path.join(self.dataset_path, "model_data")
        if not os.path.isfile(os.path.join(model_data_folder, "yolov3-tiny.weights")):
            return False
        if not os.path.isfile(os.path.join(model_data_folder, "tiny_yolo_anchors.txt")):
            return False
        return True

    # Checks for the correct correspondence between the templates and the regularization samples
    def check_templates_and_reg_folders(self):
        templates_folder = os.path.join(self.dataset_path, "templates")
        regularization_folder = os.path.join(self.dataset_path, "regularization")
        templates_classes_dirs = [name for name in os.listdir(templates_folder) if
                                  os.path.isdir(os.path.join(templates_folder, name))]
        if len(templates_classes_dirs) == 0:
            return ErrorStatus.NO_TEMPLATES
        for folder in templates_classes_dirs:
            templates = [f for f in glob.glob(os.path.join(templates_folder, folder) + "/*.png")]
            if len(templates) == 0:
                return ErrorStatus.NO_TEMPLATES
            label = self.get_folder_name(folder)
            label_path = os.path.join(regularization_folder, label)
            regularization_templates = [f for f in glob.glob(label_path + "/*.png")]
            if not os.path.isdir(label_path) or len(regularization_templates) == 0:
                return ErrorStatus.NO_REGULARIZATION
        return ErrorStatus.NEGATIVE

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

    @staticmethod
    def get_folder_name(folder):
        return os.path.basename(folder)
