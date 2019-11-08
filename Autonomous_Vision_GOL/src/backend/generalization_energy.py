import shutil
import os
import numpy as np
import cv2
from platypus import *
from object_detection_asg import ObjectDetectionASG
from plot_functions import ParetoPlotter
from classifier import Classifier
from optimizer_utils import bhatt, bhatt_coeff, lbp_feature

NUM_POINTS = 24
RADIUS = 8
# RESIZE_X = 32
RESIZE_X = 32
RESIZE_Y = 32


class ParetoOptimizer(Problem):
    def __init__(self, template_path, output_path, bg_path, regularization_path, berkley_path, image_step, widget):
        super(ParetoOptimizer, self).__init__(nvars=18, nobjs=1, nconstrs=6)
        self.per_epoch_results = list()
        self.pic_w = None
        self.pic_h = None
        self.pic = None
        self.num_sane = 0
        self.num_insane = 0
        # self.frame = frame
        self.berkley_path = berkley_path
        self.output_path = output_path
        self.template_path = template_path
        self.bg_path = bg_path
        self.regularization_path = regularization_path
        self.image_step = image_step
        self.parameter_ranges = [
            Integer(0, 500), Integer(0, 500), Real(-0.5, 0.5), Real(-0.5, 0.5), Real(-0.3, 0.3), Real(-0.3, 0.3),
            Integer(0, 10), Integer(0, 10), Integer(0, 20), Integer(0, 20), Integer(0, 5), Integer(0, 5),
            Integer(0, 10), Integer(0, 10), Real(0, 0.00010), Integer(0, 15), Integer(0, 50), Integer(0, 50)
        ]
        self.types[:] = self.parameter_ranges
        self.constraints[:] = ">0"
        self.epoch = 0

        self.widget = widget

    def qt_plot_results(self, epoch_nr):
        energy_v = [v[1] for v in self.per_epoch_results]
        acc_v = [v[0] for v in self.per_epoch_results]
        ParetoPlotter.plot_solution(self.widget,
                                           energy_v,
                                           acc_v,
                                           self.epoch,
                                           epoch_nr)

    # def plot_results(self, frame, epoch_nr):
    # energy_v = [v[1] for v in self.per_epoch_results]
    # acc_v = [v[0] for v in self.per_epoch_results]
    # return ParetoPlotter.plot_solution(frame,
    #                                     energy_v,
    #                                     acc_v,
    #                                     self.epoch,
    #                                     epoch_nr)

    def evaluate(self, solution):
        min_resize = solution.variables[0]
        max_resize = solution.variables[1]
        min_h_persp = solution.variables[2]
        max_h_persp = solution.variables[3]
        min_v_persp = solution.variables[4]
        max_v_persp = solution.variables[5]
        min_brightness = solution.variables[6]
        max_brightness = solution.variables[7]
        min_noise = solution.variables[8]
        max_noise = solution.variables[9]
        min_blur = solution.variables[10]
        max_blur = solution.variables[11]
        solution.constraints = [max_resize - min_resize,
                                max_h_persp - min_h_persp,
                                max_v_persp - min_v_persp,
                                max_brightness - min_brightness,
                                max_noise - min_noise,
                                max_blur - min_blur]
        solution.objectives[:] = self.run_and_evaluate(solution.variables)
        print("variables: {0}".format(solution.variables))
        print("Objective: {0}".format(solution.objectives[0]))
        self.epoch += 1

    def get_regularization_images_filepaths(self, sample_label):
        image_path = os.path.join(self.regularization_path, sample_label)
        if not os.path.exists(image_path):
            raise IOError("Path to output images {0} does not exist".format(image_path))
        images = [os.path.join(image_path, f) for f in os.listdir(image_path) if f.endswith(".png")]
        return images

    def get_generated_images_filepaths(self, sample_label):
        image_path = os.path.join(self.output_path, sample_label)
        if not os.path.exists(image_path):
            raise IOError("Path to output images does not exist")
        images = [os.path.join(image_path, f) for f in os.listdir(image_path) if f.endswith(".png")]
        return images[0:len(images) - 1: self.image_step]

    def get_labels(self):
        files = os.listdir(self.output_path)
        print("Files: %s" % files)
        return [f for f in files if os.path.isdir(os.path.join(self.output_path, f))]

    def calculate(self, label):
        reg_samples = self.get_regularization_images_filepaths(label)
        if len(reg_samples) == 0:
            raise Exception("Regularization images not found")
        generated_samples = self.get_generated_images_filepaths(label)
        if len(generated_samples) == 0:
            raise Exception("Generated images not found")
        feature_set_reg = [lbp_feature(cv2.imread(reg_s),
                                       NUM_POINTS, RADIUS, (RESIZE_X, RESIZE_Y))
                           for reg_s in reg_samples]

        feature_set_gen = [lbp_feature(cv2.imread(gen_s),
                                       NUM_POINTS, RADIUS, (RESIZE_X, RESIZE_Y))
                           for gen_s in generated_samples]
        energy = 0
        for r_s in feature_set_reg:
            for g_s in feature_set_gen:
                b_d = bhatt(r_s, g_s)
                energy += math.sqrt(1.0 - (1.0 / math.sqrt(np.mean(r_s) * np.mean(g_s) * (len(r_s) ** 2))) * b_d)
        print("energy: %s" % energy)
        return energy

    def calculate_energy_for_all_classes(self):
        """
        Calculates generalization energy (based on Batthacharyya distance) for all classes (J_k)
        :return: total energy
        """
        labels = self.get_labels()
        print("Labels: %s" % labels)
        energy_vector = list()
        for label in labels:
            energy_vector.append(self.calculate(label))
        return sum(energy_vector)

    def _get_num_templates(self):
        counter = 0
        for f in os.listdir(self.template_path):
            if f.endswith(".png"):
                counter += 1
        return counter

    def cleanup_output(self):
        labels = self.get_labels()
        abs_paths = [os.path.join(self.output_path, l) for l in labels]
        for p in abs_paths:
            shutil.rmtree(p)

    def cleanup_classifier_output(self):
        pass

    def _validate_parameters(self, params):
        min_resize = params[0]
        max_resize = params[1]
        min_h_persp = params[2]
        max_h_persp = params[3]
        min_v_persp = params[4]
        max_v_persp = params[5]
        min_brightness = params[6]
        max_brightness = params[7]
        min_noise = params[8]
        max_noise = params[9]
        min_blur = params[10]
        max_blur = params[11]
        if max_resize - min_resize <= 0 or \
                max_h_persp - min_h_persp <= 0 or \
                max_v_persp - min_v_persp <= 0 or \
                max_brightness - min_brightness <= 0 or \
                max_noise - min_noise <= 0 or \
                max_blur - min_blur <= 0:
            return False
        return True

    def train_yolo_and_evaluate_dummy(self, energy):
        accuracy = np.random.uniform(0.65, 1.0)
        self.per_epoch_results.append((accuracy, energy))

    def train_yolo_and_evaluate(self, energy):
        # Paths for training and evaluating
        base_path = os.path.join(self.output_path, "../")
        eval_path = os.path.join(base_path, "evaluation")
        if not os.path.exists(eval_path):
            raise IOError
        annotation_file_path = os.path.join(eval_path, "annotation_file.txt")
        classes_file_path = os.path.join(eval_path, "classes.txt")
        model_path = os.path.join(base_path, "model_data")
        trained_model_path = os.path.join(model_path, "trained_models")
        if not os.path.exists(trained_model_path):
            raise IOError
        if not os.path.exists(model_path):
            raise IOError
        anchors_path = os.path.join(model_path, "tiny_yolo_anchors.txt")
        if not os.path.exists(anchors_path):
            raise IOError
        classifier_out_path = os.path.join(trained_model_path, "output_network_epoch{0}.h5".format(self.epoch))
        eval_data_path = os.path.join(eval_path, "data")
        if not os.path.exists(eval_data_path):
            raise IOError
        pred_path = os.path.join(eval_path, "predicted")
        if not os.path.exists(pred_path):
            raise IOError

        classifier = Classifier()

        classifier.prepare_training_set(path2data=self.output_path, size=(RESIZE_X, RESIZE_Y))
        classifier.generate_training_annotation(path2data=self.output_path,
                                                annotation_file=annotation_file_path,
                                                classes_file=classes_file_path,
                                                berkley_path=self.berkley_path)
        classifier.train_yolo_classifier(train_annotation_file=annotation_file_path,
                                         log_dir=eval_path,
                                         classes_file=classes_file_path,
                                         anchors_file=anchors_path,
                                         size=(RESIZE_X, RESIZE_Y),
                                         output_network_path=classifier_out_path)
        classifier.yolo_detect(path2images=eval_data_path,
                               model_path=classifier_out_path,
                               anchors_path=anchors_path,
                               classes_path=classes_file_path,
                               size=(RESIZE_X, RESIZE_Y),
                               predicted_path=pred_path)

        # _ = map(os.remove, glob.glob(os.path.join(eval_path, 'events.out.tfevents*')))

        acc = classifier.accuracy_yolo(eval_path=eval_path)
        if acc is None:
            raise Exception("Could not calculate mAP")
        self.per_epoch_results.append((acc, energy))

    def run_and_evaluate(self, params):
        asg_obj = ObjectDetectionASG()
        if self._validate_parameters(params) is True:
            asg_obj.run_generator_from_parameters(parameters=params,
                                                  epoch_nr=self.num_sane + self.num_insane,
                                                  templates_path=self.template_path,
                                                  output_path=self.output_path)
            energy = self.calculate_energy_for_all_classes()

            self.train_yolo_and_evaluate(energy)
            # self.train_yolo_and_evaluate_dummy(energy)
            # self.pic = self.plot_results(self.frame, self.num_insane + self.num_sane)
            self.qt_plot_results(self.num_sane + self.num_insane)
            # self.pic_w, self.pic_h = self.pic.width(), self.pic.height()
            self.cleanup_output()
            self.num_sane += 1
            print("Got sane energy: {0}, energy value: {1}".format(self.num_sane, energy))
        else:
            self.num_insane += 1
            print("Warning: Parameters don't meet constraints")
            print("Got insane energy: {0}".format(self.num_insane))
            energy = 10000
        print("Number of epochs: {0}".format(self.num_sane + self.num_insane))
        return energy

    def decode_variables(self, sol_vars):
        param_list = list()
        for i in range(0, len(self.parameter_ranges)):
            param_list.append(self.parameter_ranges[i].decode(sol_vars[i]))
        return param_list

    def decode_results(self, result):
        costs = list()
        variables = list()
        for res in result:
            costs.append(res.objectives[0])
            variables.append(self.decode_variables(res.variables))
        return costs, variables


def run_optimization(optimizer, num_epochs):
    alg = NSGAII(optimizer, variator=CompoundOperator(SBX(), HUX(), PM(), BitFlip()))
    alg.run(num_epochs)
    return alg.result
