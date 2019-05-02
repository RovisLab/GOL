import os
import shutil
import random
import json
import cv2
import numpy as np


class BerkleyTags(object):
    FRAME_VECTOR = "frames"
    OBJECT_VECTOR = "objects"
    CATEGORY = "category"
    BOX2D = "box2d"
    X_MIN = "x1"
    Y_MIN = "y1"
    X_MAX = "x2"
    Y_MAX = "y2"


class BerkleyDatasetHandler(object):
    def __init__(self, dataset_basepath, classes, num_templates, num_regularization):
        self.dataset_basepath = dataset_basepath
        self.classes = classes
        self.num_templates = num_templates
        self.num_regularization = num_regularization

    @staticmethod
    def get_objects(json_fp, category):
        objects = list()
        with open(json_fp, "r") as f:
            json_f = json.load(f)
        for frame in json_f[BerkleyTags.FRAME_VECTOR]:
            for obj in frame[BerkleyTags.OBJECT_VECTOR]:
                if obj[BerkleyTags.CATEGORY] == category:
                    objects.append(obj)
        return objects

    @staticmethod
    def get_all_objects(json_fp):
        objects = list()
        with open(json_fp, "r") as f:
            json_f = json.load(f)
        for frame in json_f[BerkleyTags.FRAME_VECTOR]:
            objects.extend(frame[BerkleyTags.OBJECT_VECTOR])
        return objects

    def get_all_objects_of_interest(self, json_fp):
        objects = BerkleyDatasetHandler.get_all_objects(json_fp)
        return [obj for obj in objects if obj[BerkleyTags.CATEGORY] in self.get_all_classnames()]

    def get_ordered_dataset_files(self):
        ordered_files = dict()
        for k in self.classes:
            ordered_files[self.classes[k]] = list()
        images_folder = os.path.join(self.dataset_basepath, "images")
        if not os.path.exists(images_folder):
            raise IOError
        labels_folder = os.path.join(self.dataset_basepath, "labels")
        if not os.path.exists(labels_folder):
            raise IOError
        img_100k_train = os.path.join(images_folder, "100k", "train")
        labels_100k_train = os.path.join(labels_folder, "100k", "train")
        for f in os.listdir(img_100k_train):
            filename = os.path.splitext(f)[0]
            label_file = os.path.join(labels_100k_train, filename + ".json")
            image_file = os.path.join(img_100k_train, f)
            if os.path.exists(label_file):
                objects = BerkleyDatasetHandler.get_all_objects(label_file)
                for obj in objects:
                    for k in self.classes:
                        if self.classes[k] == obj[BerkleyTags.CATEGORY]:
                            ordered_files[self.classes[k]].append((image_file, label_file))
        return ordered_files

    def get_all_classnames(self):
        classes = list()
        for k in self.classes:
            classes.append(self.classes[k])
        return classes

    def get_classid(self, classname):
        for k in self.classes:
            if self.classes[k] == classname:
                return k
        raise ValueError

    def get_all_images_and_labels(self, is_training_data=True):
        pairs = list()
        images_folder = os.path.join(self.dataset_basepath, "images")
        if not os.path.exists(images_folder):
            raise IOError
        labels_folder = os.path.join(self.dataset_basepath, "labels")
        if not os.path.exists(labels_folder):
            raise IOError
        img_100k = ""
        labels_100k = ""
        if is_training_data:
            img_100k = os.path.join(images_folder, "100k", "train")
            labels_100k = os.path.join(labels_folder, "100k", "train")
        else:
            img_100k = os.path.join(images_folder, "100k", "val")
            labels_100k = os.path.join(labels_folder, "100k", "val")
        for f in os.listdir(img_100k):
            label_file = os.path.join(labels_100k, os.path.splitext(f)[0] + ".json")
            image_file = os.path.join(img_100k, f)
            if os.path.exists(label_file):
                objects = BerkleyDatasetHandler.get_all_objects(label_file)
                filtered_obj = [obj for obj in objects if obj[BerkleyTags.CATEGORY] in self.get_all_classnames()]
                if len(filtered_obj) > 0:
                    pairs.append((image_file, label_file))
        return pairs

    def sample_roi(self, category, image, label):
        objects = BerkleyDatasetHandler.get_objects(json_fp=label, category=category)
        sample = random.choice(objects)
        x_min = int(sample[BerkleyTags.BOX2D][BerkleyTags.X_MIN])
        x_max = int(sample[BerkleyTags.BOX2D][BerkleyTags.X_MAX])
        y_min = int(sample[BerkleyTags.BOX2D][BerkleyTags.Y_MIN])
        y_max = int(sample[BerkleyTags.BOX2D][BerkleyTags.Y_MAX])

        h = y_max - y_min
        w = x_max - x_min
        return image[y_min:y_min + h, x_min:x_min + w]

    def copy_templates_and_regularization(self, dataset_file_dict, gol_basepath):
        template_path = os.path.join(gol_basepath, "templates")
        reg_path = os.path.join(gol_basepath, "regularization")
        for key in self.classes:
            crt_class_path_t = os.path.join(template_path, self.classes[key])
            crt_class_path_r = os.path.join(reg_path, self.classes[key])
            if os.path.exists(crt_class_path_t):
                shutil.rmtree(crt_class_path_t)
            if os.path.exists(crt_class_path_r):
                shutil.rmtree(crt_class_path_r)
            os.mkdir(crt_class_path_t)
            os.mkdir(crt_class_path_r)
            template_choice = random.choice(dataset_file_dict[self.classes[key]])

            label = template_choice[1]
            image = template_choice[0]
            r = self.sample_roi(self.classes[key], cv2.imread(image), label)
            cv2.imwrite(os.path.join(crt_class_path_t, "template_{0}.png".format(self.classes[key])), r)

            regularization_choices = random.choices(dataset_file_dict[self.classes[key]], k=self.num_regularization)
            crt_counter = 0
            for reg_file in regularization_choices:
                image = reg_file[0]
                label = reg_file[1]
                r = self.sample_roi(self.classes[key], cv2.imread(image), label)
                cv2.imwrite(os.path.join(crt_class_path_r,
                                         "regularization_{0}_{1}.png".format(self.classes[key], crt_counter)), r)
                crt_counter += 1

    def update_annotation_file(self, dataset_files_and_labels, gol_basepath):
        eval_path = os.path.join(gol_basepath, "evaluation")
        annotation_file = os.path.join(eval_path, "annotation_file.txt")
        if os.path.exists(annotation_file):
            os.remove(annotation_file)
        index = 0
        for img, label in dataset_files_and_labels:
            if index == 1000:
                break
            obj_int = self.get_all_objects_of_interest(label)
            annotation_data = "{0} ".format(img)
            for obj in obj_int:
                annotation_data += "{0},{1},{2},{3},{4} ".format(int(obj[BerkleyTags.BOX2D][BerkleyTags.X_MIN]),
                                                                 int(obj[BerkleyTags.BOX2D][BerkleyTags.Y_MIN]),
                                                                 int(obj[BerkleyTags.BOX2D][BerkleyTags.X_MAX]),
                                                                 int(obj[BerkleyTags.BOX2D][BerkleyTags.Y_MAX]),
                                                                 self.get_classid(obj[BerkleyTags.CATEGORY]))
            annotation_data += "\n"
            with open(annotation_file, "a") as f:
                f.write(annotation_data)
            index+=1

    def get_eval_data(self, eval_data_files_and_labels, gol_basepath):
        eval_path = os.path.join(gol_basepath, "evaluation")
        eval_data_path = os.path.join(eval_path, "data")
        gt_data_path = os.path.join(eval_path, "ground-truth")
        index = 0
        for img, label in eval_data_files_and_labels:
            if index == 100:
                break
            shutil.copy(img, eval_data_path)
            obj_int = self.get_all_objects_of_interest(label)
            with open(os.path.join(gt_data_path, os.path.splitext(os.path.basename(label))[0] + ".txt"), "w") as f:
                gt_data = ""
                for obj in obj_int:
                    gt_data += "{0} {1} {2} {3} {4}".format(obj[BerkleyTags.CATEGORY],
                                                            int(obj[BerkleyTags.BOX2D][BerkleyTags.X_MIN]),
                                                            int(obj[BerkleyTags.BOX2D][BerkleyTags.Y_MIN]),
                                                            int(obj[BerkleyTags.BOX2D][BerkleyTags.X_MAX]),
                                                            int(obj[BerkleyTags.BOX2D][BerkleyTags.Y_MAX]))
                    gt_data += "\n"
                f.write(gt_data)
            index+=1

if __name__ == "__main__":
    class_dict = dict()
    class_dict[0] = "car"
    class_dict[1] = "person"
    class_dict[2] = "traffic sign"
    bd = BerkleyDatasetHandler(dataset_basepath="D:\\AndreiMihalcea\\berkley_dataset",
                               classes=class_dict,
                               num_templates=10,
                               num_regularization=10)
    # print("Copying templates and regularization files")
    # bd.copy_templates_and_regularization(dataset_file_dict=bd.get_ordered_dataset_files(),
    #                                      gol_basepath="D:\AndreiMihalcea\GOL")

    # print("Copying training data and updating annotation file")
    # bd.update_annotation_file(dataset_files_and_labels=bd.get_all_images_and_labels(True), gol_basepath="D:\AndreiMihalcea\GOL")

    print("Copying gt data")
    bd.get_eval_data(eval_data_files_and_labels=bd.get_all_images_and_labels(is_training_data=False), gol_basepath="D:\AndreiMihalcea\GOL")
