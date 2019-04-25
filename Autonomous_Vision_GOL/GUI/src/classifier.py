import os
import cv2
import mean_average_precision
from train_yolo import train_yolo
import yolo_video as y
import yolo
from dataset import BerkleyDatasetHandler


def prepare_training_set(path2data, size):
    """
    Prepare training images to be fed to classifier
    :param path2data: path to GOL output folder
    :param size: tuple (w, h)
    :return: None
    """
    for root, dirs, files in os.walk(path2data):
        for f in files:
            if f.endswith(".png"):
                img = cv2.imread(os.path.join(root, f), cv2.IMREAD_UNCHANGED)
                img_resized = cv2.resize(img, size)
                cv2.imwrite(os.path.join(root, f), img_resized)


def generate_training_annotation(path2data, annotation_file, classes_file, berkley_path):
    """
    Generate data from GOL output path to be compatible with TinyYolo
    :param path2data: path to GOL output folder (each class is a folder in this root folder)
    :param annotation_file: path to output annotation text file to train our network
    :param classes_file: path to output classes file
    :return: None
    """
    classes = dict()  # classes
    files = dict()  # files inside each class folder with full path
    counter = 0
    for f in os.listdir(path2data):
        if os.path.isdir(os.path.join(path2data, f)):
            classes[counter] = f
            counter += 1

    for k in classes:
        fp = os.path.join(path2data, classes[k])
        files[k] = [os.path.join(fp, x) for x in os.listdir(fp)]

    if berkley_path == '':
        with open(annotation_file, "w") as out_file:
            for k in classes:
                for f in files[k]:
                    if f.endswith(".png"):
                        img = cv2.imread(f)
                        height, width = img.shape[:2]
                        out_file.write(
                            "{0} {1},{2},{3},{4},{5}\n".format(f, 0, 0,
                                                               width, height, k))

    else:
        berkley_handler = BerkleyDatasetHandler(dataset_basepath=berkley_path,
                                   classes=classes,
                                   num_templates=10,
                                   num_regularization=10)
        berkley_handler.update_annotation_file(dataset_files_and_labels=berkley_handler.get_all_images_and_labels(is_training_data=True),
                                               gol_basepath=os.path.join(path2data, ".."))

        with open(annotation_file, "a") as out_file:
            for k in classes:
                for f in files[k]:
                    if f.endswith(".png"):
                        img = cv2.imread(f)
                        height, width = img.shape[:2]
                        out_file.write(
                            "{0} {1},{2},{3},{4},{5}\n".format(f, 0, 0,
                                                               width, height, k))

    with open(classes_file, "w") as class_file:
        for k in classes:
            class_file.write("{0}\n".format(classes[k]))


def train_yolo_classifier(train_annotation_file, log_dir, classes_file, anchors_file, output_network_path, size):
    """
    Train TinyYolo model for the generated samples
    :param train_annotation_file: path to file which contains annotation data for training images
    :param classes_file: path to the input classes file for training our model
    :param output_network_path: path where the trained network will be saved
    :return: None
    """
    train_yolo(annotation_path=train_annotation_file,
               classes_path=classes_file,
               anchors_path=anchors_file,
               input_shape=size,
               log_dir=log_dir,
               output_path=output_network_path)


def accuracy_yolo(eval_path):
    """
    Calculate accuracy of trained Yolo network
    :param eval_path: Path to evaluation data
    :return: accuracy
    """
    acc = mean_average_precision.run_evaluation_map(path_to_eval_data=eval_path)
    try:
        return acc["mAP"]
    except KeyError:
        return None


def yolo_detect(path2images, model_path, anchors_path, classes_path, size, predicted_path):
    """
    Use pre-trained Yolo network to detect object in input images
    :param path2images: Folder with images which we want to detect
    :param model_path: Path to h5 Yolo model file
    :param anchors_path: Path to anchors file
    :param classes_path: Path to classes description file
    :param size: Yolo model image size
    :param predicted_path: Path to output folder to save results
    :return: None
    """

    params = {
        "model_path": "{0}".format(model_path),
        "anchors_path": "{0}".format(anchors_path),
        "classes_path": "{0}".format(classes_path),
        # "model_image_size": size
    }

    for f in os.listdir(path2images):
        bboxes = y.detect_image(yolo.YOLO(**params), img_filename=os.path.join(path2images, f))
        if len(bboxes) > 0:
            for bbox in bboxes:
                pred_fname = os.path.join(predicted_path, "{0}.txt".format(f.split(".")[0]))
                with open(pred_fname, "w") as pred_file:
                    pred_file.write("{0} {1} {2} {3} {4}".format(bbox[0], bbox[2][0], bbox[2][1], bbox[3][0], bbox[3][1]))
        else:
            pred_fname = os.path.join(predicted_path, "{0}.txt".format(f.split(".")[0]))
            with open(pred_fname, "w") as pred_file:
                pred_file.write("undefined 0 0 0 0 0")
