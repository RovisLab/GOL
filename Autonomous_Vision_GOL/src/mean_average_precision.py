import glob
import json
import os
import shutil
import sys


MIN_OVERLAP = 0.5  # default value (defined in the PASCAL VOC2012 challenge)
# MIN_OVERLAP = 0.2


def is_float_between_0_and_1(value):
    """
     check if the number is a float between 0.0 and 1.0
    """
    try:
        val = float(value)
        if 0.0 < val < 1.0:
            return True
        else:
            return False
    except ValueError:
        return False


def voc_ap(rec, prec):
    """
    Calculate the AP given the recall and precision array
    1st) We compute a version of the measured precision/recall curve with
       precision monotonically decreasing
    2nd) We compute the AP as the area under this curve by numerical integration.

    --- Official matlab code VOC2012---
    mrec=[0 ; rec ; 1];
    mpre=[0 ; prec ; 0];
    for i=numel(mpre)-1:-1:1
        mpre(i)=max(mpre(i),mpre(i+1));
    end
    i=find(mrec(2:end)~=mrec(1:end-1))+1;
    ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    rec.insert(0, 0.0) # insert 0.0 at begining of list
    rec.append(1.0) # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0) # insert 0.0 at begining of list
    prec.append(0.0) # insert 0.0 at end of list
    mpre = prec[:]
    """
     This part makes the precision monotonically decreasing
      (goes from the end to the beginning)
      matlab:  for i=numel(mpre)-1:-1:1
                  mpre(i)=max(mpre(i),mpre(i+1));
    """
    # matlab indexes start in 1 but python in 0, so I have to do:
    #   range(start=(len(mpre) - 2), end=0, step=-1)
    # also the python function range excludes the end, resulting in:
    #   range(start=(len(mpre) - 2), end=-1, step=-1)
    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])
    """
     This part creates a list of indexes where the recall changes
      matlab:  i=find(mrec(2:end)~=mrec(1:end-1))+1;
    """
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i-1]:
            i_list.append(i) # if it was matlab would be i + 1
    """
     The Average Precision (AP) is the area under the curve
      (numerical integration)
      matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i]-mrec[i-1])*mpre[i])
    return ap, mrec, mpre


def file_lines_to_list(path):
    # open txt file lines to a list
    with open(path) as f:
        content = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    return content


def error(msg):
    print(msg)
    sys.exit(0)


#############################################################################################
#  GT  : class left top right bottom
#  Pred: class conf left top right bottom
#############################################################################################

def run_evaluation_map(path_to_eval_data):
    # if there are no classes to ignore then replace None by empty list
    ignore_classes = []

    """
     Create a "tmp_files/" and "results/" directory
    """
    tmp_files_path = "tmp_files"
    if not os.path.exists(os.path.join(path_to_eval_data, tmp_files_path)):  # if it doesn't exist already
        os.makedirs(os.path.join(path_to_eval_data, tmp_files_path))
    results_files_path = "results"

    if os.path.exists(os.path.join(path_to_eval_data, results_files_path)):  # if it exist already
        # reset the results directory
        shutil.rmtree(os.path.join(path_to_eval_data, results_files_path))
    os.makedirs(os.path.join(path_to_eval_data, results_files_path))

    """
     Ground-Truth
       Load each of the ground-truth files into a temporary ".json" file.
       Create a list of all the class names present in the ground-truth (gt_classes).
    """
    # get a list with the ground-truth files
    ground_truth_files_list = glob.glob(path_to_eval_data + '/ground-truth/*.txt')
    if len(ground_truth_files_list) == 0:
        error("Error: No ground-truth files found!")
    ground_truth_files_list.sort()
    # dictionary with counter per class
    gt_counter_per_class = {}

    for txt_file in ground_truth_files_list:
        # print(txt_file)
        file_id = txt_file.split(".txt", 1)[0]
        file_id = os.path.basename(os.path.normpath(file_id))
        # check if there is a correspondent predicted objects file
        if not os.path.exists(os.path.join(path_to_eval_data, 'predicted', "{0}.txt".format(file_id))):
            error_msg = "Error. File not found: predicted/" + file_id + ".txt\n"
            error_msg += "(You can avoid this error message by running extra/intersect-gt-and-pred.py)"
            error(error_msg)
        lines_list = file_lines_to_list(txt_file)
        # create ground-truth dictionary
        bounding_boxes = []
        is_difficult = False
        for line in lines_list:
            try:
                if "difficult" in line:
                    class_name, left, top, right, bottom, _difficult = line.split()
                    is_difficult = True
                else:
                    class_name, left, top, right, bottom = line.split()
            except ValueError:
                error_msg = "Error: File " + txt_file + " in the wrong format.\n"
                error_msg += " Expected: <class_name> <left> <top> <right> <bottom> ['difficult']\n"
                error_msg += " Received: " + line
                error_msg += "\n\nIf you have a <class_name> with spaces between words you should remove them\n"
                error_msg += "by running the script \"remove_space.py\" or " \
                             "\"rename_class.py\" in the \"extra/\" folder."
                raise Exception(error_msg)
            # check if class is in the ignore list, if yes skip
            if class_name in ignore_classes:
                continue
            bbox = left + " " + top + " " + right + " " + bottom
            if is_difficult:
                bounding_boxes.append({"class_name": class_name, "bbox": bbox, "used": False, "difficult": True})
                is_difficult = False
            else:
                bounding_boxes.append({"class_name": class_name, "bbox": bbox, "used": False})
                # count that object
                if class_name in gt_counter_per_class:
                    gt_counter_per_class[class_name] += 1
                else:
                    # if class didn't exist yet
                    gt_counter_per_class[class_name] = 1
        # dump bounding_boxes into a ".json" file
        with open(os.path.join(path_to_eval_data, tmp_files_path, "{0}_ground_truth.json".format(file_id)), 'w') as outfile:
            json.dump(bounding_boxes, outfile)

    gt_classes = list(gt_counter_per_class.keys())
    # let's sort the classes alphabetically
    gt_classes = sorted(gt_classes)
    n_classes = len(gt_classes)
    # print(gt_classes)
    # print(gt_counter_per_class)

    """
     Predicted
       Load each of the predicted files into a temporary ".json" file.
    """
    # get a list with the predicted files
    predicted_files_list = glob.glob(path_to_eval_data + '/predicted/*.txt')
    predicted_files_list.sort()

    for class_index, class_name in enumerate(gt_classes):
        bounding_boxes = []
        for txt_file in predicted_files_list:
            # print(txt_file)
            # the first time it checks if all the corresponding ground-truth files exist
            file_id = txt_file.split(".txt", 1)[0]
            file_id = os.path.basename(os.path.normpath(file_id))
            if class_index == 0:
                if not os.path.exists(os.path.join(path_to_eval_data, 'ground-truth', "{0}.txt".format(file_id))):
                    error_msg = "Error. File not found: ground-truth/" + file_id + ".txt\n"
                    error_msg += "(You can avoid this error message by running extra/intersect-gt-and-pred.py)"
                    error(error_msg)
            lines = file_lines_to_list(txt_file)
            for line in lines:
                try:
                    tmp_class_name, confidence, left, top, right, bottom = line.split()
                except ValueError:
                    error_msg = "Error: File " + txt_file + " in the wrong format.\n"
                    error_msg += " Expected: <class_name> <confidence> <left> <top> <right> <bottom>\n"
                    error_msg += " Received: " + line
                    raise Exception(error_msg)
                if tmp_class_name == class_name:
                    # print("match")
                    bbox = left + " " + top + " " + right + " " + bottom
                    bounding_boxes.append({"confidence": confidence, "file_id": file_id, "bbox": bbox})
                    # print(bounding_boxes)
        # sort predictions by decreasing confidence
        bounding_boxes.sort(key=lambda x: x['confidence'], reverse=True)
        with open(os.path.join(path_to_eval_data, tmp_files_path, "{0}_predictions.json".format(class_name)), 'w') as outfile:
            json.dump(bounding_boxes, outfile)

    """
     Calculate the AP for each class
    """
    sum_AP = 0.0
    ap_dictionary = {}

    # open file to store the results
    with open(os.path.join(path_to_eval_data, results_files_path, "results.txt"), 'w') as results_file:
        results_file.write("# AP and precision/recall per class\n")
        count_true_positives = {}
        for class_index, class_name in enumerate(gt_classes):
            count_true_positives[class_name] = 0
            """
             Load predictions of that class
            """
            predictions_file = os.path.join(path_to_eval_data, tmp_files_path, "{0}_predictions.json".format(class_name))
            predictions_data = json.load(open(predictions_file))

            """
             Assign predictions to ground truth objects
            """
            nd = len(predictions_data)
            tp = [0] * nd  # creates an array of zeros of size nd
            fp = [0] * nd
            for idx, prediction in enumerate(predictions_data):
                file_id = prediction["file_id"]
                # assign prediction to ground truth object if any
                #   open ground-truth with that file_id
                gt_file = os.path.join(path_to_eval_data, tmp_files_path, "{0}_ground_truth.json".format(file_id))
                ground_truth_data = json.load(open(gt_file))
                ovmax = -1
                gt_match = -1
                # load prediction bounding-box
                bb = [float(x) for x in prediction["bbox"].split()]
                for obj in ground_truth_data:
                    # look for a class_name match
                    if obj["class_name"] == class_name:
                        bbgt = [float(x) for x in obj["bbox"].split()]
                        bi = [max(bb[0], bbgt[0]), max(bb[1], bbgt[1]), min(bb[2], bbgt[2]), min(bb[3], bbgt[3])]
                        iw = bi[2] - bi[0] + 1
                        ih = bi[3] - bi[1] + 1
                        if iw > 0 and ih > 0:
                            # compute overlap (IoU) = area of intersection / area of union
                            ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
                                                                              + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
                            ov = iw * ih / ua
                            if ov > ovmax:
                                ovmax = ov
                                gt_match = obj

                # set minimum overlap
                min_overlap = MIN_OVERLAP
                if ovmax >= min_overlap:
                    if "difficult" not in gt_match:
                        if not bool(gt_match["used"]):
                            # true positive
                            tp[idx] = 1
                            gt_match["used"] = True
                            count_true_positives[class_name] += 1
                            # update the ".json" file
                            with open(gt_file, 'w') as f:
                                f.write(json.dumps(ground_truth_data))
                        else:
                            # false positive (multiple detection)
                            fp[idx] = 1
                else:
                    # false positive
                    fp[idx] = 1

            # print(tp)
            # compute precision/recall
            cumsum = 0
            for idx, val in enumerate(fp):
                fp[idx] += cumsum
                cumsum += val
            cumsum = 0
            for idx, val in enumerate(tp):
                tp[idx] += cumsum
                cumsum += val
            # print(tp)
            rec = tp[:]
            for idx, val in enumerate(tp):
                rec[idx] = float(tp[idx]) / gt_counter_per_class[class_name]
            # print(rec)
            prec = tp[:]
            for idx, val in enumerate(tp):
                prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])
            #print(prec)

            ap, mrec, mprec = voc_ap(rec, prec)
            sum_AP += ap
            text = "{0:.2f}%".format(
                ap * 100) + " = " + class_name + " AP  "  # class_name + " AP = {0:.2f}%".format(ap*100)
            """
             Write to results.txt
            """
            rounded_prec = ['%.2f' % elem for elem in prec]
            rounded_rec = ['%.2f' % elem for elem in rec]
            results_file.write(
                text + "\n Precision: " + str(rounded_prec) + "\n Recall   :" + str(rounded_rec) + "\n\n")
            ap_dictionary[class_name] = ap

        results_file.write("\n# mAP of all classes\n")
        mAP = sum_AP / n_classes
        text = "mAP = {0:.2f}%".format(mAP * 100)
        map_global = mAP
        results_file.write(text + "\n")
        print(text)

    # remove the tmp_files directory
    shutil.rmtree(os.path.join(path_to_eval_data, tmp_files_path))

    """
     Count total of Predictions
    """
    # iterate through all the files
    pred_counter_per_class = {}
    # all_classes_predicted_files = set([])
    for txt_file in predicted_files_list:
        # get lines to list
        lines_list = file_lines_to_list(txt_file)
        for line in lines_list:
            class_name = line.split()[0]
            # check if class is in the ignore list, if yes skip
            if class_name in ignore_classes:
                continue
            # count that object
            if class_name in pred_counter_per_class:
                pred_counter_per_class[class_name] += 1
            else:
                # if class didn't exist yet
                pred_counter_per_class[class_name] = 1
    # print(pred_counter_per_class)
    pred_classes = list(pred_counter_per_class.keys())

    """
     Write number of ground-truth objects per class to results.txt
    """
    with open(os.path.join(path_to_eval_data, results_files_path, "results.txt"), 'a') as results_file:
        results_file.write("\n# Number of ground-truth objects per class\n")
        for class_name in sorted(gt_counter_per_class):
            results_file.write(class_name + ": " + str(gt_counter_per_class[class_name]) + "\n")

    """
     Finish counting true positives
    """
    for class_name in pred_classes:
        # if class exists in predictions but not in ground-truth then there are no true positives in that class
        if class_name not in gt_classes:
            count_true_positives[class_name] = 0
    # print(count_true_positives)

    """
     Write number of predicted objects per class to results.txt
    """
    with open(os.path.join(path_to_eval_data, results_files_path, "results.txt"), 'a') as results_file:
        results_file.write("\n# Number of predicted objects per class\n")
        for class_name in sorted(pred_classes):
            n_pred = pred_counter_per_class[class_name]
            text = class_name + ": " + str(n_pred)
            text += " (tp:" + str(count_true_positives[class_name]) + ""
            text += ", fp:" + str(n_pred - count_true_positives[class_name]) + ")\n"
            results_file.write(text)

    # add the mean ap to the dictionary
    ap_dictionary['mAP'] = map_global

    return ap_dictionary
