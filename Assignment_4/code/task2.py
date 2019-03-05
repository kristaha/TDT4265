import numpy as np
import matplotlib.pyplot as plt
import json
import copy
from task2_tools import read_predicted_boxes, read_ground_truth_boxes

def calculate_iou(prediction_box, gt_box):
    """Calculate intersection over union of single predicted and ground truth box.

    Args:
        prediction_box (np.array of floats): location of predicted object as
            [xmin, ymin, xmax, ymax]
        gt_box (np.array of floats): location of ground truth object as
            [xmin, ymin, xmax, ymax]

        returns:
            float: value of the intersection of union for the two boxes.
    """
    intersection = (( min(prediction_box[2], gt_box[2]) - max(prediction_box[0], gt_box[0]))* 
            (min(prediction_box[3], gt_box[3]) - max(prediction_box[1], gt_box[1])))

    union = ( ((prediction_box[2] - prediction_box[0])*(prediction_box[3] - prediction_box[1])) +
            ((gt_box[2] - gt_box[0])*(gt_box[3] - gt_box[1])) - intersection)

    iou = intersection/union

    if iou < 0:
        return 0.0
    return iou


def calculate_precision(num_tp, num_fp, num_fn):
    """ Calculates the precision for the given parameters.
        Returns 1 if num_tp + num_fp = 0

    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of precision
    """
    if num_tp + num_fp == 0:
        return 1

    return num_tp / (num_tp + num_fp)


def calculate_recall(num_tp, num_fp, num_fn):
    """ Calculates the recall for the given parameters.
        Returns 0 if num_tp + num_fn = 0
    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of recall
    """
    if num_tp + num_fn == 0:
        return 0

    return num_tp / (num_tp + num_fn)


def get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold):
    """Finds all possible matches for the predicted boxes to the ground truth boxes.
        No bounding box can have more than one match.

        Remember: Matching of bounding boxes should be done with decreasing IoU order!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns the matched boxes (in corresponding order):
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of box matches, 4].
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of box matches, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    """
    ious = []
    possible_box_match_indices = [] # Array with [pred_box index, gt_box index]

    # Find all possible matches with a IoU >= iou threshold
    for i in range(prediction_boxes.shape[0]):
        for j in range(gt_boxes.shape[0]):
             iou = calculate_iou(prediction_boxes[i], gt_boxes[j])
             if iou >= iou_threshold:
                 ious.append(iou)
                 possible_box_match_indices.append([i, j])

    ious = np.array(ious)
    possible_box_match_indices = np.array(possible_box_match_indices) #Sjekk shape, ev om man må bruke np.ndarray

    # Sort all matches on IoU in descending order
    ious_index = ious.argsort()[::-1]
    ious = ious[ious_index]
    possible_box_match_indices = possible_box_match_indices[ious_index]

    #Har nå ious i synkron synkende sortering med korresponderende box indexer

    # Find all matches with the highest IoU threshold
    matched_prediction_box_indices = []
    matched_gt_box_indices = []

    for i in range(ious.size):
        pred_index = possible_box_match_indices[i][0] 
        gt_index = possible_box_match_indices[i][1]

        if not (pred_index in matched_prediction_box_indices or 
                gt_index in matched_gt_box_indices):

            matched_prediction_box_indices.append(pred_index)
            matched_gt_box_indices.append(gt_index)

    matched_prediction_boxes = prediction_boxes[matched_prediction_box_indices]
    matched_gt_boxes = gt_boxes[matched_gt_box_indices]

    return matched_prediction_boxes, matched_gt_boxes

def calculate_individual_image_result(
        prediction_boxes, gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes,
       calculates true positives, false positives and false negatives
       for a single image.
       NB: prediction_boxes and gt_boxes are not matched!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns:
        dict: containing true positives, false positives, true negatives, false negatives
            {"true_pos": int, "false_pos": int, "false_neg": int}
    """
    # Find the bounding box matches with the highes IoU threshold
    matched_predictions, matched_gt = get_all_box_matches(prediction_boxes,
            gt_boxes, iou_threshold)

    # Compute true positives, false positives, false negatives
    results = {
            "true_pos" : 0, 
            "false_pos" : 0,
            "false_neg" : 0
            }

    results["true_pos"] = int(matched_gt.shape[0])
    results["false_pos"] = int(prediction_boxes.shape[0] - matched_predictions.shape[0])
    results["false_neg"] = int(gt_boxes.shape[0] - matched_gt.shape[0])

    return results


def calculate_precision_recall_all_images(
        all_prediction_boxes, all_gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates recall and precision over all images
       for a single image.
       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
    Returns:
        tuple: (precision, recall). Both float.
    """
    # Find total true positives, false positives and false negatives
    # over all images
    true_pos_total = 0
    false_pos_total = 0
    false_neg_total = 0

    for i in range(len(all_prediction_boxes)):
        result_dict = calculate_individual_image_result(
                all_prediction_boxes[i], all_gt_boxes[i], iou_threshold)
        true_pos_total += int(result_dict.get("true_pos"))
        false_pos_total += int(result_dict.get("false_pos"))
        false_neg_total += int(result_dict.get("false_neg"))

    # Compute precision, recall

    precision = calculate_precision(true_pos_total, false_pos_total, false_neg_total)
    recall = calculate_recall(true_pos_total, false_pos_total, false_neg_total)

    return (precision, recall)


def get_precision_recall_curve(all_prediction_boxes, all_gt_boxes,
                               confidence_scores, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates the recall-precision curve over all images.
       for a single image.

       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
        scores: (list of np.array of floats): each element in the list
            is a np.array containting the confidence score for each of the
            predicted bounding box. Shape: [number of predicted boxes]

            E.g: score[0][1] is the confidence score for a predicted bounding box 1 in image 0.
    Returns:
        array of tuples: (precision, recall). Both float.
    """
    # Instead of going over every possible confidence score threshold to compute the PR
    # curve, we will use an approximation
    # DO NOT CHANGE. If you change this, the tests will not pass when we run the final
    # evaluation
    confidence_thresholds = np.linspace(0, 1, 500)

    # YOUR CODE HERE
    #Finds prediction boxses with confidence score > confidence_thresholds
    filtering_mask = np.greater(confidence_scores, confidence_thresholds)
    #confident_prediction_box_indices = all_prediction_boxes[filtering_mask]
    prediction_boxes = all_prediction_boxes[filtering_mask]

    #Hva vil de ha som output format? gir ikke mening?
    raise NotImplementedError

    


def plot_precision_recall_curve(precisions, recalls):
    """Plots the precision recall curve.
        Save the figure to precision_recall_curve.png:
        'plt.savefig("precision_recall_curve.png")'

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        None
    """
    # No need to edit this code.
    plt.figure(figsize=(20, 20))
    plt.plot(recalls, precisions)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim([0.8, 1.0])
    plt.ylim([0.8, 1.0])
    plt.savefig("precision_recall_curve.png")


def calculate_mean_average_precision(precisions, recalls):
    """ Given a precision recall curve, calculates the mean average
        precision.

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        float: mean average precision
    """
    # Calculate the mean average precision given these recall levels.
    # DO NOT CHANGE. If you change this, the tests will not pass when we run the final
    # evaluation
    recall_levels = np.linspace(0, 1.0, 11)
    # YOUR CODE HERE
    raise NotImplementedError


def mean_average_precision(ground_truth_boxes, predicted_boxes):
    """ Calculates the mean average precision over the given dataset
        with IoU threshold of 0.5

    Args:
        ground_truth_boxes: (dict)
        {
            "img_id1": (np.array of float). Shape [number of GT boxes, 4]
        }
        predicted_boxes: (dict)
        {
            "img_id1": {
                "boxes": (np.array of float). Shape: [number of pred boxes, 4],
                "scores": (np.array of float). Shape: [number of pred boxes]
            }
        }
    """
    # DO NOT EDIT THIS CODE
    all_gt_boxes = []
    all_prediction_boxes = []
    confidence_scores = []

    for image_id in ground_truth_boxes.keys():
        pred_boxes = predicted_boxes[image_id]["boxes"]
        scores = predicted_boxes[image_id]["scores"]

        all_gt_boxes.append(ground_truth_boxes[image_id])
        all_prediction_boxes.append(pred_boxes)
        confidence_scores.append(scores)
    iou_threshold = 0.5
    precisions, recalls = get_precision_recall_curve(all_prediction_boxes,
                                                     all_gt_boxes,
                                                     confidence_scores,
                                                     iou_threshold)
    plot_precision_recall_curve(precisions, recalls)
    mean_average_precision = calculate_mean_average_precision(precisions,
                                                              recalls)
    print("Mean average precision: {:.4f}".format(mean_average_precision))


if __name__ == "__main__":
    ground_truth_boxes = read_ground_truth_boxes()
    predicted_boxes = read_predicted_boxes()
    mean_average_precision(ground_truth_boxes, predicted_boxes)

#Spørsmål til studass:
    # Hva er det de vil at outputen til calculate_precision_recall... og calc_precision_recall_curve? 
    # Skjønner ikke hva de er ute etter?
    # Hva skal confidence brukes til? Hvordan ev matche den med riktig bilde?
