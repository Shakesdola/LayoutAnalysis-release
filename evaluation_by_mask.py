import cv2
import numpy as np
import os

#Count True Positives (TP), False Positives (FP), False Negatives (FN) for a given class color
def count_tp_fp_fn(gt_img, pred_img, color):
    # Initialize counts
    tp_count = fp_count = fn_count = 0

    # Check TP, FP, FN for each pixel
    for y in range(gt_img.shape[0]):
        for x in range(gt_img.shape[1]):
            gt_pixel = gt_img[y, x]
            pred_pixel = pred_img[y, x]

            # Check if the pixel color matches the given color
            if np.all(gt_pixel == color):
                # Check if the corresponding pixel in pred image matches the color
                if np.all(pred_pixel == color):
                    tp_count += 1  # True Positive
                else:
                    fn_count += 1  # False Negative
            # If the pred pixel match the color while the gt_pixel not match, it means FP
            elif np.all(pred_pixel == color):
                fp_count += 1  # False Positive

    return tp_count, fp_count, fn_count

#Calculate Intersection over Union (IoU) using given TP, FP, FN
def calculate_iou(tp, fp, fn):
    if tp + fp + fn == 0:
        return 0
    else:
        return tp / (tp + fp + fn)

def calculate_precision(tp, fp, fn):
    if tp + fp + fn == 0:
        return 0
    else:
        return tp / (tp + fp)
    
def calculate_recall(tp, fp, fn):
    if tp + fp + fn == 0:
        return 0
    else:
        return tp / (tp + fn)

#Evaluate IoU for each class and print the results.
def evaluate_iou(gt_mask_folder, pred_mask_folder):
    colors = [[0,0,255], [0,255,255], [255,0,0], [255,0,255]]
    
    # Get list of file names in gt_mask_folder
    gt_mask_files = [f for f in os.listdir(gt_mask_folder) if os.path.isfile(os.path.join(gt_mask_folder, f))]
    
    for gt_file in gt_mask_files:
        gt_img_path = os.path.join(gt_mask_folder, gt_file)
        pred_img_path = os.path.join(pred_mask_folder, gt_file)
        gt_img = cv2.imread(gt_img_path, cv2.IMREAD_COLOR)
        pred_img = cv2.imread(pred_img_path, cv2.IMREAD_COLOR)

        print(f"Evaluating IoU for {gt_file}:")
        for color in colors:
            # Count TP, FP, FN
            tp, fp, fn = count_tp_fp_fn(gt_img, pred_img, color)
            # Calculate IoU
            iou = calculate_iou(tp, fp, fn)
            pre = calculate_precision(tp, fp, fn)
            recall = calculate_recall(tp, fp, fn)
            # Print IoU for the class
            print(f"Class {color}, IoU = {iou}, Precision = {pre}, recall = {recall}")

# Example usage
gt_mask_folder = 'ground_truth_prediction'
pred_mask_folder = 'pred_gen_masks'
evaluate_iou(gt_mask_folder, pred_mask_folder)
