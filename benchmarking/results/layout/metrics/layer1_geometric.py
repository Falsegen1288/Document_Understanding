# layout_benchmark/metrics/layer1_geometric.py
import os
import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / (areaA + areaB - inter)

# Load padding margins from audit CSV
def load_margins():
    audit_path = r"d:\antigravity\benchmarking\layout_benchmark\results\gt_padding_audit.csv"
    margins = {}
    if os.path.exists(audit_path):
        df = pd.read_csv(audit_path, index_col=0 if "class" in pd.read_csv(audit_path).columns else None)
        if "class" in df.columns:
            df = df.set_index("class")
        for cls, row in df.iterrows():
            margins[cls] = {
                "left": row.get("pad_left_mean", 0.05),
                "right": row.get("pad_right_mean", 0.05),
                "top": row.get("pad_top_mean", 0.05),
                "bottom": row.get("pad_bottom_mean", 0.05)
            }
    return margins

MARGINS = load_margins()

def erode_box(box, label):
    """Erode a bounding box [x0, y0, x1, y1] inward by class-conditional margins."""
    x0, y0, x1, y1 = box
    w = x1 - x0
    h = y1 - y0
    
    # Fallback to flat 5%
    m = MARGINS.get(label, {"left": 0.05, "right": 0.05, "top": 0.05, "bottom": 0.05})
    
    x0_e = x0 + m["left"] * w
    x1_e = x1 - m["right"] * w
    y0_e = y0 + m["top"] * h
    y1_e = y1 - m["bottom"] * h
    
    # Ensure box doesn't invert
    if x0_e >= x1_e:
        x0_e, x1_e = x0 + 0.45 * w, x1 - 0.45 * w
    if y0_e >= y1_e:
        y0_e, y1_e = y0 + 0.45 * h, y1 - 0.45 * h
        
    return [x0_e, y0_e, x1_e, y1_e]

def build_iou_matrix(gt_boxes, pred_boxes, tolerant=False):
    mat = np.zeros((len(gt_boxes), len(pred_boxes)))
    for i, gt in enumerate(gt_boxes):
        gt_bbox = erode_box(gt["bbox"], gt["label"]) if tolerant else gt["bbox"]
        for j, pred in enumerate(pred_boxes):
            mat[i, j] = compute_iou(gt_bbox, pred["bbox"])
    return mat

def hungarian_match(iou_mat, threshold=0.5):
    if iou_mat.size == 0:
        return []
    cost = 1.0 - iou_mat
    row_ind, col_ind = linear_sum_assignment(cost)
    matches = []
    for r, c in zip(row_ind, col_ind):
        if iou_mat[r, c] >= threshold:
            matches.append((r, c))
    return matches

def evaluate_layer1(gt_list, pred_list, class_map, tolerant=False):
    """
    Compute Layer 1 evaluation metrics.
    gt_list, pred_list: lists of page elements (where each element has 'bbox' and 'label')
    class_map: raw_label -> canonical_label dict mapping
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0
    correct_labels = 0
    all_ious = []
    
    for gt_boxes, pred_boxes in zip(gt_list, pred_list):
        # Apply label map
        mapped_gt = [{"bbox": b["bbox"], "label": class_map.get(b["label"], b["label"])} for b in gt_boxes]
        mapped_pred = [{"bbox": b["bbox"], "label": class_map.get(b["label"], b["label"])} for b in pred_boxes]
        
        # Build Hungarian bipartite matching
        iou_mat = build_iou_matrix(mapped_gt, mapped_pred, tolerant)
        matches = hungarian_match(iou_mat, threshold=0.5)
        
        tp = len(matches)
        fn = len(mapped_gt) - tp
        fp = len(mapped_pred) - tp
        
        total_tp += tp
        total_fn += fn
        total_fp += fp
        
        for r, c in matches:
            all_ious.append(iou_mat[r, c])
            if mapped_gt[r]["label"] == mapped_pred[c]["label"]:
                correct_labels += 1
                
    p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    mean_iou = np.mean(all_ious) if all_ious else 0.0
    class_acc = correct_labels / total_tp if total_tp else 0.0
    
    return {
        "precision": p,
        "recall": r,
        "F1": f1,
        "mean_iou": mean_iou,
        "class_acc": class_acc
    }
