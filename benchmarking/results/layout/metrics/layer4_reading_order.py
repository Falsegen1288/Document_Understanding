# layout_benchmark/metrics/layer4_reading_order.py
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

def assign_reading_order(detections, row_band=40.0):
    """
    Sort boxes into a reading order (top-to-bottom, left-to-right).
    row_band: vertical threshold in coordinate units (px or pt) to group into same row.
    """
    if not detections:
        return []
        
    # Calculate centroids
    centroids = []
    for d in detections:
        x0, y0, x1, y1 = d["bbox"]
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        centroids.append((d, cx, cy))
        
    # Primary sort: vertical (cy)
    centroids.sort(key=lambda t: t[2])
    
    ordered = []
    current_band_y = None
    current_band = []
    
    for item in centroids:
        d, cx, cy = item
        if current_band_y is None or abs(cy - current_band_y) <= row_band:
            current_band.append(item)
            # Update running average cy of current row band
            current_band_y = cy if current_band_y is None else (
                (current_band_y * (len(current_band) - 1) + cy) / len(current_band)
            )
        else:
            # Sort current row band left-to-right
            current_band.sort(key=lambda t: t[1])
            ordered.extend(current_band)
            current_band = [item]
            current_band_y = cy
            
    if current_band:
        current_band.sort(key=lambda t: t[1])
        ordered.extend(current_band)
        
    # Return list of (d, order_index)
    return [(item[0], idx) for idx, item in enumerate(ordered)]

def evaluate_layer4(gt_boxes, pred_boxes, class_map, row_band=40.0):
    """
    Evaluate ROKT (Kendall's Tau) and Segment ROA.
    Uses Hungarian matching at IoU >= 0.30 to identify corresponding pairs.
    """
    # 1. Map raw labels
    mapped_gt = [{"bbox": d["bbox"], "label": class_map.get(d["label"], d["label"])} for d in gt_boxes]
    mapped_pred = [{"bbox": d["bbox"], "label": class_map.get(d["label"], d["label"])} for d in pred_boxes]
    
    if not mapped_gt or not mapped_pred:
        return {"tau": float('nan'), "roa": float('nan'), "n_matched": 0}
        
    # 2. Hungarian matching on IoU matrix to find corresponding boxes
    n_gt = len(mapped_gt)
    n_pred = len(mapped_pred)
    iou_mat = np.zeros((n_gt, n_pred))
    for i, gt in enumerate(mapped_gt):
        for j, pr in enumerate(pred_mapped if 'pred_mapped' in locals() else mapped_pred):
            iou_mat[i, j] = compute_iou(gt["bbox"], pr["bbox"])
            
    ri, ci = linear_sum_assignment(1.0 - iou_mat)
    
    matched_gt_list = []
    matched_pred_list = []
    for r, c in zip(ri, ci):
        if iou_mat[r, c] >= 0.30:
            matched_gt_list.append(mapped_gt[r])
            matched_pred_list.append(mapped_pred[c])
            
    n = len(matched_gt_list)
    if n < 2:
        return {"tau": float('nan'), "roa": float('nan'), "n_matched": n}
        
    # 3. Assign reading order on matched pairs
    gt_order = assign_reading_order(matched_gt_list, row_band)
    pred_order = assign_reading_order(matched_pred_list, row_band)
    
    # Key by matching position index
    gt_keyed = {i: r for i, (d, r) in enumerate(gt_order)}
    pred_keyed = {i: r for i, (d, r) in enumerate(pred_order)}
    
    # 4. ROKT (Kendall's Tau)
    C, D = 0, 0
    for i in range(n):
        for j in range(i+1, n):
            gt_diff = gt_keyed[i] - gt_keyed[j]
            pred_diff = pred_keyed[i] - pred_keyed[j]
            if gt_diff * pred_diff > 0:
                C += 1
            elif gt_diff * pred_diff < 0:
                D += 1
                
    denom = n * (n - 1) / 2
    tau = (C - D) / denom if denom > 0 else float('nan')
    
    # 5. ROA (Segment Reading Order Accuracy)
    pred_ranks = [pred_keyed[i] for i in range(n)]
    adjacent_inversions = sum(
        1 for i in range(len(pred_ranks) - 1)
        if pred_ranks[i] > pred_ranks[i+1]
    )
    roa = 1.0 - (adjacent_inversions / (n - 1)) if n > 1 else float('nan')
    
    return {
        "tau": tau,
        "roa": roa,
        "n_matched": n
    }
