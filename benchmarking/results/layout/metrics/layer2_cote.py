# layout_benchmark/metrics/layer2_cote.py
import numpy as np

def bbox_to_mask(bbox, page_w, page_h, scale=0.2):
    h = int(page_h * scale)
    w = int(page_w * scale)
    mask = np.zeros((h, w), dtype=bool)
    x0 = max(0, int(bbox[0] * scale))
    y0 = max(0, int(bbox[1] * scale))
    x1 = min(w, int(bbox[2] * scale))
    y1 = min(h, int(bbox[3] * scale))
    mask[y0:y1, x0:x1] = True
    return mask

def evaluate_layer2(gt_boxes, pred_boxes, page_w, page_h, scale=0.2):
    """
    Compute COTe area metrics on page-sized binary masks as defined in Section 5.
    Formula:
      Coverage = Area(GT ∩ anyPred) / Area(GT)
      Overlap  = Area(GT ∩ doubly_covered) / Area(Pred)
      Trespass = Area(Pred outside GT) / Area(Pred)
      Excess   = Area(Pred) / Area(GT)
    """
    H = int(page_h * scale)
    W = int(page_w * scale)
    
    # 1. Rasterize GT boxes into a single binary union mask
    gt_mask = np.zeros((H, W), dtype=bool)
    for gt in gt_boxes:
        gt_mask |= bbox_to_mask(gt["bbox"], page_w, page_h, scale)
        
    # 2. Rasterize predicted boxes
    pred_masks = [bbox_to_mask(pr["bbox"], page_w, page_h, scale) for pr in pred_boxes]
    
    # Union of all predictions
    pred_union = np.zeros((H, W), dtype=bool)
    # Sum of predictions to identify doubly-covered areas
    pred_sum = np.zeros((H, W), dtype=np.int16)
    
    for pm in pred_masks:
        pred_union |= pm
        pred_sum += pm.astype(np.int16)
        
    # 3. Calculate areas
    area_gt = float(np.sum(gt_mask))
    area_pred = float(np.sum(pred_union))
    
    if area_gt == 0:
        return {
            "Coverage": 0.0,
            "Overlap": 0.0,
            "Trespass": 1.0 if area_pred > 0 else 0.0,
            "Excess": 0.0
        }
        
    # Coverage = Area(GT ∩ anyPred) / Area(GT)
    intersection = gt_mask & pred_union
    area_intersection = float(np.sum(intersection))
    coverage = area_intersection / area_gt
    
    # Overlap = Area(GT ∩ doubly-covered) / Area(Pred)
    doubly_covered = gt_mask & (pred_sum > 1)
    area_overlap = float(np.sum(doubly_covered))
    overlap = area_overlap / area_pred if area_pred > 0 else 0.0
    
    # Trespass = Area(Pred outside GT) / Area(Pred)
    pred_outside_gt = pred_union & (~gt_mask)
    area_trespass = float(np.sum(pred_outside_gt))
    trespass = area_trespass / area_pred if area_pred > 0 else 0.0
    
    # Excess = Area(Pred) / Area(GT)
    excess = area_pred / area_gt
    
    return {
        "Coverage": coverage,
        "Overlap": overlap,
        "Trespass": trespass,
        "Excess": excess
    }
