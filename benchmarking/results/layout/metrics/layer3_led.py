# layout_benchmark/metrics/layer3_led.py
import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import defaultdict

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / (areaA + areaB - inter)

def evaluate_layer3(ref_dets, pred_dets, class_map):
    """
    Classify predictions and references into LED error types per page.
    Applying label mapping to resolve taxonomy mismatches before error checks.
    """
    # Map raw labels to canonical taxonomy
    ref_mapped = [{"bbox": d["bbox"], "label": class_map.get(d["label"], d["label"])} for d in ref_dets]
    pred_mapped = [{"bbox": d["bbox"], "label": class_map.get(d["label"], d["label"])} for d in pred_dets]
    
    n_ref = len(ref_mapped)
    n_pred = len(pred_mapped)
    
    errors = {
        'Missing': 0, 'Hallucination': 0, 'Size-Error': 0, 'Split': 0,
        'Merge': 0, 'Overlap-Pred': 0, 'Duplicate': 0, 'Misclassification': 0
    }
    
    if n_ref == 0:
        errors['Hallucination'] = n_pred
        return errors
    if n_pred == 0:
        errors['Missing'] = n_ref
        return errors
        
    iou_mat = np.zeros((n_ref, n_pred))
    for i, rd in enumerate(ref_mapped):
        for j, pd in enumerate(pred_mapped):
            iou_mat[i, j] = compute_iou(rd['bbox'], pd['bbox'])
            
    # 1. Duplicate: two pred boxes nearly identical (IoU >= 0.85)
    dup_pairs = []
    for j1 in range(n_pred):
        for j2 in range(j1+1, n_pred):
            if compute_iou(pred_mapped[j1]['bbox'], pred_mapped[j2]['bbox']) >= 0.85:
                dup_pairs.append((j1, j2))
    errors['Duplicate'] = len(dup_pairs)
    
    # 2. Overlap-Pred: pred boxes overlapping significantly (0.10 <= IoU < 0.85)
    overlap_pairs = []
    for j1 in range(n_pred):
        for j2 in range(j1+1, n_pred):
            iou_pp = compute_iou(pred_mapped[j1]['bbox'], pred_mapped[j2]['bbox'])
            if 0.10 <= iou_pp < 0.85:
                overlap_pairs.append((j1, j2))
    errors['Overlap-Pred'] = len(overlap_pairs)
    
    # 3. Missing: GT box with no pred overlapping at IoU >= 0.10
    missing_indices = []
    for i in range(n_ref):
        if iou_mat[i].max() < 0.10:
            missing_indices.append(i)
    errors['Missing'] = len(missing_indices)
    
    # 4. Hallucination: Pred box with no GT overlapping at IoU >= 0.10
    hallu_indices = []
    for j in range(n_pred):
        if iou_mat[:, j].max() < 0.10:
            hallu_indices.append(j)
    errors['Hallucination'] = len(hallu_indices)
    
    # Bipartite matching for remaining analysis
    cost = 1.0 - iou_mat
    ri, ci = linear_sum_assignment(cost)
    matched_ref = set()
    matched_pred = set()
    valid_matches = []
    
    for r, c in zip(ri, ci):
        if iou_mat[r, c] >= 0.10:
            matched_ref.add(r)
            matched_pred.add(c)
            valid_matches.append((r, c, iou_mat[r, c]))
            
    # 5. Misclassification: IoU >= 0.50 and label mismatch
    misclass_pairs = []
    for r, c, iou in valid_matches:
        if iou >= 0.50 and ref_mapped[r]['label'] != pred_mapped[c]['label']:
            misclass_pairs.append((r, c))
    errors['Misclassification'] = len(misclass_pairs)
    
    # 6. Size-Error: Matched but 0.10 <= IoU < 0.50
    size_errors = []
    for r, c, iou in valid_matches:
        if 0.10 <= iou < 0.50:
            size_errors.append((r, c))
    errors['Size-Error'] = len(size_errors)
    
    # 7. Split: one ref box overlapping multiple pred boxes (IoU >= 0.10)
    ref_to_preds = defaultdict(list)
    for r, c, iou in valid_matches:
        ref_to_preds[r].append(c)
    for j in range(n_pred):
        for i in range(n_ref):
            if iou_mat[i, j] >= 0.10 and j not in matched_pred:
                ref_to_preds[i].append(j)
    splits = [r for r, preds in ref_to_preds.items() if len(preds) > 1]
    errors['Split'] = len(splits)
    
    # 8. Merge: one pred box overlapping multiple ref boxes (IoU >= 0.10)
    pred_to_refs = defaultdict(list)
    for r, c, iou in valid_matches:
        pred_to_refs[c].append(r)
    for i in range(n_ref):
        for j in range(n_pred):
            if iou_mat[i, j] >= 0.10 and i not in matched_ref:
                pred_to_refs[j].append(i)
    merges = [c for c, refs in pred_to_refs.items() if len(refs) > 1]
    errors['Merge'] = len(merges)
    
    return errors
