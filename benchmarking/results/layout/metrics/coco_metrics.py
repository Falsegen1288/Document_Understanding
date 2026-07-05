# layout_benchmark/metrics/coco_metrics.py
import os
import json
import tempfile
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / (areaA + areaB - inter)

def match_predictions(gt_boxes, pred_boxes, iou_threshold=0.5):
    candidates = []
    for gi, gt in enumerate(gt_boxes):
        for pi, pred in enumerate(pred_boxes):
            if gt['label'] == pred['label']:
                iou = compute_iou(gt['bbox'], pred['bbox'])
                if iou >= iou_threshold:
                    candidates.append((iou, gi, pi))
    candidates.sort(key=lambda x: -x[0])

    used_gt, used_pred, matched = set(), set(), []
    for iou, gi, pi in candidates:
        if gi not in used_gt and pi not in used_pred:
            matched.append((gi, pi, iou))
            used_gt.add(gi); used_pred.add(pi)

    unmatched_gt = [i for i in range(len(gt_boxes)) if i not in used_gt]
    unmatched_pred = [i for i in range(len(pred_boxes)) if i not in used_pred]
    return matched, unmatched_gt, unmatched_pred

def run_cocoeval(all_gt, all_preds, valid_classes, img_sizes=None):
    class_to_id = {cls: idx + 1 for idx, cls in enumerate(sorted(list(valid_classes)))}
    categories = [{'id': idx, 'name': name} for name, idx in class_to_id.items()]
    
    gt_coco = {'images': [], 'annotations': [], 'categories': categories}
    pred_coco = []
    
    ann_id = 1
    for img_idx, (gt_boxes, pred_boxes) in enumerate(zip(all_gt, all_preds)):
        gt_f = [b for b in gt_boxes if b['label'] in valid_classes]
        pr_f = [b for b in pred_boxes if b['label'] in valid_classes]
        
        img_w, img_h = 1000, 1000
        if img_sizes and img_idx < len(img_sizes):
            img_w, img_h = img_sizes[img_idx]
            
        gt_coco['images'].append({'id': img_idx, 'width': img_w, 'height': img_h})
        
        for b in gt_f:
            x0, y0, x1, y1 = b['bbox']
            w, h = max(0, x1 - x0), max(0, y1 - y0)
            gt_coco['annotations'].append({
                'id': ann_id, 'image_id': img_idx,
                'category_id': class_to_id[b['label']],
                'bbox': [x0, y0, w, h], 'area': w * h, 'iscrowd': 0
            })
            ann_id += 1
            
        for b in pr_f:
            x0, y0, x1, y1 = b['bbox']
            w, h = max(0, x1 - x0), max(0, y1 - y0)
            pred_coco.append({
                'image_id': img_idx, 'category_id': class_to_id[b['label']],
                'bbox': [x0, y0, w, h], 'score': b.get('score', 0.9), 'area': w * h
            })
            
    if not gt_coco['annotations']:
        return {'mAP50': 0.0, 'mAP5095': 0.0}
        
    with tempfile.NamedTemporaryFile(suffix='.json', mode='w', delete=False) as f_gt:
        json.dump(gt_coco, f_gt)
        gt_path = f_gt.name
        
    if not pred_coco:
        os.unlink(gt_path)
        return {'mAP50': 0.0, 'mAP5095': 0.0}
        
    with tempfile.NamedTemporaryFile(suffix='.json', mode='w', delete=False) as f_pred:
        json.dump(pred_coco, f_pred)
        pred_path = f_pred.name
        
    try:
        import io, sys
        stdout = sys.stdout
        sys.stdout = io.StringIO()
        
        coco_gt = COCO(gt_path)
        coco_dt = coco_gt.loadRes(pred_path)
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.params.catIds = list(class_to_id.values())
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        sys.stdout = stdout
        ap5095 = max(0.0, coco_eval.stats[0])
        ap50 = max(0.0, coco_eval.stats[1])
    except Exception as e:
        print(f"COCOeval failed: {e}. Falling back to manual AP.")
        ap50, ap5095 = 0.0, 0.0
    finally:
        if os.path.exists(gt_path): os.unlink(gt_path)
        if os.path.exists(pred_path): os.unlink(pred_path)
        
    return {'mAP50': ap50, 'mAP5095': ap5095}

def evaluate_coco_metrics(all_gt, all_preds, valid_classes, img_sizes=None):
    """Compute overall P, R, F1, mean IoU at IoU=0.50, and mAP via COCOeval."""
    coco_res = run_cocoeval(all_gt, all_preds, valid_classes, img_sizes)
    
    tp, fp, fn = 0, 0, 0
    ious = []
    for gt_boxes, pred_boxes in zip(all_gt, all_preds):
        gt_f = [b for b in gt_boxes if b['label'] in valid_classes]
        pr_f = [b for b in pred_boxes if b['label'] in valid_classes]
        matched, ugt, upr = match_predictions(gt_f, pr_f, 0.5)
        tp += len(matched)
        fn += len(ugt)
        fp += len(upr)
        for _, _, iou in matched:
            ious.append(iou)
            
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    mean_iou = np.mean(ious) if ious else 0.0
    
    return {
        'mAP50': coco_res['mAP50'],
        'mAP5095': coco_res['mAP5095'],
        'precision': p,
        'recall': r,
        'F1': f1,
        'mean_iou': mean_iou
    }
