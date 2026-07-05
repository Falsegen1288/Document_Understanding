"""
Benchmark helper module for layout detection evaluation.
All dataset loading, inference wrappers, evaluation metrics,
and visualization functions live here so the notebook stays clean.
"""
import os
import re
import json
import random
import tempfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from PIL import Image
from collections import defaultdict
from tqdm.auto import tqdm

# ═══════════════════════════════════════════════════════════
# 1. CLASS MAPPINGS
# ═══════════════════════════════════════════════════════════

CANONICAL = {
    'title', 'text', 'table', 'picture', 'list_item', 'caption',
    'section_header', 'page_footer', 'page_header', 'formula', 'footnote'
}

DOCLAYNET_TO_CANONICAL = {
    'Caption': 'caption', 'Footnote': 'footnote', 'Formula': 'formula',
    'List-item': 'list_item', 'Page-footer': 'page_footer',
    'Page-header': 'page_header', 'Picture': 'picture',
    'Section-header': 'section_header', 'Table': 'table',
    'Text': 'text', 'Title': 'title',
    # lowercase variants
    'caption': 'caption', 'footnote': 'footnote', 'formula': 'formula',
    'list-item': 'list_item', 'list_item': 'list_item',
    'page-footer': 'page_footer', 'page_footer': 'page_footer',
    'page-header': 'page_header', 'page_header': 'page_header',
    'picture': 'picture', 'section-header': 'section_header',
    'section_header': 'section_header', 'table': 'table',
    'text': 'text', 'title': 'title',
    # docling internal names
    'sectionheader': 'section_header', 'listitem': 'list_item',
    'figurecaption': 'caption', 'pageheader': 'page_header',
    'pagefooter': 'page_footer', 'keyvalueitem': 'text',
    'inlinemathexpression': 'formula',
}

PUBLAYNET_TO_CANONICAL = {
    'text': 'text', 'title': 'title', 'list': 'list_item',
    'figure': 'picture', 'table': 'table',
}

DOCBANK_TO_CANONICAL = {
    'abstract': 'text', 'author': 'text', 'caption': 'caption',
    'equation': 'formula', 'figure': 'picture', 'footer': 'page_footer',
    'list': 'list_item', 'paragraph': 'text', 'reference': 'text',
    'section': 'section_header', 'table': 'table', 'title': 'title',
}

ADE_TO_CANONICAL = {
    'text': 'text', 'table': 'table', 'figure': 'picture',
    'logo': 'picture', 'barcode': 'picture', 'attestation': 'text',
}

VALID_CLASSES_PER_DATASET = {
    'DocLayNet': {'caption', 'footnote', 'formula', 'list_item', 'page_footer',
                  'page_header', 'picture', 'section_header', 'table', 'text', 'title'},
    'PubLayNet': {'text', 'title', 'list_item', 'picture', 'table'},
    'DocBank': {'text', 'caption', 'formula', 'picture', 'page_footer',
                'list_item', 'section_header', 'table', 'title'},
}

DLN_CAT_NAMES = {
    1: 'Caption', 2: 'Footnote', 3: 'Formula', 4: 'List-item',
    5: 'Page-footer', 6: 'Page-header', 7: 'Picture',
    8: 'Section-header', 9: 'Table', 10: 'Text', 11: 'Title'
}

PLN_CAT_NAMES = {1: 'text', 2: 'title', 3: 'list', 4: 'table', 5: 'figure'}

DOCBANK_INT_TO_STR = {
    0: 'abstract', 1: 'author', 2: 'caption', 3: 'equation',
    4: 'figure', 5: 'footer', 6: 'list', 7: 'paragraph',
    8: 'reference', 9: 'section', 10: 'table', 11: 'title'
}

# Nemotron constants
PROC_W, PROC_H = 1648, 2048
BLOCK_PATTERN = re.compile(
    r'<x_([0-9.]+)><y_([0-9.]+)>(.*?)'
    r'<x_([0-9.]+)><y_([0-9.]+)><class_([^>]+)>',
    re.DOTALL
)

# ═══════════════════════════════════════════════════════════
# 2. DATASET LOADING
# ═══════════════════════════════════════════════════════════

def _parse_coco_annos(annos, cat_names, label_map):
    """Parse COCO-format annotations (dict-of-lists or list-of-dicts)."""
    gt_boxes = []
    if isinstance(annos, dict):
        n = len(annos.get('bbox', []))
        for i in range(n):
            bbox = annos['bbox'][i]
            cat_id = annos.get('category_id', annos.get('category', [0]))[i]
            cat_name = cat_names.get(cat_id)
            if not cat_name:
                cat_name = cat_names.get(cat_id + 1, 'text')
            canonical = label_map.get(cat_name, None)
            if canonical:
                x, y, w, h = bbox
                gt_boxes.append({'bbox': [x, y, x + w, y + h], 'label': canonical})
    elif isinstance(annos, list):
        for ann in annos:
            bbox = ann['bbox']
            cat_id = ann.get('category_id', ann.get('category', 0))
            cat_name = cat_names.get(cat_id)
            if not cat_name:
                cat_name = cat_names.get(cat_id + 1, 'text')
            canonical = label_map.get(cat_name, None)
            if canonical:
                x, y, w, h = bbox
                gt_boxes.append({'bbox': [x, y, x + w, y + h], 'label': canonical})
    return gt_boxes


def load_doclaynet_samples(n_samples=30, seed=42):
    """Load DocLayNet test split and sample n_samples pages using streaming."""
    from datasets import load_dataset
    import io
    print("Loading DocLayNet (test split, streaming)...")
    dln = load_dataset("ds4sd/DocLayNet", split="test", streaming=True, trust_remote_code=True)
    if seed is not None:
        dln = dln.shuffle(seed=seed, buffer_size=10)

    samples = []
    iterator = iter(dln)
    for _ in tqdm(range(n_samples), desc="DocLayNet"):
        try:
            item = next(iterator)
        except StopIteration:
            break
        pil_img = item['image'].convert('RGB') if isinstance(item['image'], Image.Image) \
            else Image.open(io.BytesIO(item['image'])).convert('RGB')

        gt_boxes = []
        if 'objects' in item and item['objects']:
            for obj in item['objects']:
                bbox = obj['bbox']
                cat_id = obj['category_id']
                cat_name = DLN_CAT_NAMES.get(cat_id)
                if not cat_name:
                    cat_name = DLN_CAT_NAMES.get(cat_id + 1, 'Text')
                canonical = DOCLAYNET_TO_CANONICAL.get(cat_name)
                if canonical:
                    x, y, w, h = bbox
                    gt_boxes.append({'bbox': [x, y, x + w, y + h], 'label': canonical})
        elif 'annotations' in item and item['annotations']:
            gt_boxes = _parse_coco_annos(item['annotations'], DLN_CAT_NAMES, DOCLAYNET_TO_CANONICAL)
        elif 'bboxes' in item:
            cats = item.get('category_id', item.get('categories', []))
            for i, bbox in enumerate(item['bboxes']):
                cat_id = cats[i] if i < len(cats) else 10
                cat_name = DLN_CAT_NAMES.get(cat_id, 'Text')
                canonical = DOCLAYNET_TO_CANONICAL.get(cat_name)
                if canonical:
                    x, y, w, h = bbox
                    gt_boxes.append({'bbox': [x, y, x + w, y + h], 'label': canonical})

        samples.append({'image': pil_img, 'gt': gt_boxes})

    avg = np.mean([len(s['gt']) for s in samples]) if samples else 0
    print(f"  ✅ DocLayNet: {len(samples)} pages, avg {avg:.1f} boxes/page")
    return samples


def load_publaynet_samples(n_samples=30, seed=42):
    """Load PubLayNet val split and sample n_samples pages using streaming."""
    from datasets import load_dataset
    import io
    print("Loading PubLayNet (val split, streaming)...")
    pub = load_dataset("shunk031/PubLayNet",
                        split="validation", trust_remote_code=True, streaming=True,
                        batch_size=10)
    if seed is not None:
        pub = pub.shuffle(seed=seed, buffer_size=10)

    samples = []
    iterator = iter(pub)
    for _ in tqdm(range(n_samples), desc="PubLayNet"):
        try:
            item = next(iterator)
        except StopIteration:
            break
        pil_img = item['image'].convert('RGB') if isinstance(item['image'], Image.Image) \
            else Image.open(io.BytesIO(item['image'])).convert('RGB')

        gt_boxes = _parse_coco_annos(
            item.get('annotations', {}), PLN_CAT_NAMES, PUBLAYNET_TO_CANONICAL)
        samples.append({'image': pil_img, 'gt': gt_boxes})

    avg = np.mean([len(s['gt']) for s in samples]) if samples else 0
    print(f"  ✅ PubLayNet: {len(samples)} pages, avg {avg:.1f} boxes/page")
    return samples


def _aggregate_docbank_tokens(example):
    """Group consecutive tokens of same label into block-level bboxes."""
    words = example.get('words', example.get('tokens', []))
    bboxes = example.get('bboxes', example.get('bbox', []))
    labels = example.get('structures', example.get('labels', example.get('ner_tags', [])))

    if 'pagesize' in example:
        pw, ph = example['pagesize']
    elif 'image' in example and isinstance(example['image'], Image.Image):
        pw, ph = example['image'].size
    else:
        pw, ph = 1000, 1000

    groups = []
    cur_label, cur_bboxes = None, []
    for i in range(len(words)):
        bbox = bboxes[i] if i < len(bboxes) else [0, 0, 0, 0]
        label = labels[i] if i < len(labels) else 'paragraph'
        if isinstance(label, int):
            label = DOCBANK_INT_TO_STR.get(label, 'paragraph')
        if label == cur_label:
            cur_bboxes.append(bbox)
        else:
            if cur_bboxes and cur_label:
                groups.append((cur_label, cur_bboxes))
            cur_label, cur_bboxes = label, [bbox]
    if cur_bboxes and cur_label:
        groups.append((cur_label, cur_bboxes))

    blocks = []
    for label, token_bboxes in groups:
        canonical = DOCBANK_TO_CANONICAL.get(label)
        if not canonical:
            continue
        arr = np.array(token_bboxes, dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 4:
            continue
        x0 = float(arr[:, 0].min()) * pw / 1000
        y0 = float(arr[:, 1].min()) * ph / 1000
        x1 = float(arr[:, 2].max()) * pw / 1000
        y1 = float(arr[:, 3].max()) * ph / 1000
        if (x1 - x0) > 1 and (y1 - y0) > 1:
            blocks.append({'bbox': [x0, y0, x1, y1], 'label': canonical})
    return blocks


def load_docbank_samples(n_samples=30, seed=42):
    """Load DocBank test split and sample n_samples pages using streaming."""
    from datasets import load_dataset
    print("Loading DocBank (test split, streaming)...")
    dkb = load_dataset("liminghao1630/DocBank", split="test", trust_remote_code=True, streaming=True)
    if seed is not None:
        dkb = dkb.shuffle(seed=seed, buffer_size=10)

    samples = []
    iterator = iter(dkb)
    for _ in tqdm(range(n_samples), desc="DocBank"):
        try:
            item = next(iterator)
        except StopIteration:
            break
        if 'image' in item and isinstance(item['image'], Image.Image):
            pil_img = item['image'].convert('RGB')
        else:
            pw = item.get('pagesize', [612, 792])[0] if 'pagesize' in item else 612
            ph = item.get('pagesize', [612, 792])[1] if 'pagesize' in item else 792
            pil_img = Image.new('RGB', (int(pw), int(ph)), 'white')

        gt_boxes = _aggregate_docbank_tokens(item)
        samples.append({'image': pil_img, 'gt': gt_boxes})

    avg = np.mean([len(s['gt']) for s in samples]) if samples else 0
    print(f"  ✅ DocBank: {len(samples)} pages, avg {avg:.1f} boxes/page")
    return samples


# ═══════════════════════════════════════════════════════════
# 3. INFERENCE WRAPPERS
# ═══════════════════════════════════════════════════════════

def run_doclay_on_image(pil_image, converter):
    """Run DocLayoutYOLO on a single PIL image. Returns list of canonical detections."""
    preds = []
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        pil_image.save(tmp.name)
        tmp_path = tmp.name

    try:
        try:
            result = converter.convert(tmp_path)
        except Exception:
            import img2pdf
            tmp_pdf = tmp_path.replace('.png', '.pdf')
            with open(tmp_pdf, 'wb') as f:
                f.write(img2pdf.convert(tmp_path))
            result = converter.convert(tmp_pdf)
            os.unlink(tmp_pdf)

        doc = result.document
        img_w, img_h = pil_image.size

        scale_x, scale_y = 1.0, 1.0
        if hasattr(doc, 'pages') and doc.pages:
            for pk, pv in doc.pages.items():
                pw = pv.size.width if hasattr(pv.size, 'width') else img_w
                ph = pv.size.height if hasattr(pv.size, 'height') else img_h
                scale_x = img_w / pw if pw > 0 else 1
                scale_y = img_h / ph if ph > 0 else 1
                break

        for item_tuple in (doc.iterate_items() if hasattr(doc, 'iterate_items') else []):
            element = item_tuple[0] if isinstance(item_tuple, tuple) else item_tuple
            label_raw = type(element).__name__.lower().replace('item', '').replace('docling', '')
            canonical = DOCLAYNET_TO_CANONICAL.get(label_raw)
            if not canonical:
                continue
            if hasattr(element, 'prov') and element.prov:
                bbox = element.prov[0].bbox
                x0 = bbox.l * scale_x
                y0_raw = bbox.t * scale_y
                x1 = bbox.r * scale_x
                y1_raw = bbox.b * scale_y
                y0, y1 = min(y0_raw, y1_raw), max(y0_raw, y1_raw)
                if (x1 - x0) * (y1 - y0) > 50:
                    preds.append({'bbox': [x0, y0, x1, y1], 'label': canonical, 'score': 0.9})
    except Exception as e:
        print(f"    DocLayout error: {e}")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    return preds


def run_nemotron_on_image(pil_image, model, processor, tokenizer, gen_config, device='cuda'):
    """Run Nemotron-Parse on a single PIL image. Returns list of canonical detections."""
    import torch

    orig_w, orig_h = pil_image.size
    padded = Image.new('RGB', (PROC_W, PROC_H), (255, 255, 255))

    if orig_w > PROC_W or orig_h > PROC_H:
        scale = min(PROC_W / orig_w, PROC_H / orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        resized = pil_image.resize((new_w, new_h), Image.LANCZOS)
        pad_x = (PROC_W - new_w) // 2
        pad_y = (PROC_H - new_h) // 2
        padded.paste(resized, (pad_x, pad_y))
        effective_scale = scale
    else:
        pad_x = (PROC_W - orig_w) // 2
        pad_y = (PROC_H - orig_h) // 2
        padded.paste(pil_image, (pad_x, pad_y))
        effective_scale = 1.0

    task_prompt = '</s><s><predict_bbox><predict_classes>'
    inputs = processor(images=padded, text=task_prompt,
                       return_tensors='pt', add_special_tokens=False).to(device)

    max_tokens = 512
    try:
        with torch.no_grad(), torch.cuda.amp.autocast():
            outputs = model.generate(**inputs, generation_config=gen_config,
                                     max_new_tokens=max_tokens)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        with torch.no_grad(), torch.cuda.amp.autocast():
            outputs = model.generate(**inputs, generation_config=gen_config,
                                     max_new_tokens=256)

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=False)
    del inputs, outputs
    torch.cuda.empty_cache()

    detections = []
    for m in BLOCK_PATTERN.finditer(decoded):
        x0_n, y0_n = float(m.group(1)), float(m.group(2))
        x1_n, y1_n = float(m.group(4)), float(m.group(5))
        cls = m.group(6).strip()

        x0_px = x0_n * PROC_W - pad_x
        y0_px = y0_n * PROC_H - pad_y
        x1_px = x1_n * PROC_W - pad_x
        y1_px = y1_n * PROC_H - pad_y

        if effective_scale != 1.0:
            x0_px /= effective_scale; y0_px /= effective_scale
            x1_px /= effective_scale; y1_px /= effective_scale

        x0_px = max(0, min(orig_w, x0_px))
        y0_px = max(0, min(orig_h, y0_px))
        x1_px = max(0, min(orig_w, x1_px))
        y1_px = max(0, min(orig_h, y1_px))

        canonical = DOCLAYNET_TO_CANONICAL.get(cls, DOCLAYNET_TO_CANONICAL.get(cls.lower()))
        if canonical and (x1_px - x0_px) > 1 and (y1_px - y0_px) > 1:
            detections.append({'bbox': [x0_px, y0_px, x1_px, y1_px],
                               'label': canonical, 'score': 0.85})
    return detections


def run_ade_on_image(pil_image, client):
    """Run LandingAI ADE DPT-2 on a single PIL image. Returns list of canonical detections."""
    preds = []
    from pathlib import Path
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        pil_image.save(tmp.name)
        tmp_path = tmp.name
    try:
        response = client.parse(document=Path(tmp_path), model="dpt-2-20260410")
        img_w, img_h = pil_image.size
        for chunk in (response.chunks if hasattr(response, 'chunks') else []):
            chunk_type = chunk.type if hasattr(chunk, 'type') else 'text'
            if chunk_type == 'marginalia':
                if hasattr(chunk, 'grounding') and hasattr(chunk.grounding, 'box'):
                    y_center = (chunk.grounding.box.top + chunk.grounding.box.bottom) / 2
                    canonical = 'page_header' if y_center < img_h * 0.15 else 'page_footer'
                else:
                    canonical = 'page_footer'
            else:
                canonical = ADE_TO_CANONICAL.get(chunk_type)
            if canonical and hasattr(chunk, 'grounding') and hasattr(chunk.grounding, 'box'):
                b = chunk.grounding.box
                preds.append({'bbox': [b.left, b.top, b.right, b.bottom],
                              'label': canonical, 'score': 0.8})
    except Exception as e:
        print(f"    ADE error: {e}")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    return preds


# ═══════════════════════════════════════════════════════════
# 4. EVALUATION FUNCTIONS
# ═══════════════════════════════════════════════════════════

def compute_iou(boxA, boxB):
    """Standard IoU for [x0, y0, x1, y1] boxes."""
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / (areaA + areaB - inter)


def match_predictions(gt_boxes, pred_boxes, iou_threshold=0.5):
    """Greedy matching: same canonical label AND IoU >= threshold."""
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


def compute_map(all_gt, all_preds, iou_thresholds, valid_classes):
    """Compute mAP@50, mAP@50:95, precision, recall, F1, mean IoU."""
    ap_per_thr = []
    ap50_per_class = {}

    for iou_thr in iou_thresholds:
        tp_cls, fp_cls, fn_cls = defaultdict(int), defaultdict(int), defaultdict(int)
        ious_at_thr = []
        for gt_boxes, pred_boxes in zip(all_gt, all_preds):
            gt_f = [b for b in gt_boxes if b['label'] in valid_classes]
            pr_f = [b for b in pred_boxes if b['label'] in valid_classes]
            matched, ugt, upr = match_predictions(gt_f, pr_f, iou_thr)
            for gi, pi, iou in matched:
                tp_cls[gt_f[gi]['label']] += 1; ious_at_thr.append(iou)
            for gi in ugt: fn_cls[gt_f[gi]['label']] += 1
            for pi in upr: fp_cls[pr_f[pi]['label']] += 1

        class_f1s = []
        for cls in valid_classes:
            tp = tp_cls[cls]; fp = fp_cls[cls]; fn = fn_cls[cls]
            p = tp / (tp + fp) if (tp + fp) else 0
            r = tp / (tp + fn) if (tp + fn) else 0
            f1 = 2 * p * r / (p + r) if (p + r) else 0
            class_f1s.append(f1)
            if iou_thr == iou_thresholds[0]:
                ap50_per_class[cls] = p
        ap_per_thr.append(np.mean(class_f1s) if class_f1s else 0)

    # Overall P/R/F1 at first threshold (0.5)
    tp_all, fp_all, fn_all = defaultdict(int), defaultdict(int), defaultdict(int)
    all_ious = []
    for gt_boxes, pred_boxes in zip(all_gt, all_preds):
        gt_f = [b for b in gt_boxes if b['label'] in valid_classes]
        pr_f = [b for b in pred_boxes if b['label'] in valid_classes]
        matched, ugt, upr = match_predictions(gt_f, pr_f, iou_thresholds[0])
        for gi, pi, iou in matched:
            tp_all[gt_f[gi]['label']] += 1; all_ious.append(iou)
        for gi in ugt: fn_all[gt_f[gi]['label']] += 1
        for pi in upr: fp_all[pr_f[pi]['label']] += 1

    s_tp = sum(tp_all.values()); s_fp = sum(fp_all.values()); s_fn = sum(fn_all.values())
    precision = s_tp / (s_tp + s_fp) if (s_tp + s_fp) else 0
    recall = s_tp / (s_tp + s_fn) if (s_tp + s_fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    return {
        'mAP50': ap_per_thr[0] if ap_per_thr else 0,
        'mAP5095': np.mean(ap_per_thr) if ap_per_thr else 0,
        'precision': precision, 'recall': recall, 'F1': f1,
        'mean_iou': np.mean(all_ious) if all_ious else 0,
        'ap50_per_class': ap50_per_class,
        'total_tp': s_tp, 'total_fp': s_fp, 'total_fn': s_fn,
    }


# ═══════════════════════════════════════════════════════════
# 5. VISUALIZATION
# ═══════════════════════════════════════════════════════════

MODEL_COLORS = {
    'GT': '#2ecc71', 'DocLayoutYOLO': '#3498db',
    'Nemotron': '#e67e22', 'ADE-DPT2': '#9b59b6',
}

def plot_summary_table(results, n_samples, output_dir):
    """Build styled 3×3 summary table and save CSV."""
    from IPython.display import display
    rows = []
    for ds in ['DocLayNet', 'PubLayNet', 'DocBank']:
        if ds not in results:
            continue
        row = {'Dataset': ds}
        for model in ['DocLayoutYOLO', 'Nemotron', 'ADE-DPT2']:
            m = results[ds].get(model, {})
            pf = model[:4]
            for k, mk in [('mAP50','mAP50'),('mAP5095','mAP5095'),
                           ('P','precision'),('R','recall'),('F1','F1'),('mIoU','mean_iou')]:
                row[f'{pf}_{k}'] = round(m.get(mk, 0), 3)
        rows.append(row)

    df = pd.DataFrame(rows).set_index('Dataset')
    models = ['DocLayoutYOLO', 'Nemotron', 'ADE-DPT2']
    metrics = ['mAP@50', 'mAP@50:95', 'P', 'R', 'F1', 'mIoU']
    col_keys = [f'{m[:4]}_{s}' for m in models for s in ['mAP50','mAP5095','P','R','F1','mIoU']]
    df_disp = df[[c for c in col_keys if c in df.columns]].copy()
    df_disp.columns = pd.MultiIndex.from_arrays(
        ([m for m in models for _ in metrics], metrics * len(models)))

    def highlight_best(s):
        best = s == s.max()
        return ['background-color:#1a5c1a;color:white;font-weight:bold' if v else '' for v in best]

    styled = df_disp.style.apply(highlight_best, axis=1).format('{:.3f}').set_caption(
        f'Layout Detection Benchmark — {n_samples} pages × 3 datasets × 3 models')
    display(styled)
    df_disp.to_csv(os.path.join(output_dir, 'benchmark_summary.csv'))
    return df_disp


def plot_perclass_heatmap(results, dataset_name, output_dir):
    """Seaborn heatmap of per-class precision@50 for one dataset."""
    classes = sorted(VALID_CLASSES_PER_DATASET[dataset_name])
    models = ['DocLayoutYOLO', 'Nemotron', 'ADE-DPT2']
    data = [[results[dataset_name].get(m, {}).get('ap50_per_class', {}).get(c, 0)
             for m in models] for c in classes]
    hm_df = pd.DataFrame(data, index=classes, columns=models)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(hm_df, annot=True, fmt='.3f', cmap='RdYlGn',
                linewidths=.5, vmin=0, vmax=1, ax=ax,
                cbar_kws={'label': 'Precision @ IoU 0.50'})
    ax.set_title(f'{dataset_name} — Per-Class Precision @ IoU 0.50', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, f'{dataset_name.lower()}_perclass_ap.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"📊 Saved: {path}")


def plot_visual_comparison(samples, preds_by_model, dataset_name, page_idx, output_dir):
    """4-column visual comparison: GT + 3 models."""
    sample = samples[page_idx]
    model_keys = ['DocLayoutYOLO', 'Nemotron', 'ADE-DPT2']
    fig, axes = plt.subplots(1, 4, figsize=(24, 8))
    fig.suptitle(f'{dataset_name} — Page {page_idx}', fontsize=14, fontweight='bold', y=1.02)

    def _draw(ax, img, boxes, color, title):
        ax.imshow(img); ax.set_title(title, fontsize=9, fontweight='bold', color=color); ax.axis('off')
        for b in boxes:
            x0, y0, x1, y1 = b['bbox']
            ax.add_patch(patches.Rectangle((x0, y0), x1-x0, y1-y0,
                         linewidth=1.5, edgecolor=color, facecolor='none', alpha=.8))
            ax.text(x0+2, y0-3, b['label'], fontsize=5, color='white', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.15', facecolor=color, alpha=.75, edgecolor='none'))

    _draw(axes[0], sample['image'], sample['gt'], MODEL_COLORS['GT'],
          f'Ground Truth ({len(sample["gt"])})')
    for i, mk in enumerate(model_keys):
        boxes = preds_by_model.get(mk, {}).get(page_idx, [])
        _draw(axes[i+1], sample['image'], boxes, MODEL_COLORS[mk], f'{mk} ({len(boxes)})')

    plt.tight_layout()
    path = os.path.join(output_dir, f'visual_sample_{dataset_name}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"🖼️  Saved: {path}")
