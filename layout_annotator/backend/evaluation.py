import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from collections import Counter

DOCLING_COLORS = {
    'title': '#e63946',
    'section_header': '#ff6b35',
    'text': '#457b9d',
    'list_item': '#2a9d8f',
    'table': '#e9c46a',
    'picture': '#f4a261',
    'caption': '#8ecae6',
    'footnote': '#a8dadc',
    'formula': '#6d6875',
    'page_header': '#b5838d',
    'page_footer': '#e5989b',
}

NEMOTRON_COLORS = {
    'Title': '#e63946',
    'Text': '#457b9d',
    'Table': '#e9c46a',
    'Picture': '#f4a261',
    'Caption': '#8ecae6',
    'Footnote': '#a8dadc',
    'Formula': '#6d6875',
    'List-item': '#2a9d8f',
    'Section-header': '#ff6b35',
    'Page-header': '#b5838d',
    'Page-footer': '#e5989b',
}

DEFAULT_COLOR = '#888888'


def draw_bboxes(pil_img, detections, color_map, title):
    img = pil_img.copy().convert('RGBA')
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for det in detections:
        cls = det['type']
        bbox = det['bbox']
        if len(bbox) < 4:
            continue
        hex_color = color_map.get(cls, DEFAULT_COLOR)
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        x0, y0, x1, y1 = bbox
        draw.rectangle([x0, y0, x1, y1], fill=(r, g, b, 60))
        draw.rectangle([x0, y0, x1, y1], outline=(r, g, b, 230), width=2)
        label = cls[:15]
        lw = len(label) * 6 + 6
        lh = 14
        ty = max(y0 - lh - 1, 0)
        draw.rectangle([x0, ty, x0 + lw, ty + lh], fill=(r, g, b, 210))
        draw.text((x0 + 3, ty + 1), label, fill=(255, 255, 255, 255))

    merged = Image.alpha_composite(img, overlay).convert('RGB')
    draw2 = ImageDraw.Draw(merged)
    draw2.rectangle([0, 0, merged.width, 28], fill=(20, 20, 20))
    draw2.text((8, 6), title, fill=(255, 255, 255))
    return merged


def build_legend(color_map, title, width=200):
    items = list(color_map.items())
    height = len(items) * 20 + 35
    leg = Image.new('RGB', (width, height), (250, 250, 250))
    draw = ImageDraw.Draw(leg)
    draw.rectangle([0, 0, width - 1, height - 1], outline=(180, 180, 180))
    draw.text((8, 5), title, fill=(30, 30, 30))
    for i, (cls, hex_col) in enumerate(items):
        y = 28 + i * 20
        r = int(hex_col[1:3], 16)
        g = int(hex_col[3:5], 16)
        b_ = int(hex_col[5:7], 16)
        draw.rectangle([8, y, 22, y + 14], fill=(r, g, b_))
        draw.text((28, y + 1), cls[:18], fill=(30, 30, 30))
    return leg


def generate_side_by_side_visualizations(page_images, dl_results, nm_results, pages, output_dir):
    """
    Generate side-by-side comparison images for all pages that have nemotron results.
    """
    for pg in pages:
        orig_img = page_images.get(pg)
        if not orig_img:
            continue
            
        dl_dets = dl_results.get(pg, [])
        nm_dets = nm_results.get(pg, [])
        
        # We only generate visualization if there are detections to compare
        if not dl_dets and not nm_dets:
            continue

        dl_vis = draw_bboxes(
            orig_img, dl_dets, DOCLING_COLORS,
            f'DocLayoutYOLO  |  Page {pg}  |  {len(dl_dets)} elements'
        )
        nm_vis = draw_bboxes(
            orig_img, nm_dets, NEMOTRON_COLORS,
            f'Nemotron-Parse (NVIDIA)  |  Page {pg}  |  {len(nm_dets)} elements'
        )

        dl_leg = build_legend(DOCLING_COLORS, 'DocLayoutYOLO classes')
        nm_leg = build_legend(NEMOTRON_COLORS, 'Nemotron classes')

        total_w = dl_vis.width + nm_vis.width + 10
        total_h = max(dl_vis.height, nm_vis.height)
        canvas = Image.new('RGB', (total_w, total_h), (240, 240, 240))
        canvas.paste(dl_vis, (0, 0))
        canvas.paste(nm_vis, (dl_vis.width + 10, 0))

        leg_h = max(dl_leg.height, nm_leg.height)
        final = Image.new('RGB', (total_w, total_h + leg_h + 10), (240, 240, 240))
        final.paste(canvas, (0, 0))
        final.paste(dl_leg, (0, total_h + 5))
        final.paste(nm_leg, (dl_vis.width + 10, total_h + 5))

        fig, ax = plt.subplots(1, 1, figsize=(20, 12))
        ax.imshow(final)
        ax.axis('off')
        ax.set_title(
            f'Page {pg} — DocLayoutYOLO (left) vs NVIDIA Nemotron-Parse (right)',
            fontsize=13, fontweight='bold', pad=10
        )
        plt.tight_layout()

        save_path = os.path.join(output_dir, f"page_{pg:02d}_comparison.png")
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close(fig)


def generate_comparison_chart(dl_results, nm_results, pages, output_dir):
    """
    Generates a chart comparing the total counts by class and by page.
    """
    class_map = {
        'Title': 'title',
        'Text': 'text',
        'Table': 'table',
        'Picture': 'picture',
        'Caption': 'caption',
        'Footnote': 'footnote',
        'Formula': 'formula',
        'List-item': 'list_item',
        'Section-header': 'section_header',
        'Page-header': 'page_header',
        'Page-footer': 'page_footer',
    }

    rows = []
    summary_rows = []
    
    for pg in pages:
        dl = dl_results.get(pg, [])
        nm = nm_results.get(pg, [])
        dl_types = Counter(e['type'] for e in dl)
        nm_types = Counter(e['type'] for e in nm)
        nm_mapped = Counter({class_map.get(k, k.lower()): v for k, v in nm_types.items()})
        
        all_classes = set(dl_types.keys()) | set(nm_mapped.keys())
        for cls in all_classes:
            rows.append({
                'page': pg,
                'class': cls,
                'DocLayoutYOLO': dl_types.get(cls, 0),
                'Nemotron': nm_mapped.get(cls, 0),
            })
            
        summary_rows.append({
            'Page': pg,
            'DocLayoutYOLO': len(dl),
            'Nemotron': len(nm),
        })

    if not rows:
        return

    df = pd.DataFrame(rows)
    df_agg = df.groupby('class')[['DocLayoutYOLO', 'Nemotron']].sum().reset_index()
    df_agg = df_agg.sort_values('DocLayoutYOLO', ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle('DocLayoutYOLO vs NVIDIA Nemotron-Parse-v1.1', fontsize=14, fontweight='bold')

    ax = axes[0]
    x = np.arange(len(df_agg))
    width = 0.35
    b1 = ax.bar(x - width / 2, df_agg['DocLayoutYOLO'], width, label='DocLayoutYOLO', color='#457b9d', alpha=0.85)
    b2 = ax.bar(x + width / 2, df_agg['Nemotron'], width, label='Nemotron (NVIDIA)', color='#e63946', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(df_agg['class'], rotation=35, ha='right', fontsize=9)
    ax.set_ylabel('Number of elements detected')
    ax.set_title('Elements by Class Type')
    ax.legend()
    
    for bar in b1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                str(int(bar.get_height())), ha='center', va='bottom', fontsize=8)
    for bar in b2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                str(int(bar.get_height())), ha='center', va='bottom', fontsize=8)

    ax2 = axes[1]
    pg_nums = [r['Page'] for r in summary_rows]
    dl_vals = [r['DocLayoutYOLO'] for r in summary_rows]
    nm_vals = [r['Nemotron'] for r in summary_rows]
    x2 = np.arange(len(pg_nums))
    ax2.bar(x2 - width / 2, dl_vals, width, label='DocLayoutYOLO', color='#457b9d', alpha=0.85)
    ax2.bar(x2 + width / 2, nm_vals, width, label='Nemotron (NVIDIA)', color='#e63946', alpha=0.85)
    ax2.set_xticks(x2)
    ax2.set_xticklabels([f'Page {p}' for p in pg_nums])
    ax2.set_ylabel('Total elements detected')
    ax2.set_title('Total Detections per Page')
    ax2.legend()

    plt.tight_layout()
    chart_path = os.path.join(output_dir, 'comparison_chart.png')
    plt.savefig(chart_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def bbox_to_mask(bbox, page_w, page_h, scale=1.0):
    h = int(page_h * scale)
    w = int(page_w * scale)
    mask = np.zeros((h, w), dtype=bool)
    x0 = max(0, int(bbox[0] * scale))
    y0 = max(0, int(bbox[1] * scale))
    x1 = min(w, int(bbox[2] * scale))
    y1 = min(h, int(bbox[3] * scale))
    mask[y0:y1, x0:x1] = True
    return mask


def compute_cote_scores(ref_dets, pred_dets, page_w, page_h, scale=0.2):
    if not ref_dets or not pred_dets:
        return None

    H = int(page_h * scale)
    W = int(page_w * scale)
    page_area = H * W

    ssu_masks = [bbox_to_mask(d['bbox'], page_w, page_h, scale) for d in ref_dets]
    pred_masks = [bbox_to_mask(d['bbox'], page_w, page_h, scale) for d in pred_dets]

    gt_union = np.zeros((H, W), dtype=bool)
    for m in ssu_masks:
        gt_union |= m

    assignments = {}
    for j, pm in enumerate(pred_masks):
        best_ssu = -1
        best_area = 0
        for i, sm in enumerate(ssu_masks):
            overlap = int(np.sum(pm & sm))
            if overlap > best_area:
                best_area = overlap
                best_ssu = i
        if best_ssu >= 0:
            assignments[j] = best_ssu

    coverage_scores = []
    for i, sm in enumerate(ssu_masks):
        ssu_area = int(np.sum(sm))
        if ssu_area == 0:
            continue
        assigned_preds = [j for j, s in assignments.items() if s == i]
        if not assigned_preds:
            coverage_scores.append(0.0)
            continue
        combined = np.zeros((H, W), dtype=bool)
        for j in assigned_preds:
            combined |= pred_masks[j]
        covered = int(np.sum(sm & combined))
        coverage_scores.append(covered / ssu_area)

    overlap_scores = []
    for i, sm in enumerate(ssu_masks):
        ssu_area = int(np.sum(sm))
        if ssu_area == 0:
            continue
        assigned_preds = [j for j, s in assignments.items() if s == i]
        if len(assigned_preds) < 2:
            overlap_scores.append(0.0)
            continue
        combined_sum = np.zeros((H, W), dtype=np.int16)
        for j in assigned_preds:
            combined_sum += pred_masks[j].astype(np.int16)
        redundant = int(np.sum(sm & (combined_sum > 1).astype(bool)))
        overlap_scores.append(redundant / ssu_area)

    trespass_scores = []
    for j, pm in enumerate(pred_masks):
        if j not in assignments:
            continue
        my_ssu = assignments[j]
        ssu_area = int(np.sum(ssu_masks[my_ssu]))
        if ssu_area == 0:
            continue
        other_mask = np.zeros((H, W), dtype=bool)
        for i, sm in enumerate(ssu_masks):
            if i != my_ssu:
                other_mask |= sm
        trespass = int(np.sum(other_mask & pm))
        trespass_scores.append(trespass / ssu_area)

    pred_union = np.zeros((H, W), dtype=bool)
    for pm in pred_masks:
        pred_union |= pm
    non_gt_area = page_area - int(np.sum(gt_union))
    if non_gt_area > 0:
        unassigned_pred = pred_union & ~gt_union
        excess = int(np.sum(unassigned_pred)) / non_gt_area
    else:
        excess = 0.0

    return {
        'Coverage': float(np.mean(coverage_scores)) if coverage_scores else 0.0,
        'Overlap': float(np.mean(overlap_scores)) if overlap_scores else 0.0,
        'Trespass': float(np.mean(trespass_scores)) if trespass_scores else 0.0,
        'Excess': float(excess),
        'n_ref': len(ref_dets),
        'n_pred': len(pred_dets),
        'n_assigned': len(assignments),
    }


def generate_pycote_report(dl_results, nm_results, page_images, pages, output_dir):
    """
    Computes pyCOTe metrics for the given pages and generates a text report.
    """
    report_lines = []
    report_lines.append('='*65)
    report_lines.append('  pyCOTe-STYLE EVALUATION')
    report_lines.append('  Cross-model spatial agreement (no ground truth required)')
    report_lines.append('='*65)

    cote_results = {}
    
    for pg in pages:
        img = page_images.get(pg)
        if not img:
            continue
        pw, ph = img.size
        dl_dets = dl_results.get(pg, [])
        nm_dets = nm_results.get(pg, [])

        if not dl_dets or not nm_dets:
            continue

        scores_a = compute_cote_scores(dl_dets, nm_dets, pw, ph)
        scores_b = compute_cote_scores(nm_dets, dl_dets, pw, ph)
        
        if scores_a is None or scores_b is None:
            continue

        cote_results[pg] = {'DL->NM': scores_a, 'NM->DL': scores_b}

        report_lines.append(f'\n  -- Page {pg} ({pw}x{ph}px) --')
        report_lines.append(f'  {"Metric":12s}  {"DL as ref (NM pred)":22s}  {"NM as ref (DL pred)":22s}')
        report_lines.append(f'  {"-"*12}  {"-"*22}  {"-"*22}')
        for metric in ['Coverage', 'Overlap', 'Trespass', 'Excess']:
            va = scores_a[metric]
            vb = scores_b[metric]
            report_lines.append(f'  {metric:12s}  {va:22.4f}  {vb:22.4f}')

    if cote_results:
        report_lines.append('\n' + '='*65)
        report_lines.append('  AGGREGATE COTe SCORES (mean across all compared pages)')
        report_lines.append('='*65)
        for direction in ['DL->NM', 'NM->DL']:
            vals = {m: [] for m in ['Coverage', 'Overlap', 'Trespass', 'Excess']}
            for pg, res in cote_results.items():
                for m in vals:
                    vals[m].append(res[direction][m])
            label = 'DocLayoutYOLO->Nemotron' if direction == 'DL->NM' else 'Nemotron->DocLayoutYOLO'
            report_lines.append(f'\n  {label}:')
            for m, v in vals.items():
                mean_val = float(np.mean(v))
                indicator = ''
                if m == 'Coverage' and mean_val > 0.7:
                    indicator = '✅ good'
                elif m == 'Trespass' and mean_val > 0.2:
                    indicator = '⚠️ high — boxes bleeding into neighbours'
                elif m == 'Overlap' and mean_val > 0.2:
                    indicator = '⚠️ high — duplicate coverage'
                report_lines.append(f'    {m:12s}: {mean_val:.4f}  {indicator}')

    report_path = os.path.join(output_dir, 'evaluation_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    return report_path
