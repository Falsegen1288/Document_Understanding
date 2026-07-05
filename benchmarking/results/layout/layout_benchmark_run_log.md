# Layout Detection Benchmark Re-run Log

### [STEP 0a] label_map.py — PASS — attempt 1
Verified raw label sets from docling (DocLayoutYOLO) and Nemotron-Parse caches. Created deterministic CLASS_MAP in label_map_config.json mapping all model output categories to the canonical GT categories. Verified that all raw labels map correctly.

### [STEP 0b] gt_audit.py — PASS — attempt 1
Completed the ground truth padding audit across all 80 elements in MedCore_GT_v2.json. Generated results/gt_padding_audit.csv. Confirmed that padding is highly class-conditional (e.g. price_tag bottom-padding is ~44%, whereas product_card and caption are close to 0%).

### [STEP 0c] gt_tighten.py — PASS — attempt 1
Ran pixel-content auto-tightening on text-bearing classes from MedCore_GT_v2.json. Tightened 64 out of 80 elements successfully. Saved output to layout_GT_custom_tightened.json and generated visual overlays under results/verification/page_*_verification.png.

### [STEP 1] run_traditional_benchmark.py — PASS — attempt 1
Executed the traditional benchmark across 2 models and 3 datasets (DocLayNet, PubLayNet, DocBank) using pycocotools. Saved the results to results/traditional_results.csv. Confirmed DocBank collapse as expected, and Nemotron/YOLO performance trends align with the historical baseline.

### [STEP 2] run_custom_benchmark.py — PASS — attempt 1
Executed the custom layout benchmark on MedCore_Catalogue_v2.pdf across 2 models (DocLayoutYOLO, Nemotron) and 2 GT variants (GT-raw, GT-tight). Evaluated all 4 layers (Geometric, COTe, LED structural errors, Reading order) and generated results/custom_results_layer{1-4}.csv. Verified that Class Accuracy is no longer structurally zero (DocLayoutYOLO = 50.0%, Nemotron = 25.0%).

### [STEP 3] build_scorecard.py — PASS — attempt 1
Compiled the consolidated scorecard comparing GT-raw vs GT-tight results side-by-side with padding-tolerant metrics and footnotes. Saved results to results/consolidated_scorecard.csv. Confirmed that tightening the GT significantly increases DocLayoutYOLO F1 (from 0.0359 to 0.0551, a 53% relative improvement) while slightly decreasing Nemotron F1 (from 0.0270 to 0.0192) due to Nemotron's looser localization.
