# Re-run Results Summary

This folder contains the complete output files from the bug-fixed corrective layout detection re-run:
- `traditional_results.csv`: Evaluation metrics on DocLayNet, PubLayNet, and DocBank.
- `custom_results_layer1.csv`: Layer 1 Geometric matching metrics (P/R/F1, mean IoU, and Class Acc).
- `custom_results_layer2.csv`: Layer 2 COTe area mask metrics.
- `custom_results_layer3.csv`: Layer 3 LED structural error counts.
- `custom_results_layer4.csv`: Layer 4 Reading order correlation.
- `consolidated_scorecard.csv`: Consolidated raw vs tight GT scorecard comparison.
- `gt_padding_audit.csv`: Per-class padding margins computed from pixel audit.

For details of the fixes and ranking analyses, refer to the [walkthrough.md](../../walkthrough.md) artifact.
