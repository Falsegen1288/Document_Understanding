import os
import re
import json
import numpy as np
import pandas as pd
import yaml
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from pycocoevalcap.cider.cider import Cider
from bert_score import score

def tokenize(text):
    return re.findall(r'\w+', text.lower())

def main():
    config_path = "D:/antigravity/benchmarking/vlm_benchmark/config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    gt_file_path = config["paths"]["gt_file"]
    with open(gt_file_path, "r") as f:
        gt_data = json.load(f)
        
    images = gt_data["images"]
    predictions_dir = config["paths"]["predictions_dir"]
    results_dir = config["paths"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)
    
    # Ground truths mapping
    gt_captions_map = {img["image_id"]: img["reference_captions"] for img in images}
    
    # Identify prediction files
    target_files = [
        "groq_meta_llama_llama_4_scout_17b_16e_instruct_predictions.jsonl",
        "groq_qwen_qwen3_6_27b_predictions.jsonl",
        "local_moondream_latest_predictions.jsonl",
        "local_qwen2_5vl_3b_predictions.jsonl"
    ]
    pred_files = [f for f in os.listdir(predictions_dir) if f in target_files]
    
    print(f"Found prediction files to score: {pred_files}")
    
    results = []
    
    # Initialize ROUGE scorer
    r_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    for pf in pred_files:
        p_path = os.path.join(predictions_dir, pf)
        print(f"\nScoring: {pf}...")
        
        # Load predictions
        preds = []
        with open(p_path, "r") as f:
            for line in f:
                if line.strip():
                    preds.append(json.loads(line))
                    
        if not preds:
            print(f" WARNING: No predictions found in {pf}. Skipping.")
            continue
            
        cands = []
        refs = []
        
        cands_tokens = []
        refs_tokens = []
        
        gts_cider = {}
        res_cider = {}
        
        # We match predictions to ground truth by image_id
        for idx, p in enumerate(preds):
            img_id = p["image_id"]
            if img_id not in gt_captions_map:
                continue
                
            cand_caption = p["caption_prediction"]
            ref_captions = gt_captions_map[img_id]
            
            cands.append(cand_caption)
            refs.append(ref_captions)
            
            cands_tokens.append(tokenize(cand_caption))
            refs_tokens.append([tokenize(ref) for ref in ref_captions])
            
            gts_cider[idx] = ref_captions
            res_cider[idx] = [cand_caption]
            
        if not cands:
            print(" WARNING: No matched predictions found. Skipping.")
            continue
            
        # 1. Compute BLEU-4 (corpus-level)
        smooth = SmoothingFunction().method1
        bleu4 = corpus_bleu(refs_tokens, cands_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)
        
        # 2. Compute ROUGE-L (sentence-level max, then mean)
        rouge_scores = []
        for cand, img_refs in zip(cands, refs):
            score_vals = [r_scorer.score(ref, cand)['rougeL'].fmeasure for ref in img_refs]
            rouge_scores.append(max(score_vals))
        rougel = np.mean(rouge_scores)
        
        # 3. Compute CIDEr
        cider_scorer = Cider()
        cider_val, _ = cider_scorer.compute_score(gts_cider, res_cider)
        
        # 4. Compute BERTScore (using distilbert-base-uncased)
        print("  Running BERTScore...")
        P, R, F1 = score(cands, refs, model_type="distilbert-base-uncased", lang="en", verbose=False)
        bertscore_f1 = F1.mean().item()
        
        # Parse model details
        model_name = preds[0]["model"]
        is_local = "local" in pf
        model_type = "local" if is_local else "groq"
        
        results.append({
            "model": model_name,
            "type": model_type,
            "file": pf,
            "BLEU-4": bleu4,
            "ROUGE-L": rougel,
            "CIDEr": cider_val,
            "BERTScore": bertscore_f1
        })
        
        print(f"  Results for {model_name}:")
        print(f"    BLEU-4:    {bleu4:.4f}")
        print(f"    ROUGE-L:   {rougel:.4f}")
        print(f"    CIDEr:     {cider_val:.4f}")
        print(f"    BERTScore: {bertscore_f1:.4f}")
        
    # Save to CSV
    df = pd.DataFrame(results)
    csv_path = os.path.join(results_dir, "captioning_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved captioning metrics to: {csv_path}")

if __name__ == "__main__":
    main()
