# VLM Captioning & Attribute Extraction Leaderboard

Dataset: 6 medical instrument catalogue images (`MedCore_GT_v2.json` ground truth)

## 1. Image Captioning Quality Leaderboard

| Model                                     | Type      |    BLEU-4 |   ROUGE-L |       CIDEr |   BERTScore | Avg Latency (s)   |
|:------------------------------------------|:----------|----------:|----------:|------------:|------------:|:------------------|
| meta-llama/llama-4-scout-17b-16e-instruct | Groq API  | 0.0252296 | 0.275748  | 0.000124806 |    0.832876 | 1.97s             |
| qwen2.5vl:3b                              | Local VLM | 0.0685384 | 0.264954  | 0.00396258  |    0.817106 | 41.90s            |
| qwen/qwen3.6-27b                          | Groq API  | 0.0132963 | 0.0758992 | 0           |    0.719656 | 5.20s             |
| moondream:latest                          | Local VLM | 0.0194716 | 0.18947   | 0.149949    |    0.554712 | 14.88s            |

## 2. Attribute Extraction Accuracy Leaderboard

| Model                                     | Type      | JSON Validity   |   Attr Precision |   Attr Recall |   Attr F1 | Avg Latency (s)   |
|:------------------------------------------|:----------|:----------------|-----------------:|--------------:|----------:|:------------------|
| meta-llama/llama-4-scout-17b-16e-instruct | Groq API  | 100.0%          |        0.328704  |     0.361111  | 0.337963  | 1.97s             |
| qwen/qwen3.6-27b                          | Groq API  | 83.3%           |        0.291667  |     0.319444  | 0.301852  | 5.20s             |
| qwen2.5vl:3b                              | Local VLM | 100.0%          |        0.305556  |     0.25463   | 0.272222  | 41.90s            |
| moondream:latest                          | Local VLM | 50.0%           |        0.0277778 |     0.0277778 | 0.0277778 | 14.88s            |

## 3. Resource & Cost Footprint Leaderboard

| Model                                     | Type      | Avg Latency (s)   | VRAM Footprint (Peak/Delta)   | Estimated Cost (6 images)   |
|:------------------------------------------|:----------|:------------------|:------------------------------|:----------------------------|
| meta-llama/llama-4-scout-17b-16e-instruct | Groq API  | 1.97s             | N/A (Cloud)                   | $0.00217                    |
| moondream:latest                          | Local VLM | 14.88s            | 2143 MB / +2136 MB            | $0.0000 (Free/Local)        |
| qwen2.5vl:3b                              | Local VLM | 41.90s            | 3921 MB / +1778 MB            | $0.0000 (Free/Local)        |
| qwen/qwen3.6-27b                          | Groq API  | 5.20s             | N/A (Cloud)                   | $0.04109                    |

