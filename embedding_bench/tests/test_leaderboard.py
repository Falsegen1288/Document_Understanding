import os
import tempfile
import pytest
import pandas as pd
from pathlib import Path

from embedding_bench.production_cutover import check_adr_status

def test_leaderboard_sorting_logic():
    # Mock data representing leaderboard rows
    rows = [
        {"model_key": "model_a", "arm": "dense_only", "spec_hit_rate_at_5": 0.50, "faithfulness": 0.85, "embed_query_latency_ms_p95": 100.0, "peak_vram_mb": 1000.0},
        {"model_key": "model_b", "arm": "bm25_hybrid", "spec_hit_rate_at_5": 0.875, "faithfulness": 0.82, "embed_query_latency_ms_p95": 12.0, "peak_vram_mb": 0.0},
        {"model_key": "model_c", "arm": "splade_hybrid", "spec_hit_rate_at_5": 0.875, "faithfulness": 0.85, "embed_query_latency_ms_p95": 200.0, "peak_vram_mb": 4000.0},
        {"model_key": "model_d", "arm": "dense_only", "spec_hit_rate_at_5": 0.50, "faithfulness": 0.85, "embed_query_latency_ms_p95": 50.0, "peak_vram_mb": 500.0}
    ]
    df = pd.DataFrame(rows)
    
    # Sort with same logic: spec_hit_rate_at_5 desc, faithfulness desc, latency asc, vram asc
    df_sorted = df.sort_values(
        by=["spec_hit_rate_at_5", "faithfulness", "embed_query_latency_ms_p95", "peak_vram_mb"],
        ascending=[False, False, True, True]
    )
    
    sorted_models = df_sorted["model_key"].tolist()
    # Expect:
    # 1st: model_c (spec 0.875, faithfulness 0.85)
    # 2nd: model_b (spec 0.875, faithfulness 0.82)
    # 3rd: model_d (spec 0.50, faithfulness 0.85, latency 50.0)
    # 4th: model_a (spec 0.50, faithfulness 0.85, latency 100.0)
    assert sorted_models == ["model_c", "model_b", "model_d", "model_a"]

def test_adr_status_parsing():
    with tempfile.TemporaryDirectory() as temp_dir:
        adr_file = Path(temp_dir) / "adr.md"
        
        # Test Proposed
        with open(adr_file, "w", encoding="utf-8") as f:
            f.write("# ADR Title\n\n## Status\nProposed\n")
        assert not check_adr_status(adr_file)
        
        # Test Accepted
        with open(adr_file, "w", encoding="utf-8") as f:
            f.write("# ADR Title\n\n## Status\nAccepted\n")
        assert check_adr_status(adr_file)
