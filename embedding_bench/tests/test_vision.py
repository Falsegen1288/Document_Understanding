import os
import tempfile
import json
import pytest
from pathlib import Path
import numpy as np

# Force mock mode
os.environ["EMBEDDING_BENCH_TEST_MODE"] = "1"

from embedding_bench.backends.factory import EmbeddingBackendFactory
from embedding_bench.backends.vision_backend import VisionEmbeddingBackend
from embedding_bench.data.figure_chunk_linker import build_linked_chunk_set, LinkedChunk

def test_vision_backend_mock():
    granite = EmbeddingBackendFactory.create("granite-vision-embedding")
    assert isinstance(granite, VisionEmbeddingBackend)
    assert granite.dim == 128
    
    # Test embed_documents
    chunks = [
        {"chunk_id": "c1", "text": "descriptor 1", "figure_image_path": "fake.png"},
        {"chunk_id": "c2", "text": "descriptor 2", "figure_image_path": None}
    ]
    res = granite.embed_documents(chunks)
    assert res.dense.shape == (2, 128)
    
    # Test embed_query
    q_res = granite.embed_query("query text")
    assert q_res.dense.shape == (1, 128)

def test_chunk_linking_mock():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        
        # Write dummy chunks file
        chunks = [
            {
                "chunk_id": "Medical_004_demo_30p_1",
                "text": "surgical blade table",
                "page": 1,
                "element_types": ["table"],
                "bbox_union": [10.0, 10.0, 50.0, 50.0],
                "source_element_indices": [0]
            }
        ]
        chunks_file = temp_dir_path / "Medical_004_demo_30p_hybrid_semantic.json"
        chunks_file.parent.mkdir(parents=True, exist_ok=True)
        with open(chunks_file, "w") as f:
            json.dump(chunks, f)
            
        # Create a fake image file to pass path checks
        fake_img_path = temp_dir_path / "page_1_figure_10_60.png"
        fake_img_path.touch()
        
        layout = {
            "elements": [
                {
                    "type": "figure",
                    "bbox": [10.0, 60.0, 50.0, 100.0],
                    "page": 1,
                    "image_path": str(fake_img_path)
                }
            ]
        }
        layout_file = temp_dir_path / "Medical_004_demo_30p" / "Medical_004_demo_30p.json"
        layout_file.parent.mkdir(parents=True, exist_ok=True)
        with open(layout_file, "w") as f:
            json.dump(layout, f)
            
        # Write empty chunks and layout files for the other 2 stems
        for stem in ["Researchpaper_KAI", "Scientific_001"]:
            c_f = temp_dir_path / f"{stem}_hybrid_semantic.json"
            with open(c_f, "w") as f:
                json.dump([], f)
            l_f = temp_dir_path / stem / f"{stem}.json"
            l_f.parent.mkdir(parents=True, exist_ok=True)
            with open(l_f, "w") as f:
                json.dump({"elements": []}, f)
                
        output_file = temp_dir_path / "linked_chunks.jsonl"
        linked = build_linked_chunk_set(temp_dir_path, output_file)
        
        assert len(linked) == 1
        assert linked[0].figure_image_path == str(fake_img_path).replace("\\", "/")
        assert linked[0].link_confidence == 1.0
