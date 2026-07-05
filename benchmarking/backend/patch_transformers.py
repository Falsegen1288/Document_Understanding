import torch
import safetensors

# Monkeypatch to bypass PyTorch float8_e8m0fnu missing attribute in Windows and safetensors metadata NoneType error
try:
    if not hasattr(torch, "float8_e8m0fnu"):
        # Mock float8_e8m0fnu so transformers doesn't crash on import
        torch.float8_e8m0fnu = getattr(torch, "float8_e4m3fn", torch.uint8)
except Exception:
    pass

try:
    if hasattr(safetensors, "safe_open"):
        original_safe_open = safetensors.safe_open
        
        class PatchedSafeOpen:
            def __init__(self, *args, **kwargs):
                self.file = original_safe_open(*args, **kwargs)
            
            def metadata(self):
                meta = self.file.metadata()
                return meta if meta is not None else {}
                
            def __getattr__(self, name):
                return getattr(self.file, name)
                
            def __enter__(self):
                return self
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                if hasattr(self.file, "__exit__"):
                    return self.file.__exit__(exc_type, exc_val, exc_tb)
                    
        safetensors.safe_open = PatchedSafeOpen
except Exception:
    pass

print("Monkeypatch applied: torch.float8_e8m0fnu and safetensors.safe_open.metadata successfully secured.")
