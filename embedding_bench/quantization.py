from typing import Literal
import torch

def get_quantization_config(bits: Literal[4, 8]):
    from transformers import BitsAndBytesConfig
    if bits == 4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
    elif bits == 8:
        return BitsAndBytesConfig(
            load_in_8bit=True
        )
    else:
        raise ValueError(f"Unsupported quantization bits: {bits}")
