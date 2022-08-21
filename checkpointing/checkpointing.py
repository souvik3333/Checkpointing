import torch
import torch.nn as nn
import os
import logging
from typing import Optional, Tuple

def create_checkpoint(
    model_arch: nn.Module, 
    ckpt_path: str, 
    input_img_size: Optional[Tuple]=None, 
    strict: bool =True, 
    state_dict_key: Optional[str] = None, 
    file_name_prefix: Optional[str] = "checkpointing_"):
    
    ckpt = torch.load(ckpt_path)
    state_dict = ckpt[state_dict_key] if state_dict_key else ckpt
    mismatch = model_arch.load_state_dict(state_dict, strict=strict)
    missing_keys, _ = mismatch

    model_arch.eval()

    if len(missing_keys) >0 and not strict:
        logging.warning(f"Model loading is not strict. Found missing keys in  {missing_keys}")

    with torch.no_grad():
        input_tensor = torch.rand(input_img_size)
        traced_model = torch.jit.trace(model_arch, input_tensor)

    
    save_dirname = os.path.dirname(ckpt_path)
    save_path = os.path.join(save_dirname, file_name_prefix+os.path.basename(ckpt_path))
    
    torch.jit.save(traced_model, save_path)
    logging.info(f"Checkpoint saved at {save_path}")



def load_checkpoint(ckpt_path: str):
    loaded_ckpt = torch.jit.load(ckpt_path)

    return loaded_ckpt
