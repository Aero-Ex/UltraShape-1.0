import os
import sys
import torch
from omegaconf import OmegaConf

# Add project root to path
project_root = "/home/aero/UltraShape-1.0"
if project_root not in sys.path:
    sys.path.append(project_root)

# Import the modified load_models from the script
# We'll just copy the relevant parts to keep it independent
from ultrashape.utils.misc import instantiate_from_config
from ultrashape.utils.gguf_loader import patch_model

def test_gguf_init():
    config_path = "/home/aero/UltraShape-1.0/configs/infer_dit_refine.yaml"
    if not os.path.exists(config_path):
        print(f"Config not found at {config_path}")
        return

    config = OmegaConf.load(config_path)
    
    print("Testing DiT instantiation on meta device...")
    with torch.device("meta"):
        dit = instantiate_from_config(config.model.params.dit_cfg)
    
    print("Patching DiT...")
    patch_model(dit)
    
    print("Verification successful: DiT shell initialized and patched on meta device.")
    # Check for a few GGML layers
    has_ggml = False
    for m in dit.modules():
        if "GGMLLinear" in str(type(m)):
            has_ggml = True
            break
    
    if has_ggml:
        print("Success: GGMLLinear layers found in patched model.")
    else:
        print("Warning: No GGMLLinear layers found. Check patching logic.")

if __name__ == "__main__":
    test_gguf_init()
