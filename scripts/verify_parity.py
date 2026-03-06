import os
import sys
import torch
import torch.nn as nn
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultrashape.utils.gguf_loader import patch_model, load_gguf
from ultrashape.utils.misc import instantiate_from_config

def compare_parity():
    pt_path = "/home/aero/ComfyUI-GGUF/ultrashape_v1.pt"
    gguf_path = "/home/aero/ComfyUI-GGUF/ultrashape_v1-Q8.gguf"
    config_path = "configs/infer_dit_refine.yaml"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 # Standard matching dtype for BFloat16/Float16 checks
    
    print("=== Model Parity Check (.pt vs Q8 GGUF) ===")
    
    config = OmegaConf.load(config_path)
    
    # --- 1. Load Original PT Model ---
    print("\n[1/3] Instantiating and loading Original PT Model...")
    model_pt = instantiate_from_config(config.model.params.dit_cfg)
    
    print(f"Loading weights from {pt_path}...")
    sd_full = torch.load(pt_path, map_location='cpu', weights_only=True)
    sd_dit = sd_full['dit'] if 'dit' in sd_full else sd_full
    model_pt.load_state_dict(sd_dit, strict=True)
    model_pt.to(device, dtype=dtype)
    model_pt.eval()
    
    # --- 2. Load GGUF Model ---
    print("\n[2/3] Instantiating and loading GGUF Model...")
    with torch.device("meta"):
        model_gguf = instantiate_from_config(config.model.params.dit_cfg)
    
    patch_model(model_gguf)
    load_gguf(gguf_path, model_gguf, device=device)
    model_gguf.to(device, dtype=dtype)
    model_gguf.eval()
    
    # --- 3. Forward Pass Comparison ---
    print("\n[3/3] Running Forward Pass Comparison...")
    
    batch_size = 1
    seq_len = 8192
    
    # Create dummy inputs matching the RefineDiT requirements
    # x: [B, N, C], c: [B, N, C], text_states: [B, L, C]
    x = torch.randn(batch_size, seq_len, 64, device=device, dtype=dtype)
    
    # In pipelines.py, voxel_cond is [B, N, 3] usually, but here c is embedded text or voxel features?
    # RefineDiT expects x [B, N, in_channels (64)], and conditional info. 
    # For a simple forward test, we just provide zeros for optional conditioning
    
    c = torch.zeros(batch_size, seq_len, 1024, device=device, dtype=dtype) # Assume context_dim
    text_states = torch.zeros(batch_size, 1370, 1024, device=device, dtype=dtype)
    
    # We will test a specific layer's output to avoid complex conditioning missing
    # Let's compare the first DiT Block instead of the whole unroll if the whole unroll crashes
    block_pt = model_pt.blocks[0]
    block_gguf = model_gguf.blocks[0]
    
    # Input to block is [B, N, hidden_size]
    hidden_size = 2048
    x_block = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)
    
    with torch.no_grad():
        out_pt = block_pt(x_block)
        out_gguf = block_gguf(x_block)
        
    out_pt_f32 = out_pt.float().flatten()
    out_gguf_f32 = out_gguf.float().flatten()
    
    mse = torch.nn.functional.mse_loss(out_gguf_f32, out_pt_f32).item()
    cos_sim = (torch.dot(out_gguf_f32, out_pt_f32) / (torch.norm(out_gguf_f32) * torch.norm(out_pt_f32))).item()
    
    print("\n=== LAYER 0 SUMMARY ===")
    print(f"MSE: {mse:.8e}")
    print(f"Cosine Similarity: {cos_sim:.6f}")
    if cos_sim > 0.999:
        print("Status: PASSED (Near perfect parity)")
    else:
        print("Status: FAILED/MARGINAL")

if __name__ == "__main__":
    compare_parity()
