import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import gc
from omegaconf import OmegaConf
from safetensors.torch import safe_open
from ultrashape.utils.gguf_loader import patch_model, load_gguf
from ultrashape.utils.misc import instantiate_from_config
from ultrashape.pipelines import UltraShapePipeline

def print_vram(tag):
    allocated = torch.cuda.memory_allocated() / (1024**2)
    reserved = torch.cuda.memory_reserved() / (1024**2)
    print(f"[{tag}] Allocated: {allocated:.1f}MB, Reserved: {reserved:.1f}MB")

def main():
    print_vram("Start")
    config = OmegaConf.load("configs/infer_dit_refine.yaml")
    
    with torch.device("meta"):
        vae = instantiate_from_config(config.model.params.vae_config)
        dit = instantiate_from_config(config.model.params.dit_cfg)
        
    conditioner = instantiate_from_config(config.model.params.conditioner_config)
    scheduler = instantiate_from_config(config.model.params.scheduler_cfg)
    image_processor = instantiate_from_config(config.model.params.image_processor_cfg)
    
    vae.to_empty(device="cpu")
    dit.to_empty(device="cpu")
    conditioner.to("cpu")
    print_vram("After Meta Instantiation to CPU")
    
    def seq_load(m, p):
        keys = set(m.state_dict().keys())
        with safe_open(p, framework="pt", device="cpu") as f:
            for k in f.keys():
                if k in keys:
                    obj = m
                    for part in k.split(".")[:-1]: obj = getattr(obj, part)
                    param = getattr(obj, k.split(".")[-1])
                    with torch.no_grad(): param.data.copy_(f.get_tensor(k))

    seq_load(vae, "/home/aero/ComfyUI-GGUF/ultrashape_vae.safetensors")
    seq_load(conditioner, "/home/aero/ComfyUI-GGUF/ultrashape_conditioner.safetensors")
    print_vram("After Loading VAE & Cond to CPU")
    
    patch_model(dit)
    load_gguf("/home/aero/ComfyUI-GGUF/ultrashape_v1-Q8.gguf", dit, device="cpu")
    print_vram("After Loading GGUF DiT to CPU")
    
    pipeline = UltraShapePipeline(vae=vae, model=dit, scheduler=scheduler, conditioner=conditioner, image_processor=image_processor)
    pipeline.enable_model_cpu_offload(device=torch.device("cuda"))
    print_vram("After Accelerate Offload Setup")

if __name__ == "__main__":
    main()
