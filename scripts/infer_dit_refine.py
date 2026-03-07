import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import sys
import argparse
import torch
import warnings
# Suppress the noisy RequestsDependencyWarning from the environment
try:
    from requests.exceptions import RequestsDependencyWarning
    warnings.filterwarnings('ignore', category=RequestsDependencyWarning)
except ImportError:
    pass
import numpy as np
import gc
from PIL import Image
from omegaconf import OmegaConf

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ultrashape.rembg import BackgroundRemover
from ultrashape.utils.misc import instantiate_from_config
from ultrashape.surface_loaders import SharpEdgeSurfaceLoader
from ultrashape.utils import voxelize_from_point
from ultrashape.pipelines import UltraShapePipeline 
from ultrashape.utils.gguf_loader import patch_model, load_gguf

def sequential_load_safetensors(model, path, device='cpu'):
    from safetensors import safe_open
    print(f"Sequentially loading {path} to {device}...")
    
    # Use named_parameters and named_buffers to avoid state_dict() overhead
    weight_map = {k: v for k, v in model.named_parameters()}
    weight_map.update({k: v for k, v in model.named_buffers()})
    
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            if key in weight_map:
                param = weight_map[key]
                with torch.no_grad():
                    param.data.copy_(f.get_tensor(key).to(device))
            else:
                # Some keys might be prefixed, handle potential common prefixes
                print(f"Skipping {key} (not in model)")
    del weight_map
    gc.collect()

def load_models(config_path, ckpt_path, device='cuda', args=None):
    print(f"Loading config from {config_path}...")
    config = OmegaConf.load(config_path)
    
    weights = None
    if ckpt_path and os.path.exists(ckpt_path):
        print(f"Loading weights from {ckpt_path} to CPU (mmap=True)...")
        try:
            # Use mmap=True to avoid loading everything into RAM at once
            weights = torch.load(ckpt_path, map_location='cpu', mmap=True, weights_only=True)
        except Exception as e:
            print(f"Standard mmap load failed ({e}), trying without weights_only...")
            try:
                weights = torch.load(ckpt_path, map_location='cpu', mmap=True)
            except Exception as e2:
                print(f"mmap load failed ({e2}), falling back to direct load...")
                weights = torch.load(ckpt_path, map_location='cpu')

    def cleanup():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 1. Sequential VAE
    print("Instantiating VAE...")
    vae = instantiate_from_config(config.model.params.vae_config)
    if args and args.vae and os.path.exists(args.vae):
        sequential_load_safetensors(vae, args.vae, device=device)
    elif weights and 'vae' in weights:
        vae.load_state_dict(weights['vae'], strict=True)
        del weights['vae']
    else:
        print("Warning: No VAE weights found/provided.")
    
    vae.eval().to(device)
    if hasattr(vae, 'enable_flashvdm_decoder'):
        vae.enable_flashvdm_decoder()
    cleanup()

    # 2. Sequential Conditioner
    print("Instantiating Conditioner...")
    conditioner = instantiate_from_config(config.model.params.conditioner_config)
    if args and args.conditioner and os.path.exists(args.conditioner):
        sequential_load_safetensors(conditioner, args.conditioner, device=device)
    elif weights and 'conditioner' in weights:
        conditioner.load_state_dict(weights['conditioner'], strict=True)
        del weights['conditioner']
    else:
        print("Warning: No Conditioner weights found/provided.")
    
    conditioner.eval().to(device)
    cleanup()

    # 3. Sequential DiT
    print("Instantiating DiT...")
    if args and args.gguf:
        # Use meta device to avoid ~6GB CPU RAM spike
        with torch.device("meta"):
            dit = instantiate_from_config(config.model.params.dit_cfg)
        print(f"Patching DiT and loading GGUF weights from {args.gguf}...")
        patch_model(dit)
        load_gguf(args.gguf, dit, device=device)
        dit.eval().to(device)
    elif weights and 'dit' in weights:
        dit = instantiate_from_config(config.model.params.dit_cfg)
        dit.load_state_dict(weights['dit'], strict=True)
        del weights['dit']
        dit.eval().to(device)
    else:
        dit = instantiate_from_config(config.model.params.dit_cfg)
        print("Warning: No DiT weights found/provided.")
        dit.eval().to(device)
    
    cleanup()

    # 4. Other components
    print("Instantiating Scheduler & Processor...")
    scheduler = instantiate_from_config(config.model.params.scheduler_cfg)
    image_processor = instantiate_from_config(config.model.params.image_processor_cfg)
    
    if weights is not None:
        del weights
        cleanup()

    components = {
        "vae": vae,
        "dit": dit,
        "conditioner": conditioner,
        "scheduler": scheduler,
        "image_processor": image_processor,
    }
    
    return components, config

def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    components, config = load_models(args.config, args.ckpt, device, args)
    
    pipeline = UltraShapePipeline(
        vae=components['vae'],
        model=components['dit'],
        scheduler=components['scheduler'],
        conditioner=components['conditioner'],
        image_processor=components['image_processor']
    )

    if args.compile:
        print("Compiling models...")
        pipeline.compile()

    if args.low_vram:
        print("Enabling official model CPU offloading...")
        pipeline.enable_model_cpu_offload()

    token_num = args.num_latents
    voxel_res = config.model.params.vae_config.params.voxel_query_res
    
    print(f"Initializing Surface Loader (Token Num: {token_num})...")
    loader = SharpEdgeSurfaceLoader(
        num_sharp_points=204800,
        num_uniform_points=204800,
    )

    print(f"Processing inputs: {args.image} & {args.mesh}")
    image = Image.open(args.image)
    
    if args.remove_bg or image.mode != 'RGBA':
        rembg = BackgroundRemover()
        image = rembg(image)
    
    surface = loader(args.mesh, normalize_scale=args.scale).to(device, dtype=torch.bfloat16)
    pc = surface[:, :, :3] # [B, N, 3]
    
    # Voxelize
    _, voxel_idx = voxelize_from_point(pc, token_num, resolution=voxel_res)
    
    print("Running diffusion process...")
    generator = torch.Generator(device).manual_seed(args.seed)
    
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        mesh, _ = pipeline(
            image=image,
            voxel_cond=voxel_idx,
            generator=generator,
            box_v=1.0,
            mc_level=0.0,
            octree_resolution=args.octree_res,
            num_inference_steps=args.steps,
            num_chunks=args.chunk_size,
            high_token_mode=args.high_token_mode,
            moe_offload=args.moe_offload,
            sequential_cfg=args.seq_cfg,
        )
    
    os.makedirs(args.output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.image))[0]
    save_path = os.path.join(args.output_dir, f"{base_name}_refined.glb")
    
    mesh = mesh[0]
    mesh.export(save_path)
    print(f"Successfully saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UltraShape Inference Script")
    
    parser.add_argument("--config", type=str, default="configs/infer_dit_refine.yaml", help="Path to inference config")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to split checkpoint (.pt) (optional if GGUF/VAE/Cond paths provided)")
    parser.add_argument("--gguf", type=str, default=None, help="Path to GGUF model (replaces DiT in .pt)")
    parser.add_argument("--vae", type=str, default=None, help="Path to VAE .safetensors (replaces VAE in .pt)")
    parser.add_argument("--conditioner", type=str, default=None, help="Path to Conditioner .safetensors (replaces Conditioner in .pt)")
    parser.add_argument("--low_vram", action="store_true", help="Optimize for low VRAM usage")
    
    parser.add_argument("--image", type=str, required=True, help="Input image path")
    parser.add_argument("--mesh", type=str, required=True, help="Input coarse mesh (.glb/.obj)")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    
    parser.add_argument("--steps", type=int, default=50, help="Inference steps")
    parser.add_argument("--scale", type=float, default=0.99, help="Mesh normalization scale")
    parser.add_argument("--num_latents", type=int, default=32768, help="Number of latents")
    parser.add_argument("--chunk_size", type=int, default=8000, help="Chunk size for inference")
    parser.add_argument("--octree_res", type=int, default=1024, help="Marching Cubes resolution")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--remove_bg", action="store_true", help="Force remove background")
    
    parser.add_argument("--high_token_mode", action="store_true", help="Enable optimizations for 16k+ tokens")
    parser.add_argument("--moe_offload", action="store_true", help="Offload MoE experts to CPU sequentially")
    parser.add_argument("--seq_cfg", action="store_true", help="Process CFG passes sequentially to save memory")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile for models")

    args = parser.parse_args()
    
    run_inference(args)
