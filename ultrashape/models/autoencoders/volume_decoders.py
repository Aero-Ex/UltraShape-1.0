# ==============================================================================
# Original work Copyright (c) 2025 Tencent.
# Modified work Copyright (c) 2025 UltraShape Team.
# 
# Modified by UltraShape on 2025.12.25
# ==============================================================================

# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import time
from typing import Union, Tuple, List, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from tqdm import tqdm

from .attention_blocks import CrossAttentionDecoder
from .attention_processors import FlashVDMCrossAttentionProcessor, FlashVDMTopMCrossAttentionProcessor
from ...utils import logger, log_vram


def extract_near_surface_volume_fn(input_tensor: torch.Tensor, alpha: float):
    val = input_tensor + alpha
    valid_mask = val > -9000

    # sign = torch.sign(val.to(torch.float32))
    # sign = (val > 0) #boolean sign for zero-memory overhead
    sign = val > 0
    mask = torch.ones_like(val, dtype=torch.bool)

    # Pad once for all directions
    padded = F.pad(val.unsqueeze(0).unsqueeze(0), [1, 1, 1, 1, 1, 1], mode='replicate').squeeze(0).squeeze(0)

    # directions: (shift, axis)
    directions = [(1, 0), (-1, 0), (1, 1), (-1, 1), (1, 2), (-1, 2)]

    for shift, axis in directions:
        slice_dims = [slice(1, -1)] * 3
        if axis == 0:
            if shift > 0: slice_dims[0] = slice(2, None)
            else: slice_dims[0] = slice(None, -2)
        elif axis == 1:
            if shift > 0: slice_dims[1] = slice(2, None)
            else: slice_dims[1] = slice(None, -2)
        elif axis == 2:
            if shift > 0: slice_dims[2] = slice(2, None)
            else: slice_dims[2] = slice(None, -2)

        neighbor = padded[tuple(slice_dims)]
        # neighbor = torch.where(neighbor > -9000, neighbor, val)
        # Sign check on boolean masks is just XOR/XNOR or equality
        neighbor_sign = neighbor > 0
        
        # If neighbor is invalid (-9000), treat it as having the same sign to ignore it
        invalid_neighbor = neighbor <= -9000
        is_same = (neighbor_sign == sign) | invalid_neighbor
        
        mask &= is_same

    # Invert mask: we want 1 where ANY neighbor has different sign
    return (~mask & valid_mask).to(torch.int32)


def generate_dense_grid_points(
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    octree_resolution: int,
    indexing: str = "ij",
):
    length = bbox_max - bbox_min
    num_cells = octree_resolution

    x = np.linspace(bbox_min[0], bbox_max[0], int(num_cells) + 1, dtype=np.float32)
    y = np.linspace(bbox_min[1], bbox_max[1], int(num_cells) + 1, dtype=np.float32)
    z = np.linspace(bbox_min[2], bbox_max[2], int(num_cells) + 1, dtype=np.float32)
    [xs, ys, zs] = np.meshgrid(x, y, z, indexing=indexing)
    xyz = np.stack((xs, ys, zs), axis=-1)
    grid_size = [int(num_cells) + 1, int(num_cells) + 1, int(num_cells) + 1]

    return xyz, grid_size, length


class VanillaVolumeDecoder:
    @torch.no_grad()
    def __call__(
        self,
        latents: torch.FloatTensor,
        geo_decoder: Callable,
        bounds: Union[Tuple[float], List[float], float] = 1.01,
        num_chunks: int = 10000,
        octree_resolution: int = None,
        enable_pbar: bool = True,
        **kwargs,
    ):
        device = latents.device
        dtype = latents.dtype
        batch_size = latents.shape[0]

        # 1. generate query points
        if isinstance(bounds, float):
            bounds = [-bounds, -bounds, -bounds, bounds, bounds, bounds]

        bbox_min, bbox_max = np.array(bounds[0:3]), np.array(bounds[3:6])
        xyz_samples, grid_size, length = generate_dense_grid_points(
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            octree_resolution=octree_resolution,
            indexing="ij"
        )
        log_vram(f"Dec: Start (Res: {octree_resolution})")
        xyz_samples = torch.from_numpy(xyz_samples).to(device, dtype=dtype).contiguous().reshape(-1, 3)

        # 2. latents to 3d volume
        batch_logits = []
        for start in tqdm(range(0, xyz_samples.shape[0], num_chunks), desc=f"Volume Decoding",
                          disable=not enable_pbar):
            chunk_queries = xyz_samples[start: start + num_chunks, :]
            chunk_queries = repeat(chunk_queries, "p c -> b p c", b=batch_size)
            log_vram("Dec: Before Geo Query")
            logits = geo_decoder(queries=chunk_queries, latents=latents)
            log_vram("Dec: After Geo Query")
            batch_logits.append(logits)

        grid_logits = torch.cat(batch_logits, dim=1)
        grid_logits = grid_logits.view((batch_size, *grid_size)).float()

        return grid_logits


class HierarchicalVolumeDecoding:
    @torch.no_grad()
    def __call__(
        self,
        latents: torch.FloatTensor,
        geo_decoder: Callable,
        bounds: Union[Tuple[float], List[float], float] = 1.01,
        num_chunks: int = 10000,
        mc_level: float = 0.0,
        octree_resolution: int = None,
        min_resolution: int = 63,
        enable_pbar: bool = True,
        **kwargs,
    ):
        device = latents.device
        dtype = latents.dtype

        resolutions = []
        if octree_resolution < min_resolution:
            resolutions.append(octree_resolution)
        while octree_resolution >= min_resolution:
            resolutions.append(octree_resolution)
            octree_resolution = octree_resolution // 2
        resolutions.reverse()

        # 1. generate query points
        if isinstance(bounds, float):
            bounds = [-bounds, -bounds, -bounds, bounds, bounds, bounds]
        bbox_min = np.array(bounds[0:3])
        bbox_max = np.array(bounds[3:6])
        bbox_size = bbox_max - bbox_min

        xyz_samples, grid_size, length = generate_dense_grid_points(
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            octree_resolution=resolutions[0],
            indexing="ij"
        )

        grid_size = np.array(grid_size)
        xyz_samples = torch.from_numpy(xyz_samples).to(device, dtype=dtype).contiguous().reshape(-1, 3)

        # 2. latents to 3d volume
        batch_logits = []
        batch_size = latents.shape[0]
        for start in tqdm(range(0, xyz_samples.shape[0], num_chunks),
                          desc=f"Hierarchical Volume Decoding [r{resolutions[0] + 1}]"):
            queries = xyz_samples[start: start + num_chunks, :]
            batch_queries = repeat(queries, "p c -> b p c", b=batch_size)
            log_vram("Dec: Before Geo Query (Hier)")
            logits = geo_decoder(queries=batch_queries, latents=latents)
            log_vram("Dec: After Geo Query (Hier)")
            batch_logits.append(logits)

        grid_logits = torch.cat(batch_logits, dim=1).view((batch_size, grid_size[0], grid_size[1], grid_size[2]))

        for octree_depth_now in resolutions[1:]:
            grid_size = np.array([octree_depth_now + 1] * 3)
            resolution = bbox_size / octree_depth_now
            
            # 1. Surface extraction at CURRENT (lower) resolution
            curr_points = extract_near_surface_volume_fn(grid_logits.squeeze(0), mc_level).bool()
            curr_points |= grid_logits.squeeze(0).abs() < 0.95
            
            # 2. Aggressive cleanup of the previous resolution's logits if not the final step
            # Actually we need grid_logits for the next resolution's queries if it's not FlashVDM 
            # But in HierarchicalVolumeDecoding, the queries don't depend on previous logits values, 
            # only on the mask. So we can clear grid_logits now.
            del grid_logits
            torch.cuda.empty_cache()

            # 3. Dilate at LOWER resolution (very light)
            # Dilating once at res N is enough to cover children at res 2N
            curr_points = F.max_pool3d(curr_points.unsqueeze(0).unsqueeze(0).to(dtype), kernel_size=3, stride=1, padding=1).squeeze(0).squeeze(0)
            
            # 4. Sparse Index Expansion (Avoids dense $1000^3$ mask OOM)
            log_vram(f"Dec: Sparse Index Expansion (Hier Res: {octree_depth_now})")
            cidx = torch.where(curr_points > 0)
            del curr_points
            
            # Generate 8 children for each point at resolution N
            x, y, z = cidx
            offsets = torch.tensor([
                [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]
            ], device=device, dtype=torch.long)
            base_idx = torch.stack([x.long(), y.long(), z.long()], dim=1) * 2
            expanded = base_idx.unsqueeze(1) + offsets.unsqueeze(0)
            expanded = expanded.view(-1, 3)
            
            # Filter bounds
            valid_mask = (expanded[:, 0] < grid_size[0]) & (expanded[:, 1] < grid_size[1]) & (expanded[:, 2] < grid_size[2])
            expanded = expanded[valid_mask]
            nidx = (expanded[:, 0], expanded[:, 1], expanded[:, 2])
            
            next_index_shape = tuple(grid_size)
            del expanded
            del base_idx
            del cidx
            torch.cuda.empty_cache()

            next_points = torch.stack(nidx, dim=1)
            next_points = (next_points * torch.tensor(resolution, dtype=torch.float32, device=device) +
                           torch.tensor(bbox_min, dtype=torch.float32, device=device))
            
            batch_logits = []
            for start in tqdm(range(0, next_points.shape[0], num_chunks),
                              desc=f"Hierarchical Volume Decoding [r{octree_depth_now + 1}]"):
                queries = next_points[start: start + num_chunks, :]
                batch_queries = repeat(queries, "p c -> b p c", b=batch_size)
                logits = geo_decoder(queries=batch_queries.to(latents.dtype), latents=latents)
                batch_logits.append(logits)

            # 5. Populate the NEXT dense grid
            next_logits = torch.full(next_index_shape, -10000., dtype=dtype, device=device)
            grid_logits_sparse = torch.cat(batch_logits, dim=1)
            next_logits[nidx] = grid_logits_sparse[0, ..., 0]
            
            grid_logits = next_logits.unsqueeze(0)
            del next_logits
            del grid_logits_sparse
            del batch_logits
            torch.cuda.empty_cache()

        # Final NaN conversion: Chunked to avoid 1GB boolean mask spike on 1024^3 grids
        for i in range(grid_logits.shape[1]):
            layer = grid_logits[:, i]
            layer[layer == -10000.] = float('nan')
        return grid_logits


class FlashVDMVolumeDecoding:
    def __init__(self, topk_mode='mean'):
        if topk_mode not in ['mean', 'merge']:
            raise ValueError(f'Unsupported topk_mode {topk_mode}, available: {["mean", "merge"]}')

        if topk_mode == 'mean':
            self.processor = FlashVDMCrossAttentionProcessor()
        else:
            self.processor = FlashVDMTopMCrossAttentionProcessor()

    @torch.no_grad()
    def __call__(
        self,
        latents: torch.FloatTensor,
        geo_decoder: CrossAttentionDecoder,
        bounds: Union[Tuple[float], List[float], float] = 1.01,
        num_chunks: int = 10000,
        mc_level: float = 0.0,
        octree_resolution: int = None,
        min_resolution: int = 63,
        mini_grid_num: int = 4,
        enable_pbar: bool = True,
        **kwargs,
    ):
        processor = self.processor
        geo_decoder.set_cross_attention_processor(processor)

        device = latents.device
        dtype = latents.dtype

        resolutions = []
        if octree_resolution < min_resolution:
            resolutions.append(octree_resolution)
        while octree_resolution >= min_resolution:
            resolutions.append(octree_resolution)
            log_vram(f"Dec: Start (Res: {octree_resolution})")
            octree_resolution = octree_resolution // 2
        resolutions.reverse()
        resolutions[0] = round(resolutions[0] / mini_grid_num) * mini_grid_num - 1
        for i, resolution in enumerate(resolutions[1:]):
            resolutions[i + 1] = resolutions[0] * 2 ** (i + 1)

        logger.info(f"FlashVDMVolumeDecoding Resolution: {resolutions}")

        # 1. generate query points
        if isinstance(bounds, float):
            bounds = [-bounds, -bounds, -bounds, bounds, bounds, bounds]
        bbox_min = np.array(bounds[0:3])
        bbox_max = np.array(bounds[3:6])
        bbox_size = bbox_max - bbox_min

        xyz_samples, grid_size, length = generate_dense_grid_points(
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            octree_resolution=resolutions[0],
            indexing="ij"
        )

        grid_size = np.array(grid_size)

        # 2. latents to 3d volume
        xyz_samples = torch.from_numpy(xyz_samples).to(device, dtype=dtype)
        batch_size = latents.shape[0]
        mini_grid_size = xyz_samples.shape[0] // mini_grid_num
        xyz_samples = xyz_samples.view(
            mini_grid_num, mini_grid_size,
            mini_grid_num, mini_grid_size,
            mini_grid_num, mini_grid_size, 3
        ).permute(
            0, 2, 4, 1, 3, 5, 6
        ).reshape(
            -1, mini_grid_size * mini_grid_size * mini_grid_size, 3
        )
        batch_logits = []
        num_batchs = max(num_chunks // xyz_samples.shape[1], 1)
        for start in tqdm(range(0, xyz_samples.shape[0], num_batchs),
                          desc=f"FlashVDM Volume Decoding", disable=not enable_pbar):
            queries = xyz_samples[start: start + num_batchs, :]
            batch = queries.shape[0]
            batch_latents = repeat(latents.squeeze(0), "p c -> b p c", b=batch)
            processor.topk = True

            # Chunk queries along dim 1 if too large
            if queries.shape[1] > num_chunks:
                batch_logits_sub = []
                for sub_start in range(0, queries.shape[1], num_chunks):
                    sub_queries = queries[:, sub_start: sub_start + num_chunks, :]
                    logits = geo_decoder(queries=sub_queries, latents=batch_latents)
                    batch_logits_sub.append(logits)
                logits = torch.cat(batch_logits_sub, dim=1)
            else:
                logits = geo_decoder(queries=queries, latents=batch_latents)

            batch_logits.append(logits)
        grid_logits = torch.cat(batch_logits, dim=0).reshape(
            mini_grid_num, mini_grid_num, mini_grid_num,
            mini_grid_size, mini_grid_size,
            mini_grid_size
        ).permute(0, 3, 1, 4, 2, 5).contiguous().view(
            (batch_size, grid_size[0], grid_size[1], grid_size[2])
        )

        for octree_depth_now in resolutions[1:]:
            grid_size = np.array([octree_depth_now + 1] * 3)
            resolution = bbox_size / octree_depth_now
            
            # 1. Surface extraction at CURRENT (lower) resolution
            curr_points = extract_near_surface_volume_fn(grid_logits.squeeze(0), mc_level).bool()
            curr_points |= grid_logits.squeeze(0).abs() < 0.95
            
            # 2. Cleanup previous resolution logits (we only need the binary mask for next steps)
            del grid_logits
            torch.cuda.empty_cache()

            # 3. Dilate at LOWER resolution (very light)
            # Dilating once at low res is enough to cover children and their immediate neighbors at high res
            curr_points = F.max_pool3d(curr_points.unsqueeze(0).unsqueeze(0).to(dtype), kernel_size=3, stride=1, padding=1).squeeze(0).squeeze(0)
            
            # 4. Sparse Index Expansion (Avoids dense $1000^3$ mask OOM)
            log_vram(f"FlashVDM: Sparse Index Expansion (Hier Res: {octree_depth_now})")
            t_expand_start = time.time()
            cidx = torch.where(curr_points > 0)
            del curr_points
            
            # Generate 8 children for each point at resolution N
            x, y, z = cidx
            offsets = torch.tensor([
                [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]
            ], device=device, dtype=torch.long)
            base_idx = torch.stack([x.long(), y.long(), z.long()], dim=1) * 2
            expanded = base_idx.unsqueeze(1) + offsets.unsqueeze(0)
            expanded = expanded.view(-1, 3)
            
            # Filter bounds
            valid_mask = (expanded[:, 0] < grid_size[0]) & (expanded[:, 1] < grid_size[1]) & (expanded[:, 2] < grid_size[2])
            expanded = expanded[valid_mask]
            nidx = (expanded[:, 0], expanded[:, 1], expanded[:, 2])
            
            next_index_shape = tuple(grid_size)
            del expanded
            del base_idx
            del cidx
            torch.cuda.empty_cache()

            t_expand = time.time() - t_expand_start
            log_vram(f"FlashVDM: Point Generation (Res: {octree_depth_now}) | Takes: {t_expand:.2f}s")
            t_prep_start = time.time()
            next_points = torch.stack(nidx, dim=1)
            next_points = (next_points * torch.tensor(resolution, dtype=torch.float32, device=device) +
                           torch.tensor(bbox_min, dtype=torch.float32, device=device))

            query_grid_num = 6
            min_val = next_points.min(axis=0).values
            max_val = next_points.max(axis=0).values
            vol_queries_index = (next_points - min_val) / (max_val - min_val) * (query_grid_num - 0.001)
            index = torch.floor(vol_queries_index).long()
            index = index[..., 0] * (query_grid_num ** 2) + index[..., 1] * query_grid_num + index[..., 2]
            index = index.sort()
            next_points = next_points[index.indices].unsqueeze(0).contiguous()

            t_prep = time.time() - t_prep_start
            log_vram(f"FlashVDM: Points Prepared ({next_points.shape[1]} pts) | Takes: {t_prep:.2f}s")
            unique_values = torch.unique(index.values, return_counts=True)
            grid_logits_sparse = torch.zeros((next_points.shape[1]), dtype=latents.dtype, device=latents.device)
            input_grid = [[], []]
            logits_grid_list = []
            start_num = 0
            sum_num = 0
            t_decode_start = time.time()
            for grid_index, count in zip(unique_values[0].cpu().tolist(), unique_values[1].cpu().tolist()):
                remaining_count = count
                while remaining_count > 0:
                    space_left = num_chunks - sum_num
                    # If buffer is full, flush it
                    if space_left <= 0:
                        processor.topk = input_grid
                        logits_grid = geo_decoder(queries=next_points[:, start_num:start_num + sum_num], latents=latents)
                        start_num = start_num + sum_num
                        logits_grid_list.append(logits_grid)
                        input_grid = [[], []]
                        sum_num = 0
                        space_left = num_chunks
                    take = min(remaining_count, space_left)
                    input_grid[0].append(grid_index)
                    input_grid[1].append(take)
                    sum_num += take
                    remaining_count -= take
            if sum_num > 0:
                processor.topk = input_grid
                log_vram(f"FlashVDM: Final Geo Query (Res: {octree_depth_now})")
                logits_grid = geo_decoder(queries=next_points[:, start_num:start_num + sum_num], latents=latents)
                logits_grid_list.append(logits_grid)
            
            t_decode = time.time() - t_decode_start
            log_vram(f"FlashVDM: Decoding Done (Res: {octree_depth_now}) | Takes: {t_decode:.2f}s")
            
            logits_grid = torch.cat(logits_grid_list, dim=1)
            grid_logits_sparse[index.indices] = logits_grid.squeeze(0).squeeze(-1)
            del logits_grid_list
            del logits_grid
            torch.cuda.empty_cache()

            # 5. Populate NEXT dense grid
            next_logits = torch.full(next_index_shape, -10000., dtype=dtype, device=device)
            next_logits[nidx] = grid_logits_sparse
            grid_logits = next_logits.unsqueeze(0)
            
            del next_logits
            del grid_logits_sparse
            torch.cuda.empty_cache()

        # Final NaN conversion: Chunked to avoid 1GB boolean mask spike on 1024^3 grids
        for i in range(grid_logits.shape[1]):
            layer = grid_logits[:, i]
            layer[layer == -10000.] = float('nan')
        return grid_logits
