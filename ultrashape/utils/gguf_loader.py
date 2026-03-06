import torch
import torch.nn.functional as F
import gguf
import logging
import os
from tqdm import tqdm
from functools import wraps

# --- Dequantization Logic (Adapted from ComfyUI-GGUF) ---

TORCH_COMPATIBLE_QTYPES = (None, gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16)

def to_uint32(x):
    x = x.view(torch.uint8).to(torch.int32)
    return (x[:, 0] | x[:, 1] << 8 | x[:, 2] << 16 | x[:, 3] << 24).unsqueeze(1)

def to_uint16(x):
    x = x.view(torch.uint8).to(torch.int32)
    return (x[:, 0] | x[:, 1] << 8).unsqueeze(1)

def split_block_dims(blocks, *args):
    n_max = blocks.shape[1]
    dims = list(args) + [n_max - sum(args)]
    return torch.split(blocks, dims, dim=1)

def dequantize_blocks_BF16(blocks, block_size, type_size, dtype=None):
    return (blocks.view(torch.int16).to(torch.int32) << 16).view(torch.float32)

def dequantize_blocks_Q8_0(blocks, block_size, type_size, dtype=None):
    d, x = split_block_dims(blocks, 2)
    d = d.view(torch.float16).to(dtype)
    x = x.view(torch.int8)
    return (d * x)

def dequantize_blocks_Q4_0(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d, qs = split_block_dims(blocks, 2)
    d  = d.view(torch.float16).to(dtype)
    qs = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    qs = (qs & 0x0F).reshape((n_blocks, -1)).to(torch.int8) - 8
    return (d * qs)

def dequantize_blocks_Q5_1(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d, m, qh, qs = split_block_dims(blocks, 2, 2, 4)
    d = d.view(torch.float16).to(dtype)
    m = m.view(torch.float16).to(dtype)
    qh = to_uint32(qh)
    qh = qh.reshape((n_blocks, 1)) >> torch.arange(32, device=d.device, dtype=torch.int32).reshape(1, 32)
    ql = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape(1, 1, 2, 1)
    qh = (qh & 1).to(torch.uint8)
    ql = (ql & 0x0F).reshape((n_blocks, -1))
    qs = (ql | (qh << 4))
    return (d * qs) + m

def dequantize_blocks_Q5_0(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d, qh, qs = split_block_dims(blocks, 2, 4)
    d  = d.view(torch.float16).to(dtype)
    qh = to_uint32(qh)
    qh = qh.reshape(n_blocks, 1) >> torch.arange(32, device=d.device, dtype=torch.int32).reshape(1, 32)
    ql = qs.reshape(n_blocks, -1, 1, block_size // 2) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape(1, 1, 2, 1)
    qh = (qh & 1).to(torch.uint8)
    ql = (ql & 0x0F).reshape(n_blocks, -1)
    qs = (ql | (qh << 4)).to(torch.int8) - 16
    return (d * qs)

def dequantize_blocks_Q4_1(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d, m, qs = split_block_dims(blocks, 2, 2)
    d = d.view(torch.float16).to(dtype)
    m = m.view(torch.float16).to(dtype)
    qs = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape(1, 1, 2, 1)
    qs = (qs & 0x0F).reshape(n_blocks, -1)
    return (d * qs) + m

def dequantize_blocks_Q4_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d, dmin, scales, qs = split_block_dims(blocks, 2, 2, 12)
    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)
    sc, m = get_scale_min(scales)
    d = (d * sc).reshape((n_blocks, -1, 1))
    dm = (dmin * m).reshape((n_blocks, -1, 1))
    qs = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    qs = (qs & 0x0F).reshape((n_blocks, -1, 32))
    return (d * qs - dm).reshape((n_blocks, 256))

def get_scale_min(scales):
    n_blocks = scales.shape[0]
    scales = scales.view(torch.uint8)
    scales = scales.reshape((n_blocks, 3, 4))
    d, m, m_d = torch.split(scales, 1, dim=-2)
    sc = torch.cat([d & 0x3F, (m_d & 0x0F) | ((d >> 2) & 0x30)], dim=-1)
    min = torch.cat([m & 0x3F, (m_d >> 4) | ((m >> 2) & 0x30)], dim=-1)
    return (sc.reshape((n_blocks, 8)), min.reshape((n_blocks, 8)))

def dequantize_blocks_Q6_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    ql, qh, scales, d, = split_block_dims(blocks, 128, 64, 16)
    scales = scales.view(torch.int8).to(dtype)
    d = d.view(torch.float16).to(dtype)
    d = (d * scales).reshape((n_blocks, 16, 1))
    ql = ql.reshape((n_blocks, -1, 1, 64)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    ql = (ql & 0x0F).reshape((n_blocks, -1, 32))
    qh = qh.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape((1, 1, 4, 1))
    qh = (qh & 0x03).reshape((n_blocks, -1, 32))
    q = (ql | (qh << 4)).to(torch.int8) - 32
    q = q.reshape((n_blocks, 16, -1))
    return (d * q).reshape((n_blocks, 256))

def dequantize_blocks_Q5_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d, dmin, scales, qh, qs = split_block_dims(blocks, 2, 2, 12, 32)
    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)
    sc, m = get_scale_min(scales)
    d = (d * sc).reshape((n_blocks, -1, 1))
    dm = (dmin * m).reshape((n_blocks, -1, 1))
    ql = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    qh = qh.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([i for i in range(8)], device=d.device, dtype=torch.uint8).reshape((1, 1, 8, 1))
    ql = (ql & 0x0F).reshape((n_blocks, -1, 32))
    qh = (qh & 0x01).reshape((n_blocks, -1, 32))
    q = (ql | (qh << 4))
    return (d * q - dm).reshape((n_blocks, 256))

def dequantize_blocks_Q3_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    hmask, qs, scales, d = split_block_dims(blocks, 32, 64, 12)
    d = d.view(torch.float16).to(dtype)
    lscales, hscales = scales[:, :8], scales[:, 8:]
    lscales = lscales.reshape((n_blocks, 1, 8)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 2, 1))
    lscales = lscales.reshape((n_blocks, 16))
    hscales = hscales.reshape((n_blocks, 1, 4)) >> torch.tensor([0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape((1, 4, 1))
    hscales = hscales.reshape((n_blocks, 16))
    scales = (lscales & 0x0F) | ((hscales & 0x03) << 4)
    scales = (scales.to(torch.int8) - 32)
    dl = (d * scales).reshape((n_blocks, 16, 1))
    ql = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape((1, 1, 4, 1))
    qh = hmask.reshape(n_blocks, -1, 1, 32) >> torch.tensor([i for i in range(8)], device=d.device, dtype=torch.uint8).reshape((1, 1, 8, 1))
    ql = ql.reshape((n_blocks, 16, 16)) & 3
    qh = (qh.reshape((n_blocks, 16, 16)) & 1) ^ 1
    q = (ql.to(torch.int8) - (qh << 2).to(torch.int8))
    return (dl * q).reshape((n_blocks, 256))

def dequantize_blocks_Q2_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    scales, qs, d, dmin = split_block_dims(blocks, 16, 64, 2)
    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)
    dl = (d * (scales & 0xF)).reshape((n_blocks, 16, 1))
    ml = (dmin * (scales >> 4)).reshape((n_blocks, 16, 1))
    shift = torch.tensor([0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape((1, 1, 4, 1))
    qs = (qs.reshape((n_blocks, -1, 1, 32)) >> shift) & 3
    qs = qs.reshape((n_blocks, 16, 16))
    qs = dl * qs - ml
    return qs.reshape((n_blocks, -1))

KVALUES = torch.tensor([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113], dtype=torch.int8)

def dequantize_blocks_IQ4_NL(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d, qs = split_block_dims(blocks, 2)
    d = d.view(torch.float16).to(dtype)
    qs = qs.reshape((n_blocks, -1, 1, block_size//2)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    qs = (qs & 0x0F).reshape((n_blocks, -1, 1)).to(torch.int64)
    kvalues = KVALUES.to(qs.device).expand(*qs.shape[:-1], 16)
    qs = torch.gather(kvalues, dim=-1, index=qs).reshape((n_blocks, -1))
    return (d * qs)

def dequantize_blocks_IQ4_XS(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d, scales_h, scales_l, qs = split_block_dims(blocks, 2, 2, 4)
    d = d.view(torch.float16).to(dtype)
    scales_h = to_uint16(scales_h)
    shift_a = torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 1, 2))
    shift_b = torch.tensor([2 * i for i in range(8)], device=d.device, dtype=torch.uint8).reshape((1, -1, 1))
    scales_l = scales_l.reshape((n_blocks, -1, 1)) >> shift_a.reshape((1, 1, 2))
    scales_h = scales_h.reshape((n_blocks, -1, 1)) >> shift_b.reshape((1, -1, 1))
    scales_l = scales_l.reshape((n_blocks, -1)) & 0x0F
    scales_h = scales_h.reshape((n_blocks, -1)).to(torch.uint8) & 0x03
    scales = (scales_l | (scales_h << 4)).to(torch.int8) - 32
    dl = (d * scales.to(dtype)).reshape((n_blocks, -1, 1))
    qs = qs.reshape((n_blocks, -1, 1, 16)) >> shift_a.reshape((1, 1, 2, 1))
    qs = qs.reshape((n_blocks, -1, 32, 1)) & 0x0F
    kvalues = KVALUES.to(qs.device).expand(*qs.shape[:-1], 16)
    qs = torch.gather(kvalues, dim=-1, index=qs.to(torch.int64)).reshape((n_blocks, -1, 32))
    return (dl * qs).reshape((n_blocks, -1))

dequantize_functions = {
    gguf.GGMLQuantizationType.BF16: dequantize_blocks_BF16,
    gguf.GGMLQuantizationType.Q8_0: dequantize_blocks_Q8_0,
    gguf.GGMLQuantizationType.Q5_1: dequantize_blocks_Q5_1,
    gguf.GGMLQuantizationType.Q5_0: dequantize_blocks_Q5_0,
    gguf.GGMLQuantizationType.Q4_1: dequantize_blocks_Q4_1,
    gguf.GGMLQuantizationType.Q4_0: dequantize_blocks_Q4_0,
    gguf.GGMLQuantizationType.Q6_K: dequantize_blocks_Q6_K,
    gguf.GGMLQuantizationType.Q5_K: dequantize_blocks_Q5_K,
    gguf.GGMLQuantizationType.Q4_K: dequantize_blocks_Q4_K,
    gguf.GGMLQuantizationType.Q3_K: dequantize_blocks_Q3_K,
    gguf.GGMLQuantizationType.Q2_K: dequantize_blocks_Q2_K,
    gguf.GGMLQuantizationType.IQ4_NL: dequantize_blocks_IQ4_NL,
    gguf.GGMLQuantizationType.IQ4_XS: dequantize_blocks_IQ4_XS,
}

def get_torch_compiler_disable_decorator():
    if hasattr(torch, "compiler") and hasattr(torch.compiler, "disable"):
        return torch.compiler.disable
    return lambda x: x

torch_compiler_disable = get_torch_compiler_disable_decorator()

def dequantize_tensor(tensor, dtype=None, qtype=None, qshape=None):
    if qtype is None:
        qtype = getattr(tensor, "tensor_type", None)
    
    # Selection of target dtype if None or uint8 (likely a mistake from calling layer)
    actual_dtype = dtype
    if actual_dtype is None or actual_dtype == torch.uint8:
        actual_dtype = torch.float16 if tensor.device.type == 'cuda' else torch.float32

    oshape = qshape
    if oshape is None:
        oshape = getattr(tensor, "tensor_shape", tensor.shape)
        
    if isinstance(oshape, torch.Size):
        oshape = tuple(oshape)
    else:
        oshape = tuple(int(x) for x in oshape)

    if qtype in TORCH_COMPATIBLE_QTYPES:
        return tensor.to(actual_dtype)
    if qtype in dequantize_functions:
        block_size, type_size = gguf.GGML_QUANT_SIZES[qtype]
        dequantize_blocks = dequantize_functions[qtype]
        rows = tensor.data.reshape((-1, tensor.data.shape[-1])).view(torch.uint8)
        n_blocks = rows.numel() // type_size
        blocks = rows.reshape((n_blocks, type_size))
        return dequantize_blocks(blocks, block_size, type_size, actual_dtype).reshape(oshape).to(actual_dtype)
    return torch.from_numpy(gguf.quants.dequantize(tensor.cpu().numpy(), qtype)).to(tensor.device, dtype=actual_dtype)

# --- Classes for Weight Handling ---

class GGMLTensor(torch.Tensor):
    def __init__(self, *args, tensor_type, tensor_shape, **kwargs):
        super().__init__()
        self.tensor_type = tensor_type
        self.tensor_shape = tensor_shape

    def __new__(cls, *args, tensor_type, tensor_shape, **kwargs):
        return super().__new__(cls, *args, **kwargs)

    def to(self, *args, **kwargs):
        new = super().to(*args, **kwargs)
        new.tensor_type = getattr(self, "tensor_type", None)
        new.tensor_shape = getattr(self, "tensor_shape", new.data.shape)
        return new

class GGMLLinear(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(in_features, out_features, bias, **factory_kwargs)

    @torch_compiler_disable()
    def forward(self, input):
        qtype = getattr(self.weight, "tensor_type", getattr(self, "weight_type", None))
        qshape = getattr(self.weight, "tensor_shape", getattr(self, "weight_shape", None))
        
        # Determine target dtype based on input (usually float16 or bfloat16)
        target_dtype = input.dtype
        if target_dtype == torch.uint8:
            target_dtype = torch.float16 if input.device.type == 'cuda' else torch.float32
            input = input.to(target_dtype)

        if qtype is not None and qtype not in TORCH_COMPATIBLE_QTYPES:
            w = dequantize_tensor(self.weight, target_dtype, qtype=qtype, qshape=qshape)
            
            # Nan guard (useful for debugging instability)
            if torch.isnan(w).any():
                print(f"Warning: NaN detected in dequantized weight for {qtype}")
                w = torch.nan_to_num(w)
            
            # Range check for the first few layers or if it's suspicious
            if not hasattr(GGMLLinear, "_debug_printed"):
                print(f"DEBUG: Weight range for {qtype}: min={w.min().item():.4f}, max={w.max().item():.4f}, mean={w.mean().item():.4f}")
                GGMLLinear._debug_printed = True
                
            b = self.bias.to(target_dtype) if self.bias is not None else None
            return torch.nn.functional.linear(input, w, b)
        
        return super().forward(input)

class GGMLEmbedding(torch.nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2., scale_grad_by_freq=False, sparse=False, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(num_embeddings, embedding_dim, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse, **factory_kwargs)

    @torch_compiler_disable()
    def forward(self, input):
        qtype = getattr(self.weight, "tensor_type", getattr(self, "weight_type", None))
        qshape = getattr(self.weight, "tensor_shape", getattr(self, "weight_shape", None))
        if qtype is not None and qtype not in TORCH_COMPATIBLE_QTYPES:
            w = dequantize_tensor(self.weight, torch.float32, qtype=qtype, qshape=qshape) # Embedding usually better in F32
            return torch.nn.functional.embedding(
                input, w, self.padding_idx, self.max_norm,
                self.norm_type, self.scale_grad_by_freq, self.sparse
            ).to(dtype=self.weight.dtype if self.weight.dtype != torch.uint8 else torch.float32)
        return super().forward(input)

class GGMLConv2d(torch.nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, **factory_kwargs)

    @torch_compiler_disable()
    def forward(self, input):
        qtype = getattr(self.weight, "tensor_type", getattr(self, "weight_type", None))
        qshape = getattr(self.weight, "tensor_shape", getattr(self, "weight_shape", None))
        if qtype is not None and qtype not in TORCH_COMPATIBLE_QTYPES:
            w = dequantize_tensor(self.weight, input.dtype, qtype=qtype, qshape=qshape)
            b = self.bias.to(input.dtype) if self.bias is not None else None
            return self._conv_forward(input, w, b)
        return super().forward(input)

class GGMLMoEGate(torch.nn.Module):
    def __init__(self, original_gate):
        super().__init__()
        # Copy everything from original gate
        for k, v in original_gate.__dict__.items():
            if k not in ['_modules', '_parameters', '_buffers']:
                setattr(self, k, v)
        for k, v in original_gate._modules.items():
            self.add_module(k, v)
        for k, v in original_gate._parameters.items():
            self.register_parameter(k, v)
        for k, v in original_gate._buffers.items():
            self.register_buffer(k, v)

    @torch_compiler_disable()
    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)
        
        input_dtype = hidden_states.dtype
        if input_dtype == torch.uint8:
            input_dtype = torch.float16 if hidden_states.device.type == 'cuda' else torch.float32
            hidden_states = hidden_states.to(input_dtype)

        qtype = getattr(self.weight, "tensor_type", getattr(self, "weight_type", None))
        qshape = getattr(self.weight, "tensor_shape", getattr(self, "weight_shape", None))
        
        if qtype is not None and qtype not in TORCH_COMPATIBLE_QTYPES:
            w = dequantize_tensor(self.weight, input_dtype, qtype=qtype, qshape=qshape)
            logits = F.linear(hidden_states, w, None)
        else:
            logits = F.linear(hidden_states, self.weight, None)

        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'Unsupported scoring function for MoE gating: {self.scoring_func}')
        
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        # Aux loss is usually skipped in inference, but let's keep it safe
        aux_loss = None
        if self.training and getattr(self, 'alpha', 0.0) > 0.0:
            # Note: We don't implement full aux loss logic here as we assume inference
            pass
            
        return topk_idx, topk_weight, aux_loss

# --- Patching & Loading ---

def patch_model(model):
    """Recursively swap standard layers with GGML-aware versions."""
    for name, module in model.named_children():
        mod_type = type(module).__name__
        if isinstance(module, torch.nn.Linear):
            new = GGMLLinear(module.in_features, module.out_features, module.bias is not None, 
                             device=module.weight.device, dtype=module.weight.dtype)
            new.weight, new.bias = module.weight, module.bias
            setattr(model, name, new)
        elif isinstance(module, torch.nn.Conv2d):
            new = GGMLConv2d(module.in_channels, module.out_channels, module.kernel_size,
                            module.stride, module.padding, module.dilation,
                            module.groups, module.bias is not None, module.padding_mode,
                            device=module.weight.device, dtype=module.weight.dtype)
            new.weight, new.bias = module.weight, module.bias
            setattr(model, name, new)
        elif isinstance(module, torch.nn.Embedding):
            new = GGMLEmbedding(module.num_embeddings, module.embedding_dim, module.padding_idx,
                               module.max_norm, module.norm_type, module.scale_grad_by_freq, module.sparse,
                               device=module.weight.device, dtype=module.weight.dtype)
            new.weight = module.weight
            setattr(model, name, new)
        elif mod_type == 'MoEGate':
            new = GGMLMoEGate(module)
            setattr(model, name, new)
        else:
            patch_model(module)

def load_gguf(path, model, device="cpu"):
    """Load GGUF tensors into a PyTorch model's parameters."""
    def rsetattr(obj, attr, val):
        pre, _, post = attr.rpartition('.')
        if pre:
            parent = obj
            for part in pre.split('.'):
                parent = getattr(parent, part)
            setattr(parent, post, val)
        else:
            setattr(obj, post, val)

    reader = gguf.GGUFReader(path)
    # Map GGUF tensors to a flat dict
    sd = {t.name: t for t in reader.tensors}
    
    import gc
    missing, loaded = [], []
    for i, (name, param) in enumerate(model.named_parameters()):
        if i % 100 == 0:
            gc.collect()
            if torch.cuda.is_available() and device == "cuda":
                torch.cuda.empty_cache()
        
        if name in sd:
            t = sd[name]
            data = torch.from_numpy(t.data).to(device)
            with torch.no_grad():
                # We must be careful not to break accelerate's internal tracking
                pre, _, post = name.rpartition('.')
                parent = model
                if pre:
                    for part in pre.split('.'):
                        parent = getattr(parent, part)
                
                existing_param = getattr(parent, post)
                
                # Set metadata directly on the parent to survive offloading
                tensor_shape = tuple(int(x) for x in reversed(t.shape))
                if name.endswith('.weight'):
                    parent.weight_type = t.tensor_type
                    parent.weight_shape = tensor_shape
                elif name.endswith('.bias'):
                    parent.bias_type = t.tensor_type
                    parent.bias_shape = tensor_shape
                
                # Safely copy data or replace if it's a meta tensor
                if existing_param.is_meta:
                    new_param = torch.nn.Parameter(data, requires_grad=False)
                    new_param.tensor_type = t.tensor_type
                    new_param.tensor_shape = tensor_shape
                    setattr(parent, post, new_param)
                else:
                    existing_param.requires_grad = False
                    existing_param.data = data
                    existing_param.tensor_type = t.tensor_type
                    existing_param.tensor_shape = tensor_shape
                    existing_param.tensor_shape = tensor_shape
            loaded.append(name)
        else:
            if param.device.type == 'meta':
                # Materialize missing parameters (biases, norms) on target device
                with torch.no_grad():
                    new_data = torch.zeros_like(param, device=device, dtype=param.dtype)
                    new_param = torch.nn.Parameter(new_data, requires_grad=param.requires_grad)
                    rsetattr(model, name, new_param)
            elif device != param.device:
                # Move existing parameters to target device
                rsetattr(model, name, torch.nn.Parameter(param.data.to(device), requires_grad=param.requires_grad))
            missing.append(name)
    
    print(f"Loaded {len(loaded)} GGUF tensors. {len(missing)} parameters remained unchanged.")
    return loaded, missing

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python gguf_loader.py <model.gguf>")
        sys.exit(1)
    
    reader = gguf.GGUFReader(sys.argv[1])
    print(f"GGUF Metadata for {sys.argv[1]}:")
    for tensor in reader.tensors:
        print(f"Tensor: {tensor.name:<50} Shape: {tensor.shape!s:<20} Type: {tensor.tensor_type!s}")
