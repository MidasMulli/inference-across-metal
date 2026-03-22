#!/usr/bin/env python3
"""Profile one layer of Qwen3.5-9B to capture:
1. Exact kernel names dispatched (via Metal GPU capture if available, else structure analysis)
2. Reference input/output tensors for verification
3. Weight tensor shapes and quantization parameters
"""

import json
import os
import time
import struct
import numpy as np

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load

MODEL_PATH = "mlx-community/Qwen3.5-9B-MLX-4bit"
OUTPUT_DIR = "/Users/midas/Desktop/cowork/inference-across-metal/reference_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading model...")
model, tokenizer = load(MODEL_PATH)

# Get the actual model (may be wrapped)
lm = model
if hasattr(lm, 'language_model'):
    lm = lm.language_model
if hasattr(lm, 'model'):
    lm = lm.model

print(f"Model type: {type(lm)}")
print(f"Layers: {len(lm.layers)}")

layer0 = lm.layers[0]
print(f"\nLayer 0 type: {type(layer0)}")
print(f"Layer 0 children: {[name for name, _ in layer0.named_modules()]}")

# ── Catalog layer 0 structure ──
print("\n=== Layer 0 Structure ===")
layer_info = {}
for name, module in layer0.named_modules():
    if name == '':
        continue
    module_type = type(module).__name__
    params = {}
    # Use leaf_modules + direct attribute check instead of named_parameters
    for attr_name in dir(module):
        if attr_name.startswith('_'):
            continue
        try:
            attr = getattr(module, attr_name)
            if isinstance(attr, mx.array):
                params[attr_name] = {
                    'shape': list(attr.shape),
                    'dtype': str(attr.dtype),
                    'nbytes': attr.nbytes,
                }
        except:
            pass
    if params:
        layer_info[name] = {'type': module_type, 'params': params}
        print(f"  {name}: {module_type}")
        for pname, pinfo in params.items():
            print(f"    {pname}: {pinfo['shape']} {pinfo['dtype']} ({pinfo['nbytes']} bytes)")

# ── Check quantization details for linear layers ──
print("\n=== Quantization Details ===")
quant_info = {}
for name, module in layer0.named_modules():
    if hasattr(module, 'weight') and hasattr(module, 'scales'):
        w = module.weight
        s = module.scales
        b = getattr(module, 'biases', None)
        group_size = getattr(module, 'group_size', None)
        bits = getattr(module, 'bits', None)
        info = {
            'weight_shape': list(w.shape),
            'weight_dtype': str(w.dtype),
            'scales_shape': list(s.shape),
            'scales_dtype': str(s.dtype),
            'group_size': group_size,
            'bits': bits,
        }
        if b is not None:
            info['biases_shape'] = list(b.shape)
            info['biases_dtype'] = str(b.dtype)
        quant_info[name] = info
        print(f"  {name}: weight={list(w.shape)} {w.dtype}, scales={list(s.shape)}, gs={group_size}, bits={bits}")

# ── Generate reference input ──
print("\n=== Generating Reference Data ===")

# Create a realistic input: batch=1, seq_len=1 (decode step), dim=4096
hidden_dim = 4096
input_tensor = mx.random.normal((1, 1, hidden_dim), dtype=mx.bfloat16)
mx.eval(input_tensor)

print(f"Input shape: {input_tensor.shape}, dtype: {input_tensor.dtype}")

# We need to trace through layer 0 manually to capture intermediate states
# First, get the layer norm output
input_norm = layer0.input_layernorm(input_tensor)
mx.eval(input_norm)
print(f"After input_layernorm: {input_norm.shape}")

# ── Save the full layer forward pass ──
# Run the full layer (need attention mask and cache for transformer layer)
# For a simpler test, let's capture individual operations

# Test 1: RMS Norm
print("\n--- Test: RMS Norm ---")
rms_input = input_tensor
rms_output = layer0.input_layernorm(rms_input)
mx.eval(rms_output)
print(f"  Input: {rms_input.shape} {rms_input.dtype}")
print(f"  Output: {rms_output.shape} {rms_output.dtype}")

# Test 2: Quantized Linear (MLP up_proj as example)
print("\n--- Test: Quantized Linear (mlp.up_proj) ---")
up_proj = layer0.mlp.up_proj
linear_input = rms_output.reshape(-1, hidden_dim)  # (1, 4096)
linear_output = up_proj(linear_input)
mx.eval(linear_output)
print(f"  Input: {linear_input.shape} {linear_input.dtype}")
print(f"  Output: {linear_output.shape} {linear_output.dtype}")
print(f"  Weight: {up_proj.weight.shape} {up_proj.weight.dtype}")
print(f"  Scales: {up_proj.scales.shape} {up_proj.scales.dtype}")
print(f"  Biases: {up_proj.biases.shape} {up_proj.biases.dtype}")
print(f"  group_size={up_proj.group_size}, bits={up_proj.bits}")

# Test 3: Full MLP
print("\n--- Test: Full MLP ---")
mlp_input = rms_output
mlp_output = layer0.mlp(mlp_input)
mx.eval(mlp_output)
print(f"  Input: {mlp_input.shape} {mlp_input.dtype}")
print(f"  Output: {mlp_output.shape} {mlp_output.dtype}")

# ── Save reference tensors as raw binary ──
def save_tensor(tensor, name):
    """Save tensor as raw binary + metadata JSON."""
    mx.eval(tensor)
    # Save in original dtype as MLX raw bytes
    # For numpy conversion, cast bf16 → f32
    if tensor.dtype == mx.bfloat16:
        arr = np.array(tensor.astype(mx.float32))
    else:
        arr = np.array(tensor)
    raw_path = os.path.join(OUTPUT_DIR, f"{name}.bin")
    meta_path = os.path.join(OUTPUT_DIR, f"{name}.json")
    arr.tofile(raw_path)
    meta = {
        'shape': list(tensor.shape),
        'dtype': str(tensor.dtype),
        'numpy_dtype': str(arr.dtype),
        'nbytes': arr.nbytes,
        'min': float(arr.min()),
        'max': float(arr.max()),
        'mean': float(arr.mean()),
        'first_8': arr.flatten()[:8].tolist(),
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved {name}: {arr.shape} {arr.dtype} ({arr.nbytes} bytes)")

print("\n=== Saving Reference Tensors ===")

# RMS Norm test pair
save_tensor(rms_input, "rms_input")
save_tensor(rms_output, "rms_output")
save_tensor(layer0.input_layernorm.weight, "rms_weight")

# Quantized linear test pair (up_proj)
save_tensor(linear_input, "qlinear_input")
save_tensor(linear_output, "qlinear_output")
save_tensor(up_proj.weight, "qlinear_weight")
save_tensor(up_proj.scales, "qlinear_scales")
save_tensor(up_proj.biases, "qlinear_biases")

# Full MLP test pair
save_tensor(mlp_input, "mlp_input")
save_tensor(mlp_output, "mlp_output")

# ── Save layer structure summary ──
summary = {
    'model': MODEL_PATH,
    'hidden_dim': hidden_dim,
    'layer_info': layer_info,
    'quant_info': quant_info,
    'input_shape': list(input_tensor.shape),
    'input_dtype': str(input_tensor.dtype),
}
with open(os.path.join(OUTPUT_DIR, "layer_summary.json"), 'w') as f:
    json.dump(summary, f, indent=2)
print(f"\nSummary saved to {OUTPUT_DIR}/layer_summary.json")

# ── Profile timing ──
print("\n=== Timing (100 iterations) ===")

# Warmup
for _ in range(10):
    _ = layer0.input_layernorm(input_tensor)
    mx.eval(_)

# RMS Norm timing
times = []
for _ in range(100):
    t0 = time.perf_counter()
    out = layer0.input_layernorm(input_tensor)
    mx.eval(out)
    times.append((time.perf_counter() - t0) * 1000)
print(f"  RMS Norm: {np.mean(times):.3f} ms (std {np.std(times):.3f})")

# Quantized Linear timing (up_proj: 4096 → 12288)
for _ in range(10):
    _ = up_proj(linear_input)
    mx.eval(_)
times = []
for _ in range(100):
    t0 = time.perf_counter()
    out = up_proj(linear_input)
    mx.eval(out)
    times.append((time.perf_counter() - t0) * 1000)
print(f"  up_proj (4096→12288): {np.mean(times):.3f} ms (std {np.std(times):.3f})")

# Full MLP timing
for _ in range(10):
    _ = layer0.mlp(mlp_input)
    mx.eval(_)
times = []
for _ in range(100):
    t0 = time.perf_counter()
    out = layer0.mlp(mlp_input)
    mx.eval(out)
    times.append((time.perf_counter() - t0) * 1000)
print(f"  Full MLP: {np.mean(times):.3f} ms (std {np.std(times):.3f})")

print("\nDone. Reference data saved to:", OUTPUT_DIR)
