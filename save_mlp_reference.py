#!/usr/bin/env python3
"""Save full MLP reference data (gate_proj, up_proj, down_proj) for Swift pipeline test."""

import os
import json
import numpy as np
import mlx.core as mx
from mlx_lm import load

MODEL_PATH = "mlx-community/Qwen3.5-9B-MLX-4bit"
OUTPUT_DIR = "/Users/midas/Desktop/cowork/inference-across-metal/reference_data"

print("Loading model...")
model, tokenizer = load(MODEL_PATH)
lm = model
if hasattr(lm, 'language_model'): lm = lm.language_model
if hasattr(lm, 'model'): lm = lm.model

layer0 = lm.layers[0]
mlp = layer0.mlp

print(f"Layers: {len(lm.layers)}")
print(f"MLP type: {type(mlp)}")

# Print MLP structure
for name in ['gate_proj', 'up_proj', 'down_proj']:
    proj = getattr(mlp, name)
    print(f"\n{name}:")
    print(f"  weight: {proj.weight.shape} {proj.weight.dtype}")
    print(f"  scales: {proj.scales.shape} {proj.scales.dtype}")
    print(f"  biases: {proj.biases.shape} {proj.biases.dtype}")
    print(f"  group_size={proj.group_size}, bits={proj.bits}")

# Generate input (reuse the saved bf16 input from Step 2)
input_bf16_path = os.path.join(OUTPUT_DIR, "qlinear_input_bf16.bin")
if os.path.exists(input_bf16_path):
    raw = np.fromfile(input_bf16_path, dtype=np.uint16)
    # Convert uint16 bf16 bits to float32, then to MLX bf16
    f32 = np.array([np.float32(np.frombuffer(x.tobytes() + b'\x00\x00', dtype=np.float32)[0]) for x in raw])
    mlp_input = mx.array(f32.reshape(1, -1), dtype=mx.bfloat16)
    print(f"\nLoaded input from Step 2: {mlp_input.shape} {mlp_input.dtype}")
else:
    mlp_input = mx.random.normal((1, 4096), dtype=mx.bfloat16)
    print(f"\nGenerated random input: {mlp_input.shape}")

mx.eval(mlp_input)

# Run each projection individually to get reference outputs
print("\n=== Running individual projections ===")

gate_out = mlp.gate_proj(mlp_input)
mx.eval(gate_out)
print(f"gate_proj: {mlp_input.shape} -> {gate_out.shape}")

up_out = mlp.up_proj(mlp_input)
mx.eval(up_out)
print(f"up_proj: {mlp_input.shape} -> {up_out.shape}")

# SwiGLU: silu(gate) * up
import mlx.nn as nn
hidden = nn.silu(gate_out) * up_out
mx.eval(hidden)
print(f"silu(gate) * up: {hidden.shape}")

down_out = mlp.down_proj(hidden)
mx.eval(down_out)
print(f"down_proj: {hidden.shape} -> {down_out.shape}")

# Full MLP forward for end-to-end reference
mlp_full_out = mlp(mlp_input.reshape(1, 1, -1))
mx.eval(mlp_full_out)
mlp_full_out = mlp_full_out.reshape(1, -1)
print(f"\nFull MLP output: {mlp_full_out.shape}")

# Verify individual matches full
diff = mx.abs(down_out - mlp_full_out).max().item()
print(f"Individual vs full MLP max diff: {diff}")

# Save everything as raw bf16 bytes via safetensors
def save_bf16(tensor, name):
    t = tensor.astype(mx.bfloat16) if tensor.dtype != mx.bfloat16 else tensor
    mx.eval(t)
    path = os.path.join(OUTPUT_DIR, f"{name}.bin")
    tmp = "/tmp/_save_tmp.safetensors"
    mx.save_safetensors(tmp, {"data": t})
    # Extract raw bytes (skip safetensors header)
    with open(tmp, "rb") as f:
        header_size = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(header_size))
        info = header["data"]
        offsets = info["data_offsets"]
        f.seek(8 + header_size + offsets[0])
        raw = f.read(offsets[1] - offsets[0])
    with open(path, "wb") as f:
        f.write(raw)
    print(f"  Saved {name}: {len(raw)} bytes")
    return raw

def save_raw(tensor, name):
    """Save uint32 weight tensor as raw bytes."""
    mx.eval(tensor)
    arr = np.array(tensor)
    path = os.path.join(OUTPUT_DIR, f"{name}.bin")
    arr.tofile(path)
    print(f"  Saved {name}: {arr.nbytes} bytes")

print("\n=== Saving MLP reference data ===")

# Gate proj weights
save_raw(mlp.gate_proj.weight, "mlp_gate_weight_raw")
save_bf16(mlp.gate_proj.scales, "mlp_gate_scales_bf16")
save_bf16(mlp.gate_proj.biases, "mlp_gate_biases_bf16")

# Down proj weights
save_raw(mlp.down_proj.weight, "mlp_down_weight_raw")
save_bf16(mlp.down_proj.scales, "mlp_down_scales_bf16")
save_bf16(mlp.down_proj.biases, "mlp_down_biases_bf16")

# Reference outputs
save_bf16(gate_out, "mlp_gate_output_bf16")
save_bf16(up_out, "mlp_up_output_bf16")
save_bf16(hidden, "mlp_hidden_bf16")
save_bf16(down_out, "mlp_down_output_bf16")
save_bf16(mlp_full_out, "mlp_full_output_bf16")

# Also save f32 versions for tolerance comparison
def save_f32(tensor, name):
    t = tensor.astype(mx.float32) if tensor.dtype != mx.float32 else tensor
    mx.eval(t)
    arr = np.array(t)
    path = os.path.join(OUTPUT_DIR, f"{name}.bin")
    arr.tofile(path)
    print(f"  Saved {name}: {arr.nbytes} bytes")

save_f32(gate_out, "mlp_gate_output_f32")
save_f32(up_out, "mlp_up_output_f32")
save_f32(hidden, "mlp_hidden_f32")
save_f32(down_out, "mlp_down_output_f32")

# Save metadata
meta = {
    "input_shape": list(mlp_input.shape),
    "gate_proj": {
        "weight_shape": list(mlp.gate_proj.weight.shape),
        "scales_shape": list(mlp.gate_proj.scales.shape),
        "biases_shape": list(mlp.gate_proj.biases.shape),
        "output_shape": list(gate_out.shape),
        "group_size": mlp.gate_proj.group_size,
        "bits": mlp.gate_proj.bits,
    },
    "up_proj": {
        "weight_shape": list(mlp.up_proj.weight.shape),
        "scales_shape": list(mlp.up_proj.scales.shape),
        "biases_shape": list(mlp.up_proj.biases.shape),
        "output_shape": list(up_out.shape),
        "group_size": mlp.up_proj.group_size,
        "bits": mlp.up_proj.bits,
    },
    "down_proj": {
        "weight_shape": list(mlp.down_proj.weight.shape),
        "scales_shape": list(mlp.down_proj.scales.shape),
        "biases_shape": list(mlp.down_proj.biases.shape),
        "output_shape": list(down_out.shape),
        "group_size": mlp.down_proj.group_size,
        "bits": mlp.down_proj.bits,
    },
    "hidden_shape": list(hidden.shape),
    "full_output_shape": list(mlp_full_out.shape),
    "first_4_gate": [float(x) for x in gate_out.astype(mx.float32).reshape(-1)[:4].tolist()],
    "first_4_up": [float(x) for x in up_out.astype(mx.float32).reshape(-1)[:4].tolist()],
    "first_4_down": [float(x) for x in down_out.astype(mx.float32).reshape(-1)[:4].tolist()],
}

with open(os.path.join(OUTPUT_DIR, "mlp_meta.json"), "w") as f:
    json.dump(meta, f, indent=2)
print(f"\nMetadata saved to mlp_meta.json")

# Timing
print("\n=== MLP Timing ===")
import time

# Warmup
for _ in range(10):
    _ = mlp(mlp_input.reshape(1, 1, -1))
    mx.eval(_)

times = []
for _ in range(100):
    t0 = time.perf_counter()
    out = mlp(mlp_input.reshape(1, 1, -1))
    mx.eval(out)
    times.append((time.perf_counter() - t0) * 1000)

print(f"  Full MLP: {np.mean(times):.3f} ms (std {np.std(times):.3f})")
print(f"  This is the target for Swift pipeline.")
