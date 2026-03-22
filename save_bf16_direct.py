#!/usr/bin/env python3
"""Save reference tensors as raw bf16 bytes directly from MLX (no f32 round-trip)."""

import os
import json
import struct
import numpy as np
import mlx.core as mx
from mlx_lm import load

OUTPUT_DIR = "/Users/midas/Desktop/cowork/inference-across-metal/reference_data"

print("Loading model...")
model, tokenizer = load("mlx-community/Qwen3.5-9B-MLX-4bit")

lm = model
if hasattr(lm, 'language_model'): lm = lm.language_model
if hasattr(lm, 'model'): lm = lm.model

layer0 = lm.layers[0]
up_proj = layer0.mlp.up_proj

# Use the SAME random input as before — load it from saved f32, convert back to bf16
# This ensures we test the exact same input
input_f32 = np.fromfile(os.path.join(OUTPUT_DIR, "qlinear_input.bin"), dtype=np.float32)
input_f32 = input_f32.reshape(1, 4096)
print(f"Input f32 first 4: {input_f32.flatten()[:4]}")

# Convert to MLX bf16 — this is the actual input the kernel will see
input_bf16 = mx.array(input_f32).astype(mx.bfloat16)
mx.eval(input_bf16)

# Run the quantized linear
output_bf16 = up_proj(input_bf16)
mx.eval(output_bf16)

print(f"Input bf16 shape: {input_bf16.shape}, dtype: {input_bf16.dtype}")
print(f"Output bf16 shape: {output_bf16.shape}, dtype: {output_bf16.dtype}")

# Save raw bf16 bytes using struct — MLX bf16 internal representation
# MLX stores bf16 as 2 bytes per element. We need the raw bytes.
# Method: use mx.save to a temp file and extract, OR cast to uint16 view

# Actually, the cleanest way: save as numpy uint16 by reinterpreting bits
# MLX bf16 → cast to float32 → reinterpret as uint32 → shift right 16 → save as uint16
# But that's what we did before and it introduces rounding...

# Better: use mx.as_strided or direct memory access
# MLX arrays can be exported via numpy IF we handle the dtype

# The issue: numpy doesn't support bf16. So we need to get raw bytes.
# MLX's internal C++ stores bf16 as actual 16-bit values.
# Let's use mx.save to get the raw safetensors representation.

import tempfile

def save_raw_bf16(tensor, name):
    """Save MLX bf16 tensor as raw bytes via safetensors format."""
    mx.eval(tensor)

    # Save via mx.savez (numpy npz format)
    tmp = os.path.join(OUTPUT_DIR, f"_tmp_{name}.npz")
    mx.savez(tmp, data=tensor)

    # Load back the raw bytes from npz
    import zipfile
    with zipfile.ZipFile(tmp, 'r') as zf:
        raw = zf.read('data.npy')
    os.unlink(tmp)

    # numpy .npy format: 128-byte header + raw data
    # Parse header to find offset
    # Magic: \x93NUMPY + major + minor + header_len
    assert raw[:6] == b'\x93NUMPY'
    major = raw[6]
    if major == 1:
        header_len = struct.unpack('<H', raw[8:10])[0]
        data_offset = 10 + header_len
    else:
        header_len = struct.unpack('<I', raw[8:12])[0]
        data_offset = 12 + header_len

    raw_data = raw[data_offset:]
    expected_size = tensor.size * 2  # bf16 = 2 bytes
    print(f"  {name}: raw_data={len(raw_data)} bytes, expected={expected_size} bytes")

    # The npz might have saved as float32 (numpy fallback)...
    if len(raw_data) == tensor.size * 4:
        print(f"  WARNING: npz saved as float32, converting back to bf16 via bit truncation")
        f32 = np.frombuffer(raw_data, dtype=np.float32)
        bf16 = (np.frombuffer(f32.tobytes(), dtype=np.uint32) >> 16).astype(np.uint16)
        raw_data = bf16.tobytes()

    out_path = os.path.join(OUTPUT_DIR, f"{name}.bin")
    with open(out_path, 'wb') as f:
        f.write(raw_data)
    print(f"  Saved {out_path}: {len(raw_data)} bytes")

# Alternative approach: use safetensors directly
def save_raw_bf16_v2(tensor, name):
    """Save MLX tensor as raw bytes via mx.save_safetensors."""
    mx.eval(tensor)
    tmp = os.path.join(OUTPUT_DIR, f"_tmp_{name}.safetensors")
    mx.save_safetensors(tmp, {"data": tensor})

    # Parse safetensors: 8-byte header length + JSON header + raw data
    with open(tmp, 'rb') as f:
        header_size = struct.unpack('<Q', f.read(8))[0]
        header_json = json.loads(f.read(header_size))
        raw_data = f.read()
    os.unlink(tmp)

    # Get tensor offset within the raw data
    meta = header_json['data']
    offsets = meta['data_offsets']
    tensor_bytes = raw_data[offsets[0]:offsets[1]]

    dtype_str = meta['dtype']
    elem_size = 2 if 'F16' in dtype_str or 'BF16' in dtype_str else 4
    print(f"  {name}: dtype={dtype_str}, shape={meta['shape']}, "
          f"raw={len(tensor_bytes)} bytes, expected={tensor.size * elem_size}")

    out_path = os.path.join(OUTPUT_DIR, f"{name}.bin")
    with open(out_path, 'wb') as f:
        f.write(tensor_bytes)
    print(f"  Saved {out_path}: {len(tensor_bytes)} bytes")
    return tensor_bytes

print("\n=== Saving raw bf16 via safetensors ===")

# Save input (bf16)
input_bytes = save_raw_bf16_v2(input_bf16.reshape(-1, 4096), "qlinear_input_bf16")

# Save output (bf16)
output_bytes = save_raw_bf16_v2(output_bf16, "qlinear_output_bf16")

# Save scales (bf16)
scales_bytes = save_raw_bf16_v2(up_proj.scales, "qlinear_scales_bf16")

# Save biases (bf16)
biases_bytes = save_raw_bf16_v2(up_proj.biases, "qlinear_biases_bf16")

# Save weight (uint32) — same as before but let's be sure
weight_bytes = save_raw_bf16_v2(up_proj.weight, "qlinear_weight_raw")

# Also save output as f32 for tolerance comparison
output_f32 = np.array(output_bf16.astype(mx.float32)).flatten()
output_f32.tofile(os.path.join(OUTPUT_DIR, "qlinear_output_f32.bin"))
print(f"\n  Saved qlinear_output_f32.bin: {output_f32.nbytes} bytes")

# Cross-check: print first 8 values
print("\n=== Cross-check first 8 values ===")
out_bf16 = np.frombuffer(output_bytes[:16], dtype=np.uint16)
out_f32_check = output_f32[:8]
print("  Output bf16 (hex):", ' '.join(f'0x{v:04x}' for v in out_bf16))
print("  Output f32:       ", ' '.join(f'{v:.4f}' for v in out_f32_check))

in_bf16 = np.frombuffer(input_bytes[:8], dtype=np.uint16)
print("  Input bf16 (hex): ", ' '.join(f'0x{v:04x}' for v in in_bf16[:4]))

print("\nDone.")
