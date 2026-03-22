#!/usr/bin/env python3
"""Run quantized_matmul directly and verify the bf16 reference matches."""

import numpy as np
import struct
import mlx.core as mx
from mlx_lm import load

REF = "/Users/midas/Desktop/cowork/inference-across-metal/reference_data"

print("Loading model...")
model, _ = load("mlx-community/Qwen3.5-9B-MLX-4bit")
lm = model
if hasattr(lm, 'language_model'): lm = lm.language_model
if hasattr(lm, 'model'): lm = lm.model
up = lm.layers[0].mlp.up_proj

# Load the saved bf16 input via safetensors approach
with open(f"{REF}/qlinear_input_bf16.bin", "rb") as f:
    input_bytes = f.read()

# Reconstruct the MLX bf16 array from raw bytes
# Save as temp safetensors, load back
import json, tempfile, os
tmp = f"{REF}/_tmp_load.safetensors"

# Build a minimal safetensors file with our raw bytes
header = {
    "data": {
        "dtype": "BF16",
        "shape": [1, 4096],
        "data_offsets": [0, len(input_bytes)]
    }
}
header_json = json.dumps(header).encode()
# Pad to 8-byte alignment
while len(header_json) % 8 != 0:
    header_json += b' '

with open(tmp, 'wb') as f:
    f.write(struct.pack('<Q', len(header_json)))
    f.write(header_json)
    f.write(input_bytes)

loaded = mx.load(tmp)
x_bf16 = loaded["data"]
os.unlink(tmp)
mx.eval(x_bf16)

print(f"Loaded input: shape={x_bf16.shape}, dtype={x_bf16.dtype}")
x_f32 = x_bf16.astype(mx.float32)
mx.eval(x_f32)
print(f"First 4: {np.array(x_f32).flatten()[:4]}")

# Run quantized matmul
y = up(x_bf16)
mx.eval(y)
print(f"Output: shape={y.shape}, dtype={y.dtype}")

# Save output as raw bf16 for comparison
mx.save_safetensors(f"{REF}/_tmp_verify.safetensors", {"data": y})
with open(f"{REF}/_tmp_verify.safetensors", 'rb') as f:
    hs = struct.unpack('<Q', f.read(8))[0]
    hj = json.loads(f.read(hs))
    raw = f.read()
meta = hj['data']
out_bytes = raw[meta['data_offsets'][0]:meta['data_offsets'][1]]
os.unlink(f"{REF}/_tmp_verify.safetensors")

# Compare against saved reference
with open(f"{REF}/qlinear_output_bf16.bin", "rb") as f:
    ref_bytes = f.read()

out_u16 = np.frombuffer(out_bytes, dtype=np.uint16)
ref_u16 = np.frombuffer(ref_bytes, dtype=np.uint16)

exact = np.sum(out_u16 == ref_u16)
print(f"\nVerify-vs-ref bit-exact: {exact}/{len(out_u16)} ({100*exact/len(out_u16):.1f}%)")
print(f"First 8 (verify): {['0x{:04x}'.format(v) for v in out_u16[:8]]}")
print(f"First 8 (ref):    {['0x{:04x}'.format(v) for v in ref_u16[:8]]}")

# Check if the model's weight/scales/biases match our saved data
# Load saved weights
with open(f"{REF}/qlinear_weight_raw.bin", "rb") as f:
    saved_w = f.read()
with open(f"{REF}/qlinear_scales_bf16.bin", "rb") as f:
    saved_s = f.read()
with open(f"{REF}/qlinear_biases_bf16.bin", "rb") as f:
    saved_b = f.read()

# Get model weights as raw bytes
for name, tensor in [("weight", up.weight), ("scales", up.scales), ("biases", up.biases)]:
    mx.eval(tensor)
    mx.save_safetensors(f"{REF}/_tmp_{name}.safetensors", {"data": tensor})
    with open(f"{REF}/_tmp_{name}.safetensors", 'rb') as f:
        hs = struct.unpack('<Q', f.read(8))[0]
        hj = json.loads(f.read(hs))
        raw = f.read()
    m = hj['data']
    live_bytes = raw[m['data_offsets'][0]:m['data_offsets'][1]]
    os.unlink(f"{REF}/_tmp_{name}.safetensors")

    if name == "weight":
        saved = saved_w
    elif name == "scales":
        saved = saved_s
    else:
        saved = saved_b

    match = live_bytes == saved
    print(f"\n{name}: saved={len(saved)}, live={len(live_bytes)}, match={match}")
    if not match:
        s_arr = np.frombuffer(saved[:32], dtype=np.uint8)
        l_arr = np.frombuffer(live_bytes[:32], dtype=np.uint8)
        diff = np.sum(s_arr != l_arr)
        print(f"  First 32 bytes differ: {diff}")

# Also run mx.quantized_matmul directly
print("\n=== Direct mx.quantized_matmul ===")
y2 = mx.quantized_matmul(x_bf16, up.weight, up.scales, up.biases,
                          group_size=up.group_size, bits=up.bits, transpose=True)
mx.eval(y2)

# Compare y vs y2
y_f32 = np.array(y.astype(mx.float32)).flatten()
y2_f32 = np.array(y2.astype(mx.float32)).flatten()
diff = np.abs(y_f32 - y2_f32)
print(f"up(x) vs quantized_matmul: max_diff={diff.max()}, exact_match={np.all(diff==0)}")
print(f"First 8 up(x):    {y_f32[:8]}")
print(f"First 8 qmm:      {y2_f32[:8]}")
