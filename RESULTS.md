# Inference Across Metal — Kill Test Results

Hardware: Apple M5 Air, 16GB unified memory, 10 GPU cores, macOS 26.3
Model: Qwen3.5-9B-MLX-4bit (9B) and Qwen3.5-27B-MLX-4bit (27B)
MLX: 0.31.1, metallib with 15,749 kernels including NAX variants

## Gate Results

| Step | Test | Gate Criteria | Result | Status |
|------|------|---------------|--------|--------|
| 0 | Download models | Models available | 9B + 27B downloaded | PASS |
| 1 | Zero-copy weight loading | < 3ms for 117MB layer | **0.065ms** (zero-copy mmap) | PASS |
| 2 | One-layer kernel dispatch | Output within tolerance of MLX | 100% < 0.1, 93% < 0.01 | PASS |
| 3 | MLP pipeline (32 layers) | Per-layer within 2x of MLX | **1.47x** (1.99ms vs 1.35ms) | PASS |
| 4 | 27B streaming | Valid output from 15GB mmap | Layer 0 correct, 3.3ms pre-loaded | PASS |
| 5 | Batch verification plateau | M=8 within 1.5x of M=1 | **M=8 = 0.77x of M=1** | PASS |

## Detailed Findings

### Step 1: Zero-Copy Weight Loading (0.065ms)
- mmap'd 5.3GB safetensors file, gathered 30 tensors (117MB) scattered across 4GB span
- `makeBuffer(bytesNoCopy:)` on page-aligned region: 0.065ms (no data copy)
- `makeBuffer(bytes:)` gather copy: 0.636ms (still 10x faster than mx.load's 6.6ms)
- Bandwidth: 184 GB/s for contiguous memcpy

### Step 2: One-Layer Kernel Dispatch
- Kernel: `affine_qmm_t_nax_bfloat16_t_gs_64_b_4_bm64_bn64_bk64_wm2_wn2_alN_true_batch_0`
- Operation: quantized matmul (1, 4096) x (12288, 4096)^T → (1, 12288), 4-bit affine, group_size=64
- Dispatch: 8 Metal buffers (w, scales, biases, x, y, K, N, M), grid 192x1, threadgroup (32,2,2)
- Result: 100% within 0.1 tolerance, 93% within 0.01, max error 0.023
- 20.4% bit-exact vs `mx.quantized_matmul` — FP accumulation ordering difference
- 99.97% bit-exact vs `mx.dequantize → matmul` — confirms kernel executes correctly
- Timing: 3.5ms single dispatch (MLX: 0.45ms — MLX amortizes across fused graph)

### Step 3: Full MLP Pipeline (9B)
- Chained 4 kernels: gate_proj + up_proj + silu_multiply + down_proj
- Custom `silu_multiply_bf16` Metal kernel (fused silu activation + element-wise multiply)
- Verification: all 4 stages within tolerance of MLX Python reference
- Per-layer: 1.99ms batched (1.47x MLX's 1.35ms)
- 32-layer batched (single command buffer): 63ms total
- Double-buffering provides no benefit (sequential layer dependency)
- The 0.6ms overhead is command buffer dispatch, not per-kernel

### Step 4: 27B Streaming (15GB model on 16GB RAM)
- mmap'd 3 safetensors files (5GB each), 64 layers
- MLP per layer: 143MB (gate/up/down projections)
- Streaming (mmap→Metal): 62ms/layer (SSD page fault limited)
- Pre-loaded (weights in Metal buffers): 3.3ms/layer
- Pre-load all 64 layers: 10.5s (9GB of MLP weights)
- Per-layer compute scales linearly with weight size: 3.3ms (27B) vs 1.99ms (9B) = 1.66x

### Step 5: Batch Verification Plateau (27B)

| Draft tokens (M) | Time (ms) | Ratio vs M=1 |
|---|---|---|
| 1 | 4.52 | 1.00x |
| 2 | 3.50 | 0.77x |
| 4 | 3.48 | 0.77x |
| 8 | 3.50 | 0.77x |
| 16 | 3.57 | 0.79x |
| 32 | 3.39 | 0.75x |

**M=8 is faster than M=1.** Dispatch overhead dominates at M=1; at M≥2, NAX loads weights once and processes all tokens from cache. Cost is flat from M=2 to M=32. This confirms the gate_zero finding on the 9B and extends it to the 27B.

Full 64-layer forward pass: M=1 = 7.4s, M=8 = 213ms (warm cache after M=1 run).

## Architecture Implications

1. **Swift can dispatch MLX Metal kernels at 1.47x overhead.** The overhead is command buffer dispatch, not kernel execution. Amortized across a full model, this is acceptable.

2. **Weight streaming from mmap works on memory-constrained hardware.** The 27B (15GB) runs on 16GB RAM. SSD page faults add 62ms/layer, but this drops to 3.3ms when weights are cached.

3. **Batch verification is effectively free.** The cost of verifying K=8 draft tokens is the same as K=1. This means speculative decoding on Apple Silicon should aggressively generate drafts — the verification cost is dominated by weight loading, not token count.

4. **On the 64GB Pro**, all 27B weights stay resident (no page faults), giving 3.3ms/layer × 64 = 211ms per full MLP pass. With attention and norms, estimate ~400-500ms per token for the full model.

## Files

| File | Purpose |
|------|---------|
| `weight_loader.swift` | Step 1: mmap + zero-copy weight loading |
| `kernel_dispatch.swift` | Step 2: single kernel dispatch + verification |
| `mlp_pipeline.swift` | Step 3: 4-kernel MLP pipeline + 32-layer benchmark |
| `streaming_27b.swift` | Step 4: 27B streaming from mmap'd safetensors |
| `batch_verify_27b.swift` | Step 5: batch verification plateau measurement |
| `save_bf16_direct.py` | Generate bf16 reference data from MLX |
| `save_mlp_reference.py` | Generate full MLP reference data |
| `profile_one_layer.py` | Profile layer structure and quantization |
| `verify_kernel.py` | Verify saved reference data against MLX |
| `reference_data/` | Binary reference tensors + metadata |
