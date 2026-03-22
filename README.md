# Inference Across Metal

Streaming 27B LLM inference on 16GB Apple Silicon. Pure Swift + Metal, no frameworks, no Python runtime.

Qwen3.5-27B (14.4GB quantized) runs on a MacBook Air M5 with 16GB unified memory. The model doesn't fit in RAM. We stream weights from mmap'd safetensors, one layer at a time, dispatching MLX's pre-compiled NAX kernels from Swift. Persistent GDN state and KV cache stay resident; everything else is loaded, processed, and released.

## Results

**75 tokens of coherent English from a 27B model on 16GB RAM.**

| Metric | Value |
|--------|-------|
| Model | Qwen3.5-27B-MLX-4bit (14.4GB) |
| Hardware | M5 Air, 16GB, 10 GPU cores |
| Forward pass | ~16.4s (SSD page fault limited) |
| Generation rate | 0.066 tok/s (SSD-limited) |
| Batch verify K=32 | Same cost as K=1 (NAX plateau) |
| Pro estimate (weights resident) | 2.5-3 tok/s |

The 0.066 tok/s is real and honest — every forward pass page-faults 14.4GB from SSD. On the 64GB Pro where weights stay resident, per-layer compute is 3.3ms, giving ~400ms per forward pass.

## Architecture

```
Qwen3.5-27B: 64 layers (48 GDN + 16 full attention)
Hidden: 5120, Intermediate: 17408, Vocab: 248320
Heads: 24 Q / 4 KV, Head dim: 256

Memory budget (16GB):
  Embed (resident):     682 MB
  GDN state (resident): 149 MB  (conv + delta across 48 layers)
  KV cache (resident):   64 MB  (16 attn layers x 1024 tokens)
  Activations:          ~100 MB
  Per-layer weights:    ~200 MB  (streamed, released after use)
  Total resident:       ~1 GB
  OS + overhead:        ~1.5 GB
```

Each forward pass:
1. Process prompt token through resident embedding
2. For each of 64 layers: mmap weights -> Metal buffers -> dispatch kernels -> release
3. GDN layers: conv1d state + delta state update (persistent across tokens)
4. Attention layers: RoPE + KV cache append + SDPA decode
5. Final norm -> lm_head -> argmax

## Key Findings

**Batch verification is effectively free.** Verifying K=32 draft tokens costs the same as K=1. NAX loads weights once and processes all tokens from cache. This is the mechanism that makes speculative decoding work on Apple Silicon — draft quality barely matters when verification is amortized.

| Draft tokens (K) | Time (ms) | vs K=1 |
|---|---|---|
| 1 | 4.52 | 1.00x |
| 8 | 3.50 | 0.77x |
| 32 | 3.39 | 0.75x |

**Swift dispatches MLX kernels at 1.47x overhead.** The gap is command buffer dispatch, not kernel execution. We use MLX's pre-compiled metallib (15,749 kernels including NAX variants) without the MLX Python runtime.

**MLX already uses TensorOps.** We audited MLX 0.31.1's Steel kernels — they use cooperative tensors (`MetalPerformancePrimitives.h`) and X-Y swizzled dispatch. The IAM value isn't faster kernels; it's running models that don't fit.

## GDN (GatedDeltaNet) Implementation

48 of 64 layers use GatedDeltaNet — a linear attention variant with persistent state:
- **Conv1d state**: depthwise convolution with kernel=4, double-buffered across tokens
- **Delta state**: 48 layers x 128 x 128 float32 matrices, rank-1 update recurrence per token
- **Gated output**: sigmoid gate + linear combination, no softmax attention

Custom Metal kernels for GDN: `gated_delta_step`, `depthwise_conv1d_step`, `sigmoid_gate_bf16`, `silu_multiply_bf16`.

Verified against MLX reference: 16/18 components PASS (out_proj at 85% from accumulated bf16 rounding, layer output recovers to 91%).

## Files

The repo shows the development path. Each file is a self-contained Swift program that compiles and runs independently.

**Current engine:**
| File | Lines | Purpose |
|------|-------|---------|
| `full_forward.swift` | 1454 | Complete 64-layer forward pass with autoregressive generation |
| `attn_forward.swift` | 1050 | Attention layer verification (14/14 components PASS) |
| `gdn_forward.swift` | 1080 | GDN layer verification (16/18 components PASS) |

**Kill test progression (steps 1-5):**
| File | Purpose |
|------|---------|
| `weight_loader.swift` | Step 1: mmap + zero-copy weight loading (0.065ms) |
| `kernel_dispatch.swift` | Step 2: single NAX kernel dispatch + verification |
| `mlp_pipeline.swift` | Step 3: 4-kernel MLP pipeline, 32-layer benchmark |
| `streaming_27b.swift` | Step 4: 27B streaming from mmap'd safetensors |
| `batch_verify_27b.swift` | Step 5: batch verification plateau measurement |

**Experiments:**
| File | Purpose |
|------|---------|
| `challenge_plateau.swift` | GPU page fault investigation (zero-copy fails) |
| `zero_copy_27b.swift` | Zero-copy attempt (page faults kill throughput) |
| `kernel_*.swift` | Early kernel dispatch iterations |

See [RESULTS.md](RESULTS.md) for detailed measurements from the kill test.

## Build & Run

```bash
# Requires: macOS 26+, Apple Silicon, MLX 0.31.1+ metallib
# Model: ~/models/Qwen3.5-27B-MLX-4bit/ (safetensors + config.json)

swiftc -O -framework Metal -framework MetalPerformanceShaders full_forward.swift -o full_forward
./full_forward
```

The binary looks for the MLX metallib at `~/.mlx-env/lib/python3.11/site-packages/mlx/lib/mlx.metallib` and model weights at `~/models/Qwen3.5-27B-MLX-4bit/`.

## What This Is Not

This is not faster than MLX for models that fit in RAM. MLX's graph compiler, fused evaluation, and lazy scheduling are well-optimized (V2.5 on Apple's TensorOps scale). On a 9B model that fits in memory, MLX wins.

This is for models that don't fit. MLX OOMs. IAM streams.
