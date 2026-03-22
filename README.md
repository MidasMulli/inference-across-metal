# Inference Across Metal

Streaming 27B LLM inference on 16GB Apple Silicon. Pure Swift + Metal, no frameworks, no Python runtime. Qwen3.5-27B runs on a MacBook Air M5 with 16GB unified memory by streaming weights from mmap'd safetensors one layer at a time.

The model is 14.4GB quantized. macOS needs ~4.6GB. The model does not fit in RAM. MLX OOMs. IAM streams.

## Results

75 tokens of coherent English from a 27B model on 16GB RAM.

| Metric | Value |
|--------|-------|
| Model | Qwen3.5-27B-MLX-4bit (14.4GB) |
| Hardware | M5 Air, 16GB, 10 GPU cores |
| Cold forward pass | ~16.4s (SSD page fault on every layer) |
| Warm forward pass | ~7.5s (MADV_WILLNEED prefetch) |
| Cold generation | 0.066 tok/s |
| Warm generation | 0.133 tok/s (2x cold, zero extra memory) |
| Batch verify K=32 | Same cost as K=1 (NAX plateau) |
| Per-layer compute (pre-loaded) | 3.3ms |
| Pro estimate (weights resident) | 2.5-3 tok/s (~400ms forward pass) |

The 0.066 tok/s cold path is real and honest -- every forward pass page-faults 14.4GB from SSD. The warm path uses `madvise(MADV_WILLNEED)` to prefetch the next layer while the GPU processes the current one. No extra memory allocated. On the 64GB Pro where weights stay resident, per-layer compute is 3.3ms, giving ~400ms per forward pass.

## Architecture

```
Qwen3.5-27B: 64 layers (48 GDN + 16 full attention)

  Layer pattern: every 4th layer is full attention (layers 3, 7, 11, ...)
  Hidden: 5120, Intermediate: 17408, Vocab: 248320
  Heads: 24 Q / 4 KV, Head dim: 256
  GDN: 16 K-heads, 48 V-heads, conv kernel=4, state 128x128 per head

Memory budget (16GB):
  Embed (resident):     682 MB
  GDN state (resident): 149 MB  (conv1d + delta state across 48 layers)
  KV cache (resident):   64 MB  (16 attn layers x 1024 tokens)
  Activations:          ~100 MB
  Per-layer weights:    ~200 MB  (streamed, released after use)
  Total resident:       ~1 GB
  OS + overhead:        ~1.5 GB
  Available for mmap:   ~13.5 GB (page cache, evicted freely by OS)
```

Each forward pass:
1. Process token through resident embedding table
2. For each of 64 layers: mmap weights -> page-aligned Metal buffers -> dispatch kernels -> release
3. GDN layers (48): conv1d state + delta state update (persistent across tokens)
4. Attention layers (16): RoPE + GQA KV cache append + SDPA decode
5. Final RMS norm -> lm_head -> argmax

## Key Findings

### Batch verification is effectively free

Verifying K=32 draft tokens costs the same as K=1. NAX loads weights once and processes all tokens from cache. This is the mechanism that makes speculative decoding work on Apple Silicon -- draft quality barely matters when verification is amortized.

| Draft tokens (K) | Time (ms) | vs K=1 |
|---|---|---|
| 1 | 4.52 | 1.00x |
| 2 | 3.50 | 0.77x |
| 4 | 3.48 | 0.77x |
| 8 | 3.50 | 0.77x |
| 16 | 3.57 | 0.79x |
| 32 | 3.39 | 0.75x |

Cost is flat from K=2 to K=32. Dispatch overhead dominates at K=1; at K>=2, the weight load is amortized across all tokens. This extends the [gate_zero finding on 9B](https://github.com/MidasMulli/four-path-mlx) to the 27B.

### GPU page faults: zero-copy mmap is 10x WORSE than memcpy

The intuitive approach -- `makeBuffer(bytesNoCopy:)` on mmap'd regions -- triggers GPU page faults that stall the entire pipeline. The GPU cannot handle page faults gracefully; each one blocks until the kernel reads the page from SSD.

The solution: `makeBuffer(bytesNoCopy:)` on **page-aligned spans** of the mmap'd file. The CPU pre-faults the pages (via `madvise` or a simple read), then hands a clean, resident region to the GPU. No copy, no fault.

| Method | Time | Notes |
|--------|------|-------|
| `mx.load` (MLX Python) | 6.6ms | Deserialize + copy |
| `makeBuffer(bytes:)` gather copy | 0.636ms | CPU memcpy to Metal buffer |
| `makeBuffer(bytesNoCopy:)` page-aligned | 0.065ms | Zero-copy, 102x faster than mx.load |
| `makeBuffer(bytesNoCopy:)` naive mmap | ~60ms | GPU page faults, 10x worse than memcpy |

### Swap-backed vs file-backed memory

Pre-loading all weights into Metal shared buffers sounds smart, but on 16GB it causes swap thrashing. Metal shared buffers are **swap-backed** -- when evicted, they must be written to swap before the page can be reused. mmap'd file pages are **file-backed** -- when evicted, the OS just drops them and re-reads from the original file. On memory-constrained hardware, mmap wins because eviction is free.

### MADV_WILLNEED prefetch

Calling `madvise(MADV_WILLNEED)` on the next layer's weight region while the GPU processes the current layer gives a 2x speedup (16.4s -> 7.5s per forward pass) at zero memory cost. The kernel begins paging in the next layer's data asynchronously. This is the single largest optimization for the cold path.

### Swift dispatches MLX kernels at 1.47x overhead

We use MLX's pre-compiled metallib (15,749 kernels including NAX variants) without the MLX Python runtime. The 1.47x overhead vs MLX is command buffer dispatch, not kernel execution. MLX amortizes dispatch across fused evaluation graphs; our per-layer dispatch pays the cost each time. Acceptable for streaming where weight I/O dominates.

### MLX already uses TensorOps

We audited MLX 0.31.1's Steel kernels -- they use cooperative tensors (`MetalPerformancePrimitives.h`) and X-Y swizzled dispatch. IAM's value is not faster kernels. It is running models that MLX cannot load.

## GDN (GatedDeltaNet) Implementation

48 of 64 layers use GatedDeltaNet -- a linear attention variant with persistent recurrent state. This is the first Metal implementation of GDN.

- **Conv1d state**: depthwise convolution with kernel=4, double-buffered across tokens. 10240-dim input, stored per layer.
- **Delta state**: 48 layers x 48 heads x 128x128 float32 matrices. Rank-1 update recurrence per token (`S = alpha * S + beta * k * v^T`).
- **Gated output**: sigmoid gate + linear combination, no softmax attention.

Custom Metal kernels: `gated_delta_step`, `depthwise_conv1d_step`, `sigmoid_gate_bf16`, `silu_multiply_bf16`.

Verified against MLX reference: 16/18 components PASS (out_proj at 85% cosine similarity from accumulated bf16 rounding across the full GDN pipeline; layer output recovers to 91%).

## Files

Each file is a self-contained Swift program that compiles and runs independently. The repo shows the development path from first kernel dispatch through full autoregressive generation.

**Engine:**

| File | Lines | Purpose |
|------|-------|---------|
| `full_forward.swift` | 1454 | Complete 64-layer forward pass with autoregressive generation |
| `warm_forward.swift` | -- | Warm-path with MADV_WILLNEED prefetch (2x over cold) |
| `attn_forward.swift` | 1050 | Attention layer implementation + verification (14/14 PASS) |
| `gdn_forward.swift` | 1080 | GDN layer implementation + verification (16/18 PASS) |
| `test_context.swift` | -- | Multi-token context processing |

**Kill test progression (Steps 1-5):**

| File | Step | Purpose |
|------|------|---------|
| `weight_loader.swift` | 1 | mmap + zero-copy weight loading (0.065ms per layer) |
| `kernel_dispatch.swift` | 2 | Single NAX kernel dispatch + verification vs MLX |
| `mlp_pipeline.swift` | 3 | 4-kernel MLP pipeline, 32-layer benchmark (1.47x MLX) |
| `streaming_27b.swift` | 4 | 27B streaming from mmap'd safetensors (3.3ms pre-loaded) |
| `batch_verify_27b.swift` | 5 | Batch verification plateau measurement (K=32 = K=1) |

**Experiments:**

| File | Purpose |
|------|---------|
| `challenge_plateau.swift` | GPU page fault investigation (zero-copy fails) |
| `zero_copy_27b.swift` | Zero-copy attempt that discovered page fault problem |
| `weight_loader_aligned.swift` | Page-aligned mmap variant |
| `kernel_*.swift` | Early kernel dispatch iterations (minimal, f32, setbytes, both) |

**Python utilities:**

| File | Purpose |
|------|---------|
| `save_bf16_direct.py` | Generate bf16 reference data from MLX |
| `save_mlp_reference.py` | Generate full MLP reference data |
| `verify_kernel.py` | Verify saved reference data against MLX |
| `profile_one_layer.py` | Profile layer structure and quantization |
| `tokenize_prompt.py` | Tokenize prompts for Swift consumption |

**Benchmark suite:**

| File | Purpose |
|------|---------|
| `benchmark/download_filings.py` | Download SEC EDGAR filings (15 banks) |
| `benchmark/build_gold.py` | Build gold-standard annotations for entity extraction |
| `benchmark/tokenize_all.py` | Pre-tokenize all benchmark prompts |
| `benchmark/run_benchmark.py` | Run full benchmark: SEC filing entity extraction |
| `benchmark/filings/` | Cached EDGAR Exhibit 21 filings |
| `benchmark/gold/` | Gold-standard entity annotations |

See [RESULTS.md](RESULTS.md) for detailed measurements from the kill test.

## Benchmark

15-bank SEC filing entity extraction benchmark with gold-standard annotations. Downloads real EDGAR Exhibit 21 filings, extracts subsidiary entities via the model, and scores against hand-annotated ground truth.

```bash
cd benchmark
python download_filings.py   # fetch filings from EDGAR
python build_gold.py          # build gold annotations
python tokenize_all.py        # pre-tokenize prompts
python run_benchmark.py       # run extraction + scoring
```

Results are written to `benchmark/results/`.

## Build & Run

```bash
# Requires:
#   macOS 26+ (Sequoia), Apple Silicon
#   MLX 0.31.1+ metallib (for pre-compiled NAX kernels)
#   Model weights at ~/models/Qwen3.5-27B-MLX-4bit/

# Compile
swiftc -O -framework Metal -framework MetalPerformanceShaders full_forward.swift -o full_forward

# Run (cold path — streams from SSD)
./full_forward

# Warm path (MADV_WILLNEED prefetch, 2x faster)
swiftc -O -framework Metal -framework MetalPerformanceShaders warm_forward.swift -o warm_forward
./warm_forward
```

The binary looks for:
- MLX metallib at `~/.mlx-env/lib/python3.11/site-packages/mlx/lib/mlx.metallib`
- Model weights at `~/models/Qwen3.5-27B-MLX-4bit/` (safetensors + config.json)

## What This Is Not

This is not faster than MLX for models that fit in RAM. MLX's graph compiler, fused evaluation, and lazy scheduling are well-optimized (V2.5 on Apple's TensorOps scale). On a 9B model that fits in memory, MLX at 25 tok/s is the right tool.

This is for models that don't fit. MLX OOMs on Qwen3.5-27B with 16GB. IAM streams it at 0.066-0.133 tok/s from SSD, with a clear path to 2.5-3 tok/s when weights are resident.

## Limitations

- **SSD-bound on 16GB**: 0.066-0.133 tok/s is not interactive. This is a proof that the architecture works, not a production system.
- **GDN verification gap**: 2 of 18 GDN sub-components show accumulated bf16 rounding error (85% cosine similarity on out_proj). Final layer output is 91%. Good enough for generation, not for numerical verification claims.
- **No KV cache persistence across sessions**: cache is in-memory only.
- **Single-sequence only**: no batched inference, no continuous batching.
- **Hardcoded to Qwen3.5-27B**: layer counts, dimensions, and GDN config are constants, not parsed from config.json.

## License

MIT
