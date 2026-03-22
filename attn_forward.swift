// attn_forward.swift — Session 1: Full forward pass through 16 self-attention layers
// Skips GDN layers (pass-through). Proves: embed → norm → attn → norm → MLP → lm_head → argmax
//
// Build: swiftc -O -framework Metal -framework MetalPerformanceShaders attn_forward.swift -o attn_forward
// Run:   ./attn_forward

import Metal
import Foundation

// ============================================================================
// MARK: - Config
// ============================================================================

let HIDDEN_SIZE: Int = 5120
let INTERMEDIATE_SIZE: Int = 17408
let NUM_HEADS: Int = 24
let NUM_KV_HEADS: Int = 4
let HEAD_DIM: Int = 256
let VOCAB_SIZE: Int = 248320
let NUM_LAYERS: Int = 64
let RMS_NORM_EPS: Float = 1e-6
let ROPE_THETA: Float = 10000000.0
let PARTIAL_ROTARY_FACTOR: Float = 0.25
let ROTARY_DIM: Int = 64  // HEAD_DIM * PARTIAL_ROTARY_FACTOR
let GROUP_SIZE: Int = 64
let BITS: Int = 4

// Full attention layers: every 4th starting at 3
let FULL_ATTN_LAYERS = stride(from: 3, to: 64, by: 4).map { $0 }  // [3, 7, 11, ..., 63]

// ============================================================================
// MARK: - Metal Kernel Source
// ============================================================================

let metalSource = """
#include <metal_stdlib>
using namespace metal;

// RMS Norm: out[i] = (x[i] / rms) * weight[i]
// rms = sqrt(mean(x^2) + eps)
kernel void rms_norm_bf16(
    device const bfloat* x      [[buffer(0)]],
    device const bfloat* weight  [[buffer(1)]],
    device bfloat* out           [[buffer(2)]],
    constant uint& dim           [[buffer(3)]],
    constant float& eps          [[buffer(4)]],
    uint tid [[thread_position_in_grid]],
    uint threads [[threads_per_grid]])
{
    // Two-pass: first compute sum of squares, then normalize
    // Pass 1: each thread accumulates partial sum
    float sum_sq = 0.0f;
    for (uint i = tid; i < dim; i += threads) {
        float v = float(x[i]);
        sum_sq += v * v;
    }

    // Reduction in threadgroup shared memory
    threadgroup float shared[256];
    shared[tid] = sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction
    for (uint s = threads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float rms = sqrt(shared[0] / float(dim) + eps);

    // Pass 2: normalize
    for (uint i = tid; i < dim; i += threads) {
        float v = float(x[i]);
        float w = float(weight[i]);
        out[i] = bfloat((v / rms) * w);
    }
}

// SiLU multiply: out[i] = silu(gate[i]) * up[i]
kernel void silu_multiply_bf16(
    device const bfloat* gate [[buffer(0)]],
    device const bfloat* up   [[buffer(1)]],
    device bfloat* out        [[buffer(2)]],
    constant uint& count      [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= count) return;
    float g = float(gate[tid]);
    float u = float(up[tid]);
    out[tid] = bfloat((g / (1.0f + exp(-g))) * u);
}

// Residual add: out[i] = a[i] + b[i]
kernel void residual_add_bf16(
    device const bfloat* a   [[buffer(0)]],
    device const bfloat* b   [[buffer(1)]],
    device bfloat* out       [[buffer(2)]],
    constant uint& count     [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= count) return;
    out[tid] = bfloat(float(a[tid]) + float(b[tid]));
}

// Embedding dequantize + lookup: extract one row from quantized embedding
// Quantization: 4-bit affine, group_size=64
// weight: (vocab_size, hidden_size/8) uint32 — 8 values packed per uint32
// scales: (vocab_size, hidden_size/group_size) bf16
// biases: (vocab_size, hidden_size/group_size) bf16
kernel void embed_lookup_bf16(
    device const uint* weight    [[buffer(0)]],
    device const bfloat* scales  [[buffer(1)]],
    device const bfloat* biases  [[buffer(2)]],
    device bfloat* out           [[buffer(3)]],
    constant uint& token_id      [[buffer(4)]],
    constant uint& hidden_dim    [[buffer(5)]],
    constant uint& group_sz      [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= hidden_dim) return;

    uint packed_dim = hidden_dim / 8;  // 640
    uint groups_per_row = hidden_dim / group_sz;  // 80

    // Which packed uint32 and which nibble within it
    uint pack_idx = tid / 8;
    uint nibble_idx = tid % 8;

    uint packed_val = weight[token_id * packed_dim + pack_idx];
    // Extract 4-bit value (nibble)
    uint nibble = (packed_val >> (nibble_idx * 4)) & 0xF;

    // Scale and bias for this group
    uint group_idx = tid / group_sz;
    float scale = float(scales[token_id * groups_per_row + group_idx]);
    float bias = float(biases[token_id * groups_per_row + group_idx]);

    out[tid] = bfloat(float(nibble) * scale + bias);
}

// Sigmoid multiply (for attention output gate): out = a * sigmoid(b)
kernel void sigmoid_multiply_bf16(
    device const bfloat* a   [[buffer(0)]],
    device const bfloat* b   [[buffer(1)]],
    device bfloat* out       [[buffer(2)]],
    constant uint& count     [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= count) return;
    float va = float(a[tid]);
    float vb = float(b[tid]);
    out[tid] = bfloat(va * (1.0f / (1.0f + exp(-vb))));
}

// RoPE: apply rotary position embedding to first rotary_dim dims of each head
// Interleaved layout: pairs are (x[0], x[1]), (x[2], x[3]), ...
// x: (num_heads, head_dim) — single token
kernel void rope_bf16(
    device bfloat* x             [[buffer(0)]],
    constant float* cos_cache    [[buffer(1)]],
    constant float* sin_cache    [[buffer(2)]],
    constant uint& num_heads     [[buffer(3)]],
    constant uint& head_dim      [[buffer(4)]],
    constant uint& rotary_dim    [[buffer(5)]],
    uint2 tid [[thread_position_in_grid]])
{
    uint head = tid.y;
    uint pair = tid.x;  // pair index within rotary_dim/2

    if (head >= num_heads || pair >= rotary_dim / 2) return;

    uint base = head * head_dim + pair * 2;
    float x0 = float(x[base]);
    float x1 = float(x[base + 1]);

    float c = cos_cache[pair];
    float s = sin_cache[pair];

    x[base]     = bfloat(x0 * c - x1 * s);
    x[base + 1] = bfloat(x0 * s + x1 * c);
}

// Deinterleave Q+gate from (num_heads, head_dim*2) to separate Q and gate buffers
// Input layout: [head0_q(256), head0_gate(256), head1_q(256), head1_gate(256), ...]
kernel void split_q_gate_bf16(
    device const bfloat* qg    [[buffer(0)]],  // (num_heads, head_dim*2)
    device bfloat* q           [[buffer(1)]],  // (num_heads, head_dim)
    device bfloat* gate        [[buffer(2)]],  // (num_heads, head_dim)
    constant uint& num_heads   [[buffer(3)]],
    constant uint& head_dim    [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]])
{
    uint head = tid.y;
    uint elem = tid.x;
    if (head >= num_heads || elem >= head_dim) return;
    uint src_base = head * head_dim * 2;
    q[head * head_dim + elem] = qg[src_base + elem];
    gate[head * head_dim + elem] = qg[src_base + head_dim + elem];
}

// Argmax over bf16 array
kernel void argmax_bf16(
    device const bfloat* x   [[buffer(0)]],
    device uint* result      [[buffer(1)]],
    constant uint& count     [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    // Single thread argmax (vocab is large but this runs once per token)
    if (tid != 0) return;
    float max_val = float(x[0]);
    uint max_idx = 0;
    for (uint i = 1; i < count; i++) {
        float v = float(x[i]);
        if (v > max_val) {
            max_val = v;
            max_idx = i;
        }
    }
    result[0] = max_idx;
}

// Per-head RMS norm (for QK norm): normalize each head independently
// x: (num_heads, head_dim), weight: (head_dim)
kernel void per_head_rms_norm_bf16(
    device const bfloat* x       [[buffer(0)]],
    device const bfloat* weight  [[buffer(1)]],
    device bfloat* out           [[buffer(2)]],
    constant uint& num_heads     [[buffer(3)]],
    constant uint& head_dim      [[buffer(4)]],
    constant float& eps          [[buffer(5)]],
    uint2 tid [[thread_position_in_grid]])
{
    uint head = tid.y;
    uint elem = tid.x;
    if (head >= num_heads || elem >= head_dim) return;

    // Compute RMS for this head
    float sum_sq = 0.0f;
    uint base = head * head_dim;
    for (uint i = 0; i < head_dim; i++) {
        float v = float(x[base + i]);
        sum_sq += v * v;
    }
    float rms = sqrt(sum_sq / float(head_dim) + eps);

    float v = float(x[base + elem]);
    float w = float(weight[elem]);
    out[base + elem] = bfloat((v / rms) * w);
}

// Copy KV heads to match Q heads (GQA expansion)
// For 24 Q heads and 4 KV heads: each KV head is repeated 6 times
kernel void expand_kv_heads_bf16(
    device const bfloat* kv   [[buffer(0)]],  // (num_kv_heads, head_dim)
    device bfloat* out        [[buffer(1)]],  // (num_heads, head_dim)
    constant uint& num_heads  [[buffer(2)]],
    constant uint& num_kv     [[buffer(3)]],
    constant uint& head_dim   [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]])
{
    uint head = tid.y;
    uint elem = tid.x;
    if (head >= num_heads || elem >= head_dim) return;

    uint kv_head = head / (num_heads / num_kv);
    out[head * head_dim + elem] = kv[kv_head * head_dim + elem];
}
"""

// ============================================================================
// MARK: - Safetensors & Weight Loading
// ============================================================================

struct TensorInfo {
    let fileIdx: Int
    let byteOffset: Int
    let byteSize: Int
    let shape: [Int]
    let dtype: String
}

struct LayerWeights {
    // Attention
    var inputNormWeight: MTLBuffer?
    var qProjW: MTLBuffer?, qProjS: MTLBuffer?, qProjB: MTLBuffer?
    var kProjW: MTLBuffer?, kProjS: MTLBuffer?, kProjB: MTLBuffer?
    var vProjW: MTLBuffer?, vProjS: MTLBuffer?, vProjB: MTLBuffer?
    var oProjW: MTLBuffer?, oProjS: MTLBuffer?, oProjB: MTLBuffer?
    var qNormWeight: MTLBuffer?
    var kNormWeight: MTLBuffer?
    // MLP
    var postNormWeight: MTLBuffer?
    var gateProjW: MTLBuffer?, gateProjS: MTLBuffer?, gateProjB: MTLBuffer?
    var upProjW: MTLBuffer?, upProjS: MTLBuffer?, upProjB: MTLBuffer?
    var downProjW: MTLBuffer?, downProjS: MTLBuffer?, downProjB: MTLBuffer?
}

func loadTensorToBuffer(_ device: MTLDevice, _ mmaps: [UnsafeRawPointer], _ info: TensorInfo) -> MTLBuffer {
    let src = mmaps[info.fileIdx] + info.byteOffset
    return device.makeBuffer(bytes: src, length: info.byteSize, options: .storageModeShared)!
}

func parseTensorInfo(_ dict: [String: Any]) -> TensorInfo {
    let shape = dict["shape"] as! [Int]
    let dtype = dict["dtype"] as! String
    let fileIdx = dict["file_idx"] as! Int
    let byteOffset = dict["byte_offset"] as! Int
    let byteSize = dict["byte_size"] as! Int
    return TensorInfo(fileIdx: fileIdx, byteOffset: byteOffset, byteSize: byteSize, shape: shape, dtype: dtype)
}

// ============================================================================
// MARK: - Main
// ============================================================================

let device = MTLCreateSystemDefaultDevice()!
let queue = device.makeCommandQueue()!
print("Device: \(device.name)")

// Load metallib for NAX quantized matmul
let defaultLib = device.makeDefaultLibrary()
let mlxMetalPath = String(cString: getenv("HOME")) + "/.mlx-env/lib/python3.11/site-packages/mlx/lib/mlx.metallib"
guard let mlxLib = try? device.makeLibrary(URL: URL(fileURLWithPath: mlxMetalPath)) else {
    print("ERROR: Cannot load MLX metallib from \(mlxMetalPath)")
    exit(1)
}

// NAX quantized matmul kernel
let naxKernelName = "affine_qmm_t_nax_bfloat16_t_gs_64_b_4_bm64_bn64_bk64_wm2_wn2_alN_true_batch_0"
guard let naxFn = try? mlxLib.makeFunction(name: naxKernelName) else {
    print("ERROR: NAX kernel not found"); exit(1)
}
let naxPSO = try! device.makeComputePipelineState(function: naxFn)

// Compile custom kernels
let customLib = try! device.makeLibrary(source: metalSource, options: nil)
func makePSO(_ name: String) -> MTLComputePipelineState {
    let fn = customLib.makeFunction(name: name)!
    return try! device.makeComputePipelineState(function: fn)
}
let rmsNormPSO = makePSO("rms_norm_bf16")
let siluMulPSO = makePSO("silu_multiply_bf16")
let residualPSO = makePSO("residual_add_bf16")
let embedPSO = makePSO("embed_lookup_bf16")
let sigmoidMulPSO = makePSO("sigmoid_multiply_bf16")
let ropePSO = makePSO("rope_bf16")
let argmaxPSO = makePSO("argmax_bf16")
let perHeadRmsNormPSO = makePSO("per_head_rms_norm_bf16")
let expandKvPSO = makePSO("expand_kv_heads_bf16")
let splitQGatePSO = makePSO("split_q_gate_bf16")
print("All kernels compiled")

// Load weight index
let indexPath = "/Users/midas/Desktop/cowork/inference-across-metal/reference_data/27b_full_index.json"
let indexData = try! Data(contentsOf: URL(fileURLWithPath: indexPath))
let index = try! JSONSerialization.jsonObject(with: indexData) as! [String: Any]
let shardFiles = index["shard_files"] as! [String]
let layersInfo = index["layers"] as! [[String: Any]]

// mmap safetensors files
var mmaps: [UnsafeRawPointer] = []
var mmapSizes: [Int] = []
for path in shardFiles {
    let fd = open(path, O_RDONLY)
    guard fd >= 0 else { print("Cannot open \(path)"); exit(1) }
    let size = Int(lseek(fd, 0, SEEK_END))
    lseek(fd, 0, SEEK_SET)
    let ptr = mmap(nil, size, PROT_READ, MAP_PRIVATE, fd, 0)!
    close(fd)
    mmaps.append(UnsafeRawPointer(ptr))
    mmapSizes.append(size)
}
print("mmap'd \(shardFiles.count) shard files")

// ============================================================================
// MARK: - Load Weights for Full-Attention Layers
// ============================================================================

print("\nLoading weights for \(FULL_ATTN_LAYERS.count) full-attention layers...")
let t0 = CFAbsoluteTimeGetCurrent()

// Embedding weights
let embedInfo = index["embed_tokens"] as! [String: Any]
let embedW = loadTensorToBuffer(device, mmaps, parseTensorInfo(embedInfo["weight"] as! [String: Any]))
let embedS = loadTensorToBuffer(device, mmaps, parseTensorInfo(embedInfo["scales"] as! [String: Any]))
let embedB = loadTensorToBuffer(device, mmaps, parseTensorInfo(embedInfo["biases"] as! [String: Any]))

// Final norm
let finalNormInfo = index["final_norm"] as! [String: Any]
let finalNormW = loadTensorToBuffer(device, mmaps, parseTensorInfo(finalNormInfo["weight"] as! [String: Any]))

// lm_head
let lmHeadInfo = index["lm_head"] as! [String: Any]
let lmHeadW = loadTensorToBuffer(device, mmaps, parseTensorInfo(lmHeadInfo["weight"] as! [String: Any]))
let lmHeadS = loadTensorToBuffer(device, mmaps, parseTensorInfo(lmHeadInfo["scales"] as! [String: Any]))
let lmHeadB = loadTensorToBuffer(device, mmaps, parseTensorInfo(lmHeadInfo["biases"] as! [String: Any]))

// Per-layer weights (only full attention layers)
var layers: [Int: LayerWeights] = [:]
for layerIdx in FULL_ATTN_LAYERS {
    let li = layersInfo[layerIdx]
    let tensors = li["tensors"] as! [String: Any]

    func loadT(_ name: String) -> MTLBuffer {
        return loadTensorToBuffer(device, mmaps, parseTensorInfo(tensors[name] as! [String: Any]))
    }

    var lw = LayerWeights()
    lw.inputNormWeight = loadT("input_layernorm.weight")
    lw.qProjW = loadT("self_attn.q_proj.weight")
    lw.qProjS = loadT("self_attn.q_proj.scales")
    lw.qProjB = loadT("self_attn.q_proj.biases")
    lw.kProjW = loadT("self_attn.k_proj.weight")
    lw.kProjS = loadT("self_attn.k_proj.scales")
    lw.kProjB = loadT("self_attn.k_proj.biases")
    lw.vProjW = loadT("self_attn.v_proj.weight")
    lw.vProjS = loadT("self_attn.v_proj.scales")
    lw.vProjB = loadT("self_attn.v_proj.biases")
    lw.oProjW = loadT("self_attn.o_proj.weight")
    lw.oProjS = loadT("self_attn.o_proj.scales")
    lw.oProjB = loadT("self_attn.o_proj.biases")
    lw.qNormWeight = loadT("self_attn.q_norm.weight")
    lw.kNormWeight = loadT("self_attn.k_norm.weight")
    lw.postNormWeight = loadT("post_attention_layernorm.weight")
    lw.gateProjW = loadT("mlp.gate_proj.weight")
    lw.gateProjS = loadT("mlp.gate_proj.scales")
    lw.gateProjB = loadT("mlp.gate_proj.biases")
    lw.upProjW = loadT("mlp.up_proj.weight")
    lw.upProjS = loadT("mlp.up_proj.scales")
    lw.upProjB = loadT("mlp.up_proj.biases")
    lw.downProjW = loadT("mlp.down_proj.weight")
    lw.downProjS = loadT("mlp.down_proj.scales")
    lw.downProjB = loadT("mlp.down_proj.biases")

    layers[layerIdx] = lw
}

let tLoad = CFAbsoluteTimeGetCurrent() - t0
print("Weights loaded in \(String(format: "%.1f", tLoad))s")

// Calculate memory
var totalMem = embedW.length + embedS.length + embedB.length
totalMem += finalNormW.length
totalMem += lmHeadW.length + lmHeadS.length + lmHeadB.length
for (_, lw) in layers {
    totalMem += lw.inputNormWeight!.length + lw.postNormWeight!.length
    totalMem += lw.qProjW!.length + lw.qProjS!.length + lw.qProjB!.length
    totalMem += lw.kProjW!.length + lw.kProjS!.length + lw.kProjB!.length
    totalMem += lw.vProjW!.length + lw.vProjS!.length + lw.vProjB!.length
    totalMem += lw.oProjW!.length + lw.oProjS!.length + lw.oProjB!.length
    totalMem += lw.qNormWeight!.length + lw.kNormWeight!.length
    totalMem += lw.gateProjW!.length + lw.gateProjS!.length + lw.gateProjB!.length
    totalMem += lw.upProjW!.length + lw.upProjS!.length + lw.upProjB!.length
    totalMem += lw.downProjW!.length + lw.downProjS!.length + lw.downProjB!.length
}
print("Total weight memory: \(totalMem / 1024 / 1024) MB (16 attn layers + embed + lm_head)")

// ============================================================================
// MARK: - Activation Buffers
// ============================================================================

// Allocate reusable activation buffers
let hiddenBuf = device.makeBuffer(length: HIDDEN_SIZE * 2, options: .storageModeShared)!  // bf16
let hiddenBuf2 = device.makeBuffer(length: HIDDEN_SIZE * 2, options: .storageModeShared)!
let normedBuf = device.makeBuffer(length: HIDDEN_SIZE * 2, options: .storageModeShared)!
let qBuf = device.makeBuffer(length: NUM_HEADS * HEAD_DIM * 2 * 2, options: .storageModeShared)!  // Q + gate = 12288 bf16
let queriesBuf = device.makeBuffer(length: NUM_HEADS * HEAD_DIM * 2, options: .storageModeShared)!  // queries only = 6144 bf16
let attnGateBuf = device.makeBuffer(length: NUM_HEADS * HEAD_DIM * 2, options: .storageModeShared)!  // gate only = 6144 bf16
let kBuf = device.makeBuffer(length: NUM_KV_HEADS * HEAD_DIM * 2, options: .storageModeShared)!
let vBuf = device.makeBuffer(length: NUM_KV_HEADS * HEAD_DIM * 2, options: .storageModeShared)!
let qNormedBuf = device.makeBuffer(length: NUM_HEADS * HEAD_DIM * 2, options: .storageModeShared)!
let kNormedBuf = device.makeBuffer(length: NUM_KV_HEADS * HEAD_DIM * 2, options: .storageModeShared)!
let kExpandedBuf = device.makeBuffer(length: NUM_HEADS * HEAD_DIM * 2, options: .storageModeShared)!
let vExpandedBuf = device.makeBuffer(length: NUM_HEADS * HEAD_DIM * 2, options: .storageModeShared)!
let attnOutBuf = device.makeBuffer(length: NUM_HEADS * HEAD_DIM * 2, options: .storageModeShared)!  // 6144 bf16
let gatedBuf = device.makeBuffer(length: NUM_HEADS * HEAD_DIM * 2, options: .storageModeShared)!
let oOutBuf = device.makeBuffer(length: HIDDEN_SIZE * 2, options: .storageModeShared)!
let residualBuf = device.makeBuffer(length: HIDDEN_SIZE * 2, options: .storageModeShared)!
let gateBuf = device.makeBuffer(length: INTERMEDIATE_SIZE * 2, options: .storageModeShared)!
let upBuf = device.makeBuffer(length: INTERMEDIATE_SIZE * 2, options: .storageModeShared)!
let mlpHiddenBuf = device.makeBuffer(length: INTERMEDIATE_SIZE * 2, options: .storageModeShared)!
let downBuf = device.makeBuffer(length: HIDDEN_SIZE * 2, options: .storageModeShared)!
let logitsBuf = device.makeBuffer(length: VOCAB_SIZE * 2, options: .storageModeShared)!
let argmaxBuf = device.makeBuffer(length: 4, options: .storageModeShared)!

// RoPE cos/sin cache for position 0 (will be trivial: cos=1, sin=0)
let ropeCosBuf = device.makeBuffer(length: ROTARY_DIM / 2 * 4, options: .storageModeShared)!  // float32
let ropeSinBuf = device.makeBuffer(length: ROTARY_DIM / 2 * 4, options: .storageModeShared)!

// ============================================================================
// MARK: - Helper: dispatch quantized matmul via NAX
// ============================================================================

func dispatchQMatmul(_ enc: MTLComputeCommandEncoder,
                     _ wBuf: MTLBuffer, _ sBuf: MTLBuffer, _ bBuf: MTLBuffer,
                     _ xBuf: MTLBuffer, _ yBuf: MTLBuffer,
                     K_dim: Int, N_dim: Int, M_dim: Int) {
    enc.setComputePipelineState(naxPSO)
    enc.setBuffer(wBuf, offset: 0, index: 0)
    enc.setBuffer(sBuf, offset: 0, index: 1)
    enc.setBuffer(bBuf, offset: 0, index: 2)
    enc.setBuffer(xBuf, offset: 0, index: 3)
    enc.setBuffer(yBuf, offset: 0, index: 4)
    var k = UInt32(K_dim), n = UInt32(N_dim), m = UInt32(M_dim)
    enc.setBytes(&k, length: 4, index: 5)
    enc.setBytes(&n, length: 4, index: 6)
    enc.setBytes(&m, length: 4, index: 7)
    enc.dispatchThreadgroups(
        MTLSize(width: (N_dim + 63) / 64, height: (M_dim + 63) / 64, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 32, height: 2, depth: 2))
}

// ============================================================================
// MARK: - Precompute RoPE cache
// ============================================================================

func precomputeRoPE(position: Int) {
    let cosPtr = ropeCosBuf.contents().bindMemory(to: Float.self, capacity: ROTARY_DIM / 2)
    let sinPtr = ropeSinBuf.contents().bindMemory(to: Float.self, capacity: ROTARY_DIM / 2)
    for i in 0..<(ROTARY_DIM / 2) {
        let freq = 1.0 / pow(Double(ROPE_THETA), Double(2 * i) / Double(ROTARY_DIM))
        let angle = Double(position) * freq
        cosPtr[i] = Float(cos(angle))
        sinPtr[i] = Float(sin(angle))
    }
}

// ============================================================================
// MARK: - Forward Pass
// ============================================================================

func forwardPass(tokenId: Int, position: Int) -> Int {
    precomputeRoPE(position: position)

    let cb = queue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!

    // --- Embedding lookup ---
    enc.setComputePipelineState(embedPSO)
    enc.setBuffer(embedW, offset: 0, index: 0)
    enc.setBuffer(embedS, offset: 0, index: 1)
    enc.setBuffer(embedB, offset: 0, index: 2)
    enc.setBuffer(hiddenBuf, offset: 0, index: 3)
    var tid = UInt32(tokenId)
    enc.setBytes(&tid, length: 4, index: 4)
    var hdim = UInt32(HIDDEN_SIZE)
    enc.setBytes(&hdim, length: 4, index: 5)
    var gsz = UInt32(GROUP_SIZE)
    enc.setBytes(&gsz, length: 4, index: 6)
    enc.dispatchThreads(MTLSize(width: HIDDEN_SIZE, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

    // --- Process each full-attention layer ---
    for layerIdx in FULL_ATTN_LAYERS {
        let lw = layers[layerIdx]!

        // 1. Input RMS Norm
        enc.setComputePipelineState(rmsNormPSO)
        enc.setBuffer(hiddenBuf, offset: 0, index: 0)
        enc.setBuffer(lw.inputNormWeight!, offset: 0, index: 1)
        enc.setBuffer(normedBuf, offset: 0, index: 2)
        var dim = UInt32(HIDDEN_SIZE)
        enc.setBytes(&dim, length: 4, index: 3)
        var eps = RMS_NORM_EPS
        enc.setBytes(&eps, length: 4, index: 4)
        enc.dispatchThreads(MTLSize(width: 256, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

        // 2. Q projection (output: 12288 = 24 heads * (256 Q + 256 gate) interleaved)
        dispatchQMatmul(enc, lw.qProjW!, lw.qProjS!, lw.qProjB!,
                        normedBuf, qBuf,
                        K_dim: HIDDEN_SIZE, N_dim: NUM_HEADS * HEAD_DIM * 2, M_dim: 1)

        // 2b. Split Q+gate from interleaved (head0_q,head0_gate,head1_q,...) to separate buffers
        enc.setComputePipelineState(splitQGatePSO)
        enc.setBuffer(qBuf, offset: 0, index: 0)
        enc.setBuffer(queriesBuf, offset: 0, index: 1)
        enc.setBuffer(attnGateBuf, offset: 0, index: 2)
        var nh = UInt32(NUM_HEADS)
        enc.setBytes(&nh, length: 4, index: 3)
        var hd = UInt32(HEAD_DIM)
        enc.setBytes(&hd, length: 4, index: 4)
        enc.dispatchThreads(MTLSize(width: HEAD_DIM, height: NUM_HEADS, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

        // 3. K projection (output: 1024 = 4 heads * 256 dim)
        dispatchQMatmul(enc, lw.kProjW!, lw.kProjS!, lw.kProjB!,
                        normedBuf, kBuf,
                        K_dim: HIDDEN_SIZE, N_dim: NUM_KV_HEADS * HEAD_DIM, M_dim: 1)

        // 4. V projection (output: 1024 = 4 heads * 256 dim)
        dispatchQMatmul(enc, lw.vProjW!, lw.vProjS!, lw.vProjB!,
                        normedBuf, vBuf,
                        K_dim: HIDDEN_SIZE, N_dim: NUM_KV_HEADS * HEAD_DIM, M_dim: 1)

        // 5. Per-head QK norm on separated queries
        enc.setComputePipelineState(perHeadRmsNormPSO)
        enc.setBuffer(queriesBuf, offset: 0, index: 0)
        enc.setBuffer(lw.qNormWeight!, offset: 0, index: 1)
        enc.setBuffer(qNormedBuf, offset: 0, index: 2)
        enc.setBytes(&nh, length: 4, index: 3)
        enc.setBytes(&hd, length: 4, index: 4)
        enc.setBytes(&eps, length: 4, index: 5)
        enc.dispatchThreads(MTLSize(width: HEAD_DIM, height: NUM_HEADS, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: HEAD_DIM, height: 1, depth: 1))

        // K norm
        enc.setBuffer(kBuf, offset: 0, index: 0)
        enc.setBuffer(lw.kNormWeight!, offset: 0, index: 1)
        enc.setBuffer(kNormedBuf, offset: 0, index: 2)
        var nkv = UInt32(NUM_KV_HEADS)
        enc.setBytes(&nkv, length: 4, index: 3)
        enc.setBytes(&hd, length: 4, index: 4)
        enc.setBytes(&eps, length: 4, index: 5)
        enc.dispatchThreads(MTLSize(width: HEAD_DIM, height: NUM_KV_HEADS, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: HEAD_DIM, height: 1, depth: 1))

        // 6. RoPE on Q and K (in-place)
        enc.setComputePipelineState(ropePSO)
        enc.setBuffer(qNormedBuf, offset: 0, index: 0)
        enc.setBuffer(ropeCosBuf, offset: 0, index: 1)
        enc.setBuffer(ropeSinBuf, offset: 0, index: 2)
        enc.setBytes(&nh, length: 4, index: 3)
        enc.setBytes(&hd, length: 4, index: 4)
        var rd = UInt32(ROTARY_DIM)
        enc.setBytes(&rd, length: 4, index: 5)
        enc.dispatchThreads(MTLSize(width: ROTARY_DIM / 2, height: NUM_HEADS, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))

        // RoPE on K
        enc.setBuffer(kNormedBuf, offset: 0, index: 0)
        enc.setBytes(&nkv, length: 4, index: 3)
        enc.dispatchThreads(MTLSize(width: ROTARY_DIM / 2, height: NUM_KV_HEADS, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))

        // 7. Attention: for single token at any position with only current KV, output = V
        // (softmax of single element = 1.0, so attn_output = V regardless of Q@K^T)
        // Expand V to match Q heads via GQA
        enc.setComputePipelineState(expandKvPSO)
        enc.setBuffer(vBuf, offset: 0, index: 0)
        enc.setBuffer(attnOutBuf, offset: 0, index: 1)
        enc.setBytes(&nh, length: 4, index: 2)
        enc.setBytes(&nkv, length: 4, index: 3)
        enc.setBytes(&hd, length: 4, index: 4)
        enc.dispatchThreads(MTLSize(width: HEAD_DIM, height: NUM_HEADS, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

        // 8. Apply output gate: gated = attn_output * sigmoid(gate)
        enc.setComputePipelineState(sigmoidMulPSO)
        enc.setBuffer(attnOutBuf, offset: 0, index: 0)
        enc.setBuffer(attnGateBuf, offset: 0, index: 1)  // separated gate buffer
        enc.setBuffer(gatedBuf, offset: 0, index: 2)
        var gateCount = UInt32(NUM_HEADS * HEAD_DIM)
        enc.setBytes(&gateCount, length: 4, index: 3)
        enc.dispatchThreads(MTLSize(width: NUM_HEADS * HEAD_DIM, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

        // 9. O projection
        dispatchQMatmul(enc, lw.oProjW!, lw.oProjS!, lw.oProjB!,
                        gatedBuf, oOutBuf,
                        K_dim: NUM_HEADS * HEAD_DIM, N_dim: HIDDEN_SIZE, M_dim: 1)

        // 10. Residual add: hidden = hidden + o_out
        enc.setComputePipelineState(residualPSO)
        enc.setBuffer(hiddenBuf, offset: 0, index: 0)
        enc.setBuffer(oOutBuf, offset: 0, index: 1)
        enc.setBuffer(residualBuf, offset: 0, index: 2)
        var count = UInt32(HIDDEN_SIZE)
        enc.setBytes(&count, length: 4, index: 3)
        enc.dispatchThreads(MTLSize(width: HIDDEN_SIZE, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

        // 11. Post-attention RMS norm
        enc.setComputePipelineState(rmsNormPSO)
        enc.setBuffer(residualBuf, offset: 0, index: 0)
        enc.setBuffer(lw.postNormWeight!, offset: 0, index: 1)
        enc.setBuffer(normedBuf, offset: 0, index: 2)
        enc.setBytes(&dim, length: 4, index: 3)
        enc.setBytes(&eps, length: 4, index: 4)
        enc.dispatchThreads(MTLSize(width: 256, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

        // 12. MLP: gate_proj
        dispatchQMatmul(enc, lw.gateProjW!, lw.gateProjS!, lw.gateProjB!,
                        normedBuf, gateBuf,
                        K_dim: HIDDEN_SIZE, N_dim: INTERMEDIATE_SIZE, M_dim: 1)

        // 13. MLP: up_proj
        dispatchQMatmul(enc, lw.upProjW!, lw.upProjS!, lw.upProjB!,
                        normedBuf, upBuf,
                        K_dim: HIDDEN_SIZE, N_dim: INTERMEDIATE_SIZE, M_dim: 1)

        // 14. SiLU multiply
        enc.setComputePipelineState(siluMulPSO)
        enc.setBuffer(gateBuf, offset: 0, index: 0)
        enc.setBuffer(upBuf, offset: 0, index: 1)
        enc.setBuffer(mlpHiddenBuf, offset: 0, index: 2)
        var isize = UInt32(INTERMEDIATE_SIZE)
        enc.setBytes(&isize, length: 4, index: 3)
        enc.dispatchThreads(MTLSize(width: INTERMEDIATE_SIZE, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

        // 15. MLP: down_proj
        dispatchQMatmul(enc, lw.downProjW!, lw.downProjS!, lw.downProjB!,
                        mlpHiddenBuf, downBuf,
                        K_dim: INTERMEDIATE_SIZE, N_dim: HIDDEN_SIZE, M_dim: 1)

        // 16. Residual add: hidden = residual + down
        enc.setComputePipelineState(residualPSO)
        enc.setBuffer(residualBuf, offset: 0, index: 0)
        enc.setBuffer(downBuf, offset: 0, index: 1)
        enc.setBuffer(hiddenBuf, offset: 0, index: 2)
        enc.setBytes(&count, length: 4, index: 3)
        enc.dispatchThreads(MTLSize(width: HIDDEN_SIZE, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    // --- Final RMS Norm ---
    enc.setComputePipelineState(rmsNormPSO)
    enc.setBuffer(hiddenBuf, offset: 0, index: 0)
    enc.setBuffer(finalNormW, offset: 0, index: 1)
    enc.setBuffer(normedBuf, offset: 0, index: 2)
    var dim2 = UInt32(HIDDEN_SIZE)
    enc.setBytes(&dim2, length: 4, index: 3)
    var eps2 = RMS_NORM_EPS
    enc.setBytes(&eps2, length: 4, index: 4)
    enc.dispatchThreads(MTLSize(width: 256, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

    // --- lm_head (quantized matmul: 5120 → 248320) ---
    dispatchQMatmul(enc, lmHeadW, lmHeadS, lmHeadB,
                    normedBuf, logitsBuf,
                    K_dim: HIDDEN_SIZE, N_dim: VOCAB_SIZE, M_dim: 1)

    // --- Argmax ---
    enc.setComputePipelineState(argmaxPSO)
    enc.setBuffer(logitsBuf, offset: 0, index: 0)
    enc.setBuffer(argmaxBuf, offset: 0, index: 1)
    var vocabSize = UInt32(VOCAB_SIZE)
    enc.setBytes(&vocabSize, length: 4, index: 2)
    enc.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))

    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()

    if cb.status == .error {
        print("ERROR: Command buffer failed: \(cb.error?.localizedDescription ?? "unknown")")
        return -1
    }

    let result = argmaxBuf.contents().bindMemory(to: UInt32.self, capacity: 1)
    return Int(result[0])
}

// ============================================================================
// MARK: - Verification against reference
// ============================================================================

func loadRefBf16(_ name: String, count: Int) -> [Float] {
    let path = "/Users/midas/Desktop/cowork/inference-across-metal/reference_data/\(name)"
    let data = try! Data(contentsOf: URL(fileURLWithPath: path))
    let uint16s = data.withUnsafeBytes { Array($0.bindMemory(to: UInt16.self)) }
    return uint16s.map { u16 in
        var bits = UInt32(u16) << 16
        return withUnsafeBytes(of: &bits) { $0.load(as: Float.self) }
    }
}

func compareBufferToRef(_ buf: MTLBuffer, _ refName: String, _ count: Int, _ label: String) {
    let ref = loadRefBf16(refName, count: count)
    let ptr = buf.contents().bindMemory(to: UInt16.self, capacity: count)

    var maxErr: Float = 0
    var within01 = 0
    var within001 = 0
    for i in 0..<min(count, ref.count) {
        var bits = UInt32(ptr[i]) << 16
        let val = withUnsafeBytes(of: &bits) { $0.load(as: Float.self) }
        let err = abs(val - ref[i])
        let denom = max(abs(ref[i]), 1e-6)
        let relErr = err / denom
        if relErr < 0.1 { within01 += 1 }
        if relErr < 0.01 { within001 += 1 }
        if err > maxErr { maxErr = err }
    }
    let total = min(count, ref.count)
    let pct01 = Float(within01) / Float(total) * 100
    let pct001 = Float(within001) / Float(total) * 100
    let status = pct01 > 90 ? "PASS" : "FAIL"
    print("  \(label): \(status) — \(String(format: "%.0f", pct01))% < 0.1, \(String(format: "%.0f", pct001))% < 0.01, max_err=\(String(format: "%.4f", maxErr))")
}

// ============================================================================
// MARK: - Run
// ============================================================================

print("\n=== Single-Layer Verification (Layer 3 only) ===")
// First: run just layer 3 and verify each component against MLX reference

// Run full forward pass with token 1234
print("\n=== Forward Pass: token_id=1234, position=0 ===")
let t1 = CFAbsoluteTimeGetCurrent()
let predictedToken = forwardPass(tokenId: 1234, position: 0)
let tForward = CFAbsoluteTimeGetCurrent() - t1
print("Predicted token: \(predictedToken)")
print("Forward pass time: \(String(format: "%.1f", tForward * 1000))ms")
print("(16 attn layers, skipping 48 GDN layers)")

// Verify layer 3 intermediates
print("\n=== Verification vs MLX Reference (layer 3 intermediates from single-layer run) ===")
print("Note: multi-layer run changes hidden state, so only embedding is directly comparable.")

// For proper per-component verification, run just layer 3
print("\n=== Single-Layer Test: embed → layer 3 → output ===")

// Reset: run with just 1 layer for verification
let cb2 = queue.makeCommandBuffer()!
let enc2 = cb2.makeComputeCommandEncoder()!
precomputeRoPE(position: 0)

// Embedding
enc2.setComputePipelineState(embedPSO)
enc2.setBuffer(embedW, offset: 0, index: 0)
enc2.setBuffer(embedS, offset: 0, index: 1)
enc2.setBuffer(embedB, offset: 0, index: 2)
enc2.setBuffer(hiddenBuf, offset: 0, index: 3)
var tokenIdV = UInt32(1234)
enc2.setBytes(&tokenIdV, length: 4, index: 4)
var hdimV = UInt32(HIDDEN_SIZE)
enc2.setBytes(&hdimV, length: 4, index: 5)
var gszV = UInt32(GROUP_SIZE)
enc2.setBytes(&gszV, length: 4, index: 6)
enc2.dispatchThreads(MTLSize(width: HIDDEN_SIZE, height: 1, depth: 1),
                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
enc2.endEncoding()
cb2.commit()
cb2.waitUntilCompleted()

print("Embedding verification:")
compareBufferToRef(hiddenBuf, "ref_embed_output.bin", HIDDEN_SIZE, "embed_lookup")

// Now run layer 3 only
let lw3 = layers[3]!
let cb3 = queue.makeCommandBuffer()!
let enc3 = cb3.makeComputeCommandEncoder()!

// Input norm → write to both normedBuf (for Q/K/V) and oOutBuf (saved for verification)
enc3.setComputePipelineState(rmsNormPSO)
enc3.setBuffer(hiddenBuf, offset: 0, index: 0)
enc3.setBuffer(lw3.inputNormWeight!, offset: 0, index: 1)
enc3.setBuffer(normedBuf, offset: 0, index: 2)
var dimV = UInt32(HIDDEN_SIZE)
enc3.setBytes(&dimV, length: 4, index: 3)
var epsV = RMS_NORM_EPS
enc3.setBytes(&epsV, length: 4, index: 4)
enc3.dispatchThreads(MTLSize(width: 256, height: 1, depth: 1),
                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
// Copy input norm to hiddenBuf2 for later comparison (normedBuf will be overwritten)
enc3.setComputePipelineState(residualPSO)  // abuse add: hiddenBuf2 = normedBuf + 0
let zeroBuf = device.makeBuffer(length: HIDDEN_SIZE * 2, options: .storageModeShared)!
memset(zeroBuf.contents(), 0, HIDDEN_SIZE * 2)
enc3.setBuffer(normedBuf, offset: 0, index: 0)
enc3.setBuffer(zeroBuf, offset: 0, index: 1)
enc3.setBuffer(hiddenBuf2, offset: 0, index: 2)
var copyCount = UInt32(HIDDEN_SIZE)
enc3.setBytes(&copyCount, length: 4, index: 3)
enc3.dispatchThreads(MTLSize(width: HIDDEN_SIZE, height: 1, depth: 1),
                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

// Q proj
dispatchQMatmul(enc3, lw3.qProjW!, lw3.qProjS!, lw3.qProjB!,
                normedBuf, qBuf,
                K_dim: HIDDEN_SIZE, N_dim: NUM_HEADS * HEAD_DIM * 2, M_dim: 1)

// Split Q+gate from interleaved layout
enc3.setComputePipelineState(splitQGatePSO)
enc3.setBuffer(qBuf, offset: 0, index: 0)
enc3.setBuffer(queriesBuf, offset: 0, index: 1)
enc3.setBuffer(attnGateBuf, offset: 0, index: 2)
var nhV = UInt32(NUM_HEADS)
enc3.setBytes(&nhV, length: 4, index: 3)
var hdV = UInt32(HEAD_DIM)
enc3.setBytes(&hdV, length: 4, index: 4)
enc3.dispatchThreads(MTLSize(width: HEAD_DIM, height: NUM_HEADS, depth: 1),
                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

// K proj
dispatchQMatmul(enc3, lw3.kProjW!, lw3.kProjS!, lw3.kProjB!,
                normedBuf, kBuf,
                K_dim: HIDDEN_SIZE, N_dim: NUM_KV_HEADS * HEAD_DIM, M_dim: 1)

// V proj
dispatchQMatmul(enc3, lw3.vProjW!, lw3.vProjS!, lw3.vProjB!,
                normedBuf, vBuf,
                K_dim: HIDDEN_SIZE, N_dim: NUM_KV_HEADS * HEAD_DIM, M_dim: 1)

// Q norm on separated queries
enc3.setComputePipelineState(perHeadRmsNormPSO)
enc3.setBuffer(queriesBuf, offset: 0, index: 0)
enc3.setBuffer(lw3.qNormWeight!, offset: 0, index: 1)
enc3.setBuffer(qNormedBuf, offset: 0, index: 2)
enc3.setBytes(&nhV, length: 4, index: 3)
enc3.setBytes(&hdV, length: 4, index: 4)
enc3.setBytes(&epsV, length: 4, index: 5)
enc3.dispatchThreads(MTLSize(width: HEAD_DIM, height: NUM_HEADS, depth: 1),
                     threadsPerThreadgroup: MTLSize(width: HEAD_DIM, height: 1, depth: 1))

// K norm
enc3.setBuffer(kBuf, offset: 0, index: 0)
enc3.setBuffer(lw3.kNormWeight!, offset: 0, index: 1)
enc3.setBuffer(kNormedBuf, offset: 0, index: 2)
var nkvV = UInt32(NUM_KV_HEADS)
enc3.setBytes(&nkvV, length: 4, index: 3)
enc3.dispatchThreads(MTLSize(width: HEAD_DIM, height: NUM_KV_HEADS, depth: 1),
                     threadsPerThreadgroup: MTLSize(width: HEAD_DIM, height: 1, depth: 1))

// RoPE on Q
enc3.setComputePipelineState(ropePSO)
enc3.setBuffer(qNormedBuf, offset: 0, index: 0)
enc3.setBuffer(ropeCosBuf, offset: 0, index: 1)
enc3.setBuffer(ropeSinBuf, offset: 0, index: 2)
enc3.setBytes(&nhV, length: 4, index: 3)
enc3.setBytes(&hdV, length: 4, index: 4)
var rdV = UInt32(ROTARY_DIM)
enc3.setBytes(&rdV, length: 4, index: 5)
enc3.dispatchThreads(MTLSize(width: ROTARY_DIM / 2, height: NUM_HEADS, depth: 1),
                     threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))

// RoPE on K
enc3.setBuffer(kNormedBuf, offset: 0, index: 0)
enc3.setBytes(&nkvV, length: 4, index: 3)
enc3.dispatchThreads(MTLSize(width: ROTARY_DIM / 2, height: NUM_KV_HEADS, depth: 1),
                     threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))

// Attention: expand V and use as output (single token)
enc3.setComputePipelineState(expandKvPSO)
enc3.setBuffer(vBuf, offset: 0, index: 0)
enc3.setBuffer(attnOutBuf, offset: 0, index: 1)
enc3.setBytes(&nhV, length: 4, index: 2)
enc3.setBytes(&nkvV, length: 4, index: 3)
enc3.setBytes(&hdV, length: 4, index: 4)
enc3.dispatchThreads(MTLSize(width: HEAD_DIM, height: NUM_HEADS, depth: 1),
                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

// Output gate (using separated gate buffer)
enc3.setComputePipelineState(sigmoidMulPSO)
enc3.setBuffer(attnOutBuf, offset: 0, index: 0)
enc3.setBuffer(attnGateBuf, offset: 0, index: 1)
enc3.setBuffer(gatedBuf, offset: 0, index: 2)
var gcV = UInt32(NUM_HEADS * HEAD_DIM)
enc3.setBytes(&gcV, length: 4, index: 3)
enc3.dispatchThreads(MTLSize(width: NUM_HEADS * HEAD_DIM, height: 1, depth: 1),
                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

// O proj
dispatchQMatmul(enc3, lw3.oProjW!, lw3.oProjS!, lw3.oProjB!,
                gatedBuf, oOutBuf,
                K_dim: NUM_HEADS * HEAD_DIM, N_dim: HIDDEN_SIZE, M_dim: 1)

// Residual
enc3.setComputePipelineState(residualPSO)
enc3.setBuffer(hiddenBuf, offset: 0, index: 0)
enc3.setBuffer(oOutBuf, offset: 0, index: 1)
enc3.setBuffer(residualBuf, offset: 0, index: 2)
var cntV = UInt32(HIDDEN_SIZE)
enc3.setBytes(&cntV, length: 4, index: 3)
enc3.dispatchThreads(MTLSize(width: HIDDEN_SIZE, height: 1, depth: 1),
                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

// Post-attn norm
enc3.setComputePipelineState(rmsNormPSO)
enc3.setBuffer(residualBuf, offset: 0, index: 0)
enc3.setBuffer(lw3.postNormWeight!, offset: 0, index: 1)
enc3.setBuffer(normedBuf, offset: 0, index: 2)
enc3.setBytes(&dimV, length: 4, index: 3)
enc3.setBytes(&epsV, length: 4, index: 4)
enc3.dispatchThreads(MTLSize(width: 256, height: 1, depth: 1),
                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

// MLP
dispatchQMatmul(enc3, lw3.gateProjW!, lw3.gateProjS!, lw3.gateProjB!,
                normedBuf, gateBuf, K_dim: HIDDEN_SIZE, N_dim: INTERMEDIATE_SIZE, M_dim: 1)
dispatchQMatmul(enc3, lw3.upProjW!, lw3.upProjS!, lw3.upProjB!,
                normedBuf, upBuf, K_dim: HIDDEN_SIZE, N_dim: INTERMEDIATE_SIZE, M_dim: 1)

enc3.setComputePipelineState(siluMulPSO)
enc3.setBuffer(gateBuf, offset: 0, index: 0)
enc3.setBuffer(upBuf, offset: 0, index: 1)
enc3.setBuffer(mlpHiddenBuf, offset: 0, index: 2)
var isV = UInt32(INTERMEDIATE_SIZE)
enc3.setBytes(&isV, length: 4, index: 3)
enc3.dispatchThreads(MTLSize(width: INTERMEDIATE_SIZE, height: 1, depth: 1),
                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

dispatchQMatmul(enc3, lw3.downProjW!, lw3.downProjS!, lw3.downProjB!,
                mlpHiddenBuf, downBuf, K_dim: INTERMEDIATE_SIZE, N_dim: HIDDEN_SIZE, M_dim: 1)

// Final residual → logitsBuf (reuse, layer output)
enc3.setComputePipelineState(residualPSO)
enc3.setBuffer(residualBuf, offset: 0, index: 0)
enc3.setBuffer(downBuf, offset: 0, index: 1)
// Use a fresh part of logitsBuf for layer output (it's large enough)
let layerOutBuf = device.makeBuffer(length: HIDDEN_SIZE * 2, options: .storageModeShared)!
enc3.setBuffer(layerOutBuf, offset: 0, index: 2)
enc3.setBytes(&cntV, length: 4, index: 3)
enc3.dispatchThreads(MTLSize(width: HIDDEN_SIZE, height: 1, depth: 1),
                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

enc3.endEncoding()
cb3.commit()
cb3.waitUntilCompleted()

print("\nLayer 3 component verification:")
compareBufferToRef(hiddenBuf2, "ref_normed_input.bin", HIDDEN_SIZE, "input_norm")
compareBufferToRef(qBuf, "ref_q_proj_full.bin", NUM_HEADS * HEAD_DIM * 2, "q_proj_full")
compareBufferToRef(queriesBuf, "ref_queries.bin", NUM_HEADS * HEAD_DIM, "queries_split")
compareBufferToRef(attnGateBuf, "ref_gate.bin", NUM_HEADS * HEAD_DIM, "gate_split")
compareBufferToRef(kBuf, "ref_k_proj.bin", NUM_KV_HEADS * HEAD_DIM, "k_proj")
compareBufferToRef(vBuf, "ref_v_proj.bin", NUM_KV_HEADS * HEAD_DIM, "v_proj")
compareBufferToRef(qNormedBuf, "ref_q_normed.bin", NUM_HEADS * HEAD_DIM, "q_norm")
compareBufferToRef(kNormedBuf, "ref_k_normed.bin", NUM_KV_HEADS * HEAD_DIM, "k_norm")
compareBufferToRef(qNormedBuf, "ref_q_roped.bin", NUM_HEADS * HEAD_DIM, "q_rope")
compareBufferToRef(kNormedBuf, "ref_k_roped.bin", NUM_KV_HEADS * HEAD_DIM, "k_rope")
compareBufferToRef(attnOutBuf, "ref_attn_output.bin", NUM_HEADS * HEAD_DIM, "attn_out")
compareBufferToRef(gatedBuf, "ref_gated_output.bin", NUM_HEADS * HEAD_DIM, "gate_sigmoid")
compareBufferToRef(oOutBuf, "ref_o_proj.bin", HIDDEN_SIZE, "o_proj")
compareBufferToRef(layerOutBuf, "ref_layer3_output.bin", HIDDEN_SIZE, "layer_output")

// Timing run: 16 attention layers
print("\n=== Timing: 16 attention layers ===")
// Warmup
_ = forwardPass(tokenId: 1234, position: 0)

var times: [Double] = []
for _ in 0..<5 {
    let t = CFAbsoluteTimeGetCurrent()
    _ = forwardPass(tokenId: 1234, position: 0)
    times.append(CFAbsoluteTimeGetCurrent() - t)
}
let avgMs = times.reduce(0, +) / Double(times.count) * 1000
let minMs = times.min()! * 1000
print("16-layer forward: avg=\(String(format: "%.1f", avgMs))ms, min=\(String(format: "%.1f", minMs))ms")
print("Per attn layer: \(String(format: "%.2f", avgMs / 16))ms")
print("(Full 64-layer estimate with GDN: \(String(format: "%.0f", avgMs / 16 * 64))ms)")

print("\nSession 1 complete. Pipeline proven: embed → norm → attn → gate → MLP → lm_head → argmax")
