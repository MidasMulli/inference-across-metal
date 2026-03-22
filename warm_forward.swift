// warm_forward.swift — Warm-path weight wiring for Qwen3.5-27B
// Pre-loads ALL 64 transformer layers + lm_head into Metal shared buffers at startup.
// Eliminates per-token SSD page faults that caused 14-16s/token in the cold path.
//
// FINDINGS on 16GB M5 Air:
// - Model is 14.4GB, OS needs ~4.6GB → can't fit all weights in RAM
// - Pre-loading into Metal buffers causes swap thrashing (WORSE than mmap)
// - Best strategy: mmap with MADV_WILLNEED prefetch + double-buffered layer loading
// - Overlap CPU weight loading with GPU compute via async command buffers
//
// Build: swiftc -O -framework Metal -framework MetalPerformanceShaders warm_forward.swift -o warm_forward
// Run:   ./warm_forward

import Metal
import Foundation

// Disable stdout buffering so output appears immediately when piped
setbuf(stdout, nil)

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
let ROTARY_DIM: Int = 64
let GROUP_SIZE: Int = 64
let BITS: Int = 4

// GDN config
let GDN_NUM_K_HEADS: Int = 16
let GDN_NUM_V_HEADS: Int = 48
let GDN_HEAD_K_DIM: Int = 128
let GDN_HEAD_V_DIM: Int = 128
let GDN_KEY_DIM: Int = 2048
let GDN_VALUE_DIM: Int = 6144
let GDN_CONV_DIM: Int = 10240
let GDN_CONV_KERNEL: Int = 4
let GDN_KV_REPEAT: Int = 3
let GDN_INV_SCALE: Float = 0.08838834764  // 1/sqrt(128)
let GDN_STATE_SIZE: Int = 48 * 128 * 128

let FULL_ATTN_INTERVAL: Int = 4

func isFullAttn(_ layerIdx: Int) -> Bool {
    return (layerIdx + 1) % FULL_ATTN_INTERVAL == 0
}

// ============================================================================
// MARK: - Metal Kernel Source
// ============================================================================

let metalSource = """
#include <metal_stdlib>
using namespace metal;

kernel void rms_norm_bf16(
    device const bfloat* x      [[buffer(0)]],
    device const bfloat* weight  [[buffer(1)]],
    device bfloat* out           [[buffer(2)]],
    constant uint& dim           [[buffer(3)]],
    constant float& eps          [[buffer(4)]],
    uint tid [[thread_position_in_grid]],
    uint threads [[threads_per_grid]])
{
    float sum_sq = 0.0f;
    for (uint i = tid; i < dim; i += threads) {
        float v = float(x[i]);
        sum_sq += v * v;
    }
    threadgroup float shared[256];
    shared[tid] = sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = threads / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float rms = sqrt(shared[0] / float(dim) + eps);
    for (uint i = tid; i < dim; i += threads) {
        float v = float(x[i]);
        float w = float(weight[i]);
        out[i] = bfloat((v / rms) * w);
    }
}

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

kernel void embed_lookup_row_bf16(
    device const uint* weight_row   [[buffer(0)]],
    device const bfloat* scale_row  [[buffer(1)]],
    device const bfloat* bias_row   [[buffer(2)]],
    device bfloat* out              [[buffer(3)]],
    constant uint& hidden_dim       [[buffer(4)]],
    constant uint& group_sz         [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= hidden_dim) return;
    uint packed_dim = hidden_dim / 8;
    uint groups_per_row = hidden_dim / group_sz;
    uint pack_idx = tid / 8;
    uint nibble_idx = tid % 8;
    uint packed_val = weight_row[pack_idx];
    uint nibble = (packed_val >> (nibble_idx * 4)) & 0xF;
    uint group_idx = tid / group_sz;
    float scale = float(scale_row[group_idx]);
    float bias = float(bias_row[group_idx]);
    out[tid] = bfloat(float(nibble) * scale + bias);
}

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
    uint pair = tid.x;
    if (head >= num_heads || pair >= rotary_dim / 2) return;
    uint base = head * head_dim + pair * 2;
    float x0 = float(x[base]);
    float x1 = float(x[base + 1]);
    float c = cos_cache[pair];
    float s = sin_cache[pair];
    x[base]     = bfloat(x0 * c - x1 * s);
    x[base + 1] = bfloat(x0 * s + x1 * c);
}

kernel void argmax_bf16(
    device const bfloat* x   [[buffer(0)]],
    device uint* result      [[buffer(1)]],
    constant uint& count     [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid != 0) return;
    float max_val = float(x[0]);
    uint max_idx = 0;
    for (uint i = 1; i < count; i++) {
        float v = float(x[i]);
        if (v > max_val) { max_val = v; max_idx = i; }
    }
    result[0] = max_idx;
}

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

kernel void expand_kv_heads_bf16(
    device const bfloat* kv   [[buffer(0)]],
    device bfloat* out        [[buffer(1)]],
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

kernel void split_q_gate_bf16(
    device const bfloat* qg    [[buffer(0)]],
    device bfloat* q           [[buffer(1)]],
    device bfloat* gate        [[buffer(2)]],
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

kernel void depthwise_conv1d_m1_bf16(
    device const bfloat* conv_state   [[buffer(0)]],
    device const bfloat* new_input    [[buffer(1)]],
    device const bfloat* conv_weight  [[buffer(2)]],
    device bfloat* out                [[buffer(3)]],
    device bfloat* new_state_out      [[buffer(4)]],
    constant uint& conv_dim           [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= conv_dim) return;
    float window[4];
    window[0] = float(conv_state[0 * conv_dim + tid]);
    window[1] = float(conv_state[1 * conv_dim + tid]);
    window[2] = float(conv_state[2 * conv_dim + tid]);
    window[3] = float(new_input[tid]);
    float sum = 0.0f;
    for (int i = 0; i < 4; i++) {
        sum += window[i] * float(conv_weight[tid * 4 + i]);
    }
    out[tid] = bfloat(sum);
    new_state_out[0 * conv_dim + tid] = conv_state[1 * conv_dim + tid];
    new_state_out[1 * conv_dim + tid] = conv_state[2 * conv_dim + tid];
    new_state_out[2 * conv_dim + tid] = new_input[tid];
}

kernel void silu_bf16(
    device const bfloat* x   [[buffer(0)]],
    device bfloat* out       [[buffer(1)]],
    constant uint& count     [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= count) return;
    float v = float(x[tid]);
    out[tid] = bfloat(v / (1.0f + exp(-v)));
}

kernel void per_head_rms_norm_scaled_bf16(
    device const bfloat* x       [[buffer(0)]],
    device bfloat* out           [[buffer(1)]],
    constant uint& num_heads     [[buffer(2)]],
    constant uint& head_dim      [[buffer(3)]],
    constant float& eps          [[buffer(4)]],
    constant float& scale        [[buffer(5)]],
    uint2 tid [[thread_position_in_grid]])
{
    uint head = tid.y;
    uint elem = tid.x;
    if (head >= num_heads || elem >= head_dim) return;
    float sum_sq = 0.0f;
    uint base = head * head_dim;
    for (uint i = 0; i < head_dim; i++) {
        float v = float(x[base + i]);
        sum_sq += v * v;
    }
    float rms = sqrt(sum_sq / float(head_dim) + eps);
    float v = float(x[base + elem]);
    out[base + elem] = bfloat(scale * (v / rms));
}

kernel void compute_g_beta_bf16(
    device const bfloat* a       [[buffer(0)]],
    device const bfloat* b       [[buffer(1)]],
    device const float* A_log    [[buffer(2)]],
    device const bfloat* dt_bias [[buffer(3)]],
    device bfloat* g_out         [[buffer(4)]],
    device bfloat* beta_out      [[buffer(5)]],
    constant uint& num_v_heads   [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= num_v_heads) return;
    float a_val = float(a[tid]);
    float b_val = float(b[tid]);
    float A_val = A_log[tid];
    float dt_val = float(dt_bias[tid]);
    float sp_input = a_val + dt_val;
    float softplus = sp_input > 20.0f ? sp_input : log(1.0f + exp(sp_input));
    float g = exp(-exp(A_val) * softplus);
    g_out[tid] = bfloat(g);
    beta_out[tid] = bfloat(1.0f / (1.0f + exp(-b_val)));
}

kernel void gated_delta_step_bf16(
    device const bfloat* q       [[buffer(0)]],
    device const bfloat* k       [[buffer(1)]],
    device const bfloat* v       [[buffer(2)]],
    device const bfloat* g       [[buffer(3)]],
    device const bfloat* beta    [[buffer(4)]],
    device float* state          [[buffer(5)]],
    device bfloat* y_out         [[buffer(6)]],
    constant uint& num_v_heads   [[buffer(7)]],
    constant uint& head_v_dim    [[buffer(8)]],
    constant uint& head_k_dim    [[buffer(9)]],
    uint2 tid [[thread_position_in_grid]])
{
    uint vh = tid.y;
    uint dv = tid.x;
    if (vh >= num_v_heads || dv >= head_v_dim) return;
    float g_val = float(g[vh]);
    float beta_val = float(beta[vh]);
    float v_val = float(v[vh * head_v_dim + dv]);
    uint state_base = (vh * head_v_dim + dv) * head_k_dim;
    float kv_mem = 0.0f;
    for (uint dk = 0; dk < head_k_dim; dk++) {
        float s = state[state_base + dk] * g_val;
        state[state_base + dk] = s;
        kv_mem += s * float(k[vh * head_k_dim + dk]);
    }
    float delta = (v_val - kv_mem) * beta_val;
    float y_val = 0.0f;
    for (uint dk = 0; dk < head_k_dim; dk++) {
        float new_s = state[state_base + dk] + float(k[vh * head_k_dim + dk]) * delta;
        state[state_base + dk] = new_s;
        y_val += new_s * float(q[vh * head_k_dim + dk]);
    }
    y_out[vh * head_v_dim + dv] = bfloat(y_val);
}

kernel void rms_norm_gated_bf16(
    device const bfloat* x       [[buffer(0)]],
    device const bfloat* z       [[buffer(1)]],
    device const bfloat* weight  [[buffer(2)]],
    device bfloat* out           [[buffer(3)]],
    constant uint& num_heads     [[buffer(4)]],
    constant uint& head_dim      [[buffer(5)]],
    constant float& eps          [[buffer(6)]],
    uint2 tid [[thread_position_in_grid]])
{
    uint head = tid.y;
    uint elem = tid.x;
    if (head >= num_heads || elem >= head_dim) return;
    uint base = head * head_dim;
    float sum_sq = 0.0f;
    for (uint i = 0; i < head_dim; i++) {
        float v = float(x[base + i]);
        sum_sq += v * v;
    }
    float rms = sqrt(sum_sq / float(head_dim) + eps);
    float x_normed = (float(x[base + elem]) / rms) * float(weight[elem]);
    float z_val = float(z[base + elem]);
    float silu_z = z_val / (1.0f + exp(-z_val));
    out[base + elem] = bfloat(silu_z * x_normed);
}

kernel void copy_bf16(
    device const bfloat* src [[buffer(0)]],
    device bfloat* dst       [[buffer(1)]],
    constant uint& count     [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= count) return;
    dst[tid] = src[tid];
}

kernel void sdpa_decode_bf16(
    device const bfloat* q        [[buffer(0)]],
    device const bfloat* k_cache  [[buffer(1)]],
    device const bfloat* v_cache  [[buffer(2)]],
    device bfloat* out            [[buffer(3)]],
    constant uint& num_heads      [[buffer(4)]],
    constant uint& num_kv_heads   [[buffer(5)]],
    constant uint& head_dim       [[buffer(6)]],
    constant uint& seq_len        [[buffer(7)]],
    uint tid [[thread_position_in_grid]])
{
    uint h = tid;
    if (h >= num_heads) return;
    uint kv_h = h / (num_heads / num_kv_heads);
    float scale = 1.0f / sqrt(float(head_dim));
    uint kv_stride = num_kv_heads * head_dim;
    float scores[1024];
    float max_s = -1e30f;
    for (uint t = 0; t < seq_len && t < 1024; t++) {
        float s = 0.0f;
        for (uint d = 0; d < head_dim; d++) {
            s += float(q[h * head_dim + d]) *
                 float(k_cache[t * kv_stride + kv_h * head_dim + d]);
        }
        s *= scale;
        scores[t] = s;
        if (s > max_s) max_s = s;
    }
    float sum_exp = 0.0f;
    for (uint t = 0; t < seq_len && t < 1024; t++) {
        scores[t] = exp(scores[t] - max_s);
        sum_exp += scores[t];
    }
    for (uint d = 0; d < head_dim; d++) {
        float v = 0.0f;
        for (uint t = 0; t < seq_len && t < 1024; t++) {
            v += (scores[t] / sum_exp) *
                 float(v_cache[t * kv_stride + kv_h * head_dim + d]);
        }
        out[h * head_dim + d] = bfloat(v);
    }
}
"""

// ============================================================================
// MARK: - Weight Loading Infrastructure
// ============================================================================

struct TensorInfo {
    let fileIdx: Int
    let byteOffset: Int
    let byteSize: Int
    let shape: [Int]
    let dtype: String
}

func parseTensorInfo(_ dict: [String: Any]) -> TensorInfo {
    return TensorInfo(
        fileIdx: dict["file_idx"] as! Int,
        byteOffset: dict["byte_offset"] as! Int,
        byteSize: dict["byte_size"] as! Int,
        shape: dict["shape"] as! [Int],
        dtype: dict["dtype"] as! String)
}

func loadTensorToBuffer(_ device: MTLDevice, _ mmaps: [UnsafeRawPointer],
                        _ info: TensorInfo) -> MTLBuffer {
    let src = mmaps[info.fileIdx] + info.byteOffset
    return device.makeBuffer(bytes: src, length: info.byteSize, options: .storageModeShared)!
}

// ============================================================================
// MARK: - Layer Weight Structures
// ============================================================================

struct AttnLayerWeights {
    var inputNormWeight: MTLBuffer?
    var qProjW: MTLBuffer?, qProjS: MTLBuffer?, qProjB: MTLBuffer?
    var kProjW: MTLBuffer?, kProjS: MTLBuffer?, kProjB: MTLBuffer?
    var vProjW: MTLBuffer?, vProjS: MTLBuffer?, vProjB: MTLBuffer?
    var oProjW: MTLBuffer?, oProjS: MTLBuffer?, oProjB: MTLBuffer?
    var qNormWeight: MTLBuffer?, kNormWeight: MTLBuffer?
    var postNormWeight: MTLBuffer?
    var gateProjW: MTLBuffer?, gateProjS: MTLBuffer?, gateProjB: MTLBuffer?
    var upProjW: MTLBuffer?, upProjS: MTLBuffer?, upProjB: MTLBuffer?
    var downProjW: MTLBuffer?, downProjS: MTLBuffer?, downProjB: MTLBuffer?
}

struct GDNLayerWeights {
    var inputNormWeight: MTLBuffer?
    var qkvProjW: MTLBuffer?, qkvProjS: MTLBuffer?, qkvProjB: MTLBuffer?
    var zProjW: MTLBuffer?, zProjS: MTLBuffer?, zProjB: MTLBuffer?
    var bProjW: MTLBuffer?, bProjS: MTLBuffer?, bProjB: MTLBuffer?
    var aProjW: MTLBuffer?, aProjS: MTLBuffer?, aProjB: MTLBuffer?
    var convWeight: MTLBuffer?
    var ALog: MTLBuffer?
    var dtBias: MTLBuffer?
    var normWeight: MTLBuffer?
    var outProjW: MTLBuffer?, outProjS: MTLBuffer?, outProjB: MTLBuffer?
    var postNormWeight: MTLBuffer?
    var gateProjW: MTLBuffer?, gateProjS: MTLBuffer?, gateProjB: MTLBuffer?
    var upProjW: MTLBuffer?, upProjS: MTLBuffer?, upProjB: MTLBuffer?
    var downProjW: MTLBuffer?, downProjS: MTLBuffer?, downProjB: MTLBuffer?
}

// ============================================================================
// MARK: - Setup
// ============================================================================

let device = MTLCreateSystemDefaultDevice()!
let queue = device.makeCommandQueue()!
print("Device: \(device.name)")
print("Max buffer: \(device.maxBufferLength / 1024 / 1024) MB")

let mlxMetalPath = String(cString: getenv("HOME")) +
    "/.mlx-env/lib/python3.11/site-packages/mlx/lib/mlx.metallib"
guard let mlxLib = try? device.makeLibrary(URL: URL(fileURLWithPath: mlxMetalPath)) else {
    print("ERROR: Cannot load MLX metallib"); exit(1)
}
let naxKernelName = "affine_qmm_t_nax_bfloat16_t_gs_64_b_4_bm64_bn64_bk64_wm2_wn2_alN_true_batch_0"
guard let naxFn = try? mlxLib.makeFunction(name: naxKernelName) else {
    print("ERROR: NAX kernel not found"); exit(1)
}
let naxPSO = try! device.makeComputePipelineState(function: naxFn)

let customLib = try! device.makeLibrary(source: metalSource, options: nil)
func makePSO(_ name: String) -> MTLComputePipelineState {
    let fn = customLib.makeFunction(name: name)!
    return try! device.makeComputePipelineState(function: fn)
}
let rmsNormPSO = makePSO("rms_norm_bf16")
let siluMulPSO = makePSO("silu_multiply_bf16")
let residualPSO = makePSO("residual_add_bf16")
let embedRowPSO = makePSO("embed_lookup_row_bf16")
let sigmoidMulPSO = makePSO("sigmoid_multiply_bf16")
let ropePSO = makePSO("rope_bf16")
let argmaxPSO = makePSO("argmax_bf16")
let perHeadRmsNormPSO = makePSO("per_head_rms_norm_bf16")
let expandKvPSO = makePSO("expand_kv_heads_bf16")
let splitQGatePSO = makePSO("split_q_gate_bf16")
let depthwiseConvPSO = makePSO("depthwise_conv1d_m1_bf16")
let siluPSO = makePSO("silu_bf16")
let perHeadRmsNormScaledPSO = makePSO("per_head_rms_norm_scaled_bf16")
let computeGBetaPSO = makePSO("compute_g_beta_bf16")
let gdnStepPSO = makePSO("gated_delta_step_bf16")
let rmsNormGatedPSO = makePSO("rms_norm_gated_bf16")
let copyPSO = makePSO("copy_bf16")
let sdpaPSO = makePSO("sdpa_decode_bf16")
print("All kernels compiled")

// ============================================================================
// MARK: - Load Weight Index & mmap
// ============================================================================

let indexPath = "/Users/midas/Desktop/cowork/inference-across-metal/reference_data/27b_full_index.json"
let indexData = try! Data(contentsOf: URL(fileURLWithPath: indexPath))
let index = try! JSONSerialization.jsonObject(with: indexData) as! [String: Any]
let shardFiles = index["shard_files"] as! [String]
let layersInfo = index["layers"] as! [[String: Any]]

var mmaps: [UnsafeRawPointer] = []
var mmapSizes: [Int] = []
var mmapMut: [UnsafeMutableRawPointer] = []  // for madvise
for path in shardFiles {
    let fd = open(path, O_RDONLY)
    guard fd >= 0 else { print("Cannot open \(path)"); exit(1) }
    let size = Int(lseek(fd, 0, SEEK_END))
    lseek(fd, 0, SEEK_SET)
    let ptr = mmap(nil, size, PROT_READ, MAP_PRIVATE, fd, 0)!
    close(fd)
    mmaps.append(UnsafeRawPointer(ptr))
    mmapMut.append(ptr)
    mmapSizes.append(size)
    // Set sequential readahead hint for all shards
    madvise(ptr, size, MADV_SEQUENTIAL)
}
print("mmap'd \(shardFiles.count) shard files (MADV_SEQUENTIAL)")

// ============================================================================
// MARK: - Embedding: targeted row lookup from mmap
// ============================================================================

let embedInfo = index["embed_tokens"] as! [String: Any]
let embedWeightInfo = parseTensorInfo(embedInfo["weight"] as! [String: Any])
let embedScalesInfo = parseTensorInfo(embedInfo["scales"] as! [String: Any])
let embedBiasesInfo = parseTensorInfo(embedInfo["biases"] as! [String: Any])

let embedPackedDim = HIDDEN_SIZE / 8
let embedWeightRowBytes = embedPackedDim * 4
let embedGroupsPerRow = HIDDEN_SIZE / GROUP_SIZE
let embedScaleRowBytes = embedGroupsPerRow * 2
let embedBiasRowBytes = embedGroupsPerRow * 2

let embedRowWBuf = device.makeBuffer(length: embedWeightRowBytes, options: .storageModeShared)!
let embedRowSBuf = device.makeBuffer(length: embedScaleRowBytes, options: .storageModeShared)!
let embedRowBBuf = device.makeBuffer(length: embedBiasRowBytes, options: .storageModeShared)!

print("Embedding: targeted row lookup (~3 KB/token)")

let finalNormInfo = index["final_norm"] as! [String: Any]
let finalNormW = loadTensorToBuffer(device, mmaps, parseTensorInfo(finalNormInfo["weight"] as! [String: Any]))

// ============================================================================
// MARK: - Pre-compute tensor info for all layers (no allocation, just metadata)
// ============================================================================

struct LayerTensorInfos {
    let layerIdx: Int
    let isAttn: Bool
    let tensors: [String: TensorInfo]
    let totalBytes: Int
}

var layerInfos: [LayerTensorInfos] = []
for layerIdx in 0..<NUM_LAYERS {
    let li = layersInfo[layerIdx]
    let tensors = li["tensors"] as! [String: Any]
    var infos: [String: TensorInfo] = [:]
    var totalBytes = 0
    for (name, val) in tensors {
        let info = parseTensorInfo(val as! [String: Any])
        infos[name] = info
        totalBytes += info.byteSize
    }
    layerInfos.append(LayerTensorInfos(
        layerIdx: layerIdx,
        isAttn: isFullAttn(layerIdx),
        tensors: infos,
        totalBytes: totalBytes))
}

let lmHeadInfo = index["lm_head"] as! [String: Any]
let lmHeadWInfo = parseTensorInfo(lmHeadInfo["weight"] as! [String: Any])
let lmHeadSInfo = parseTensorInfo(lmHeadInfo["scales"] as! [String: Any])
let lmHeadBInfo = parseTensorInfo(lmHeadInfo["biases"] as! [String: Any])

// ============================================================================
// MARK: - Prefetch helper: advise kernel to pre-load pages
// ============================================================================

func prefetchLayer(_ layerIdx: Int) {
    let info = layerInfos[layerIdx]
    for (_, tensorInfo) in info.tensors {
        let ptr = mmapMut[tensorInfo.fileIdx] + tensorInfo.byteOffset
        madvise(ptr, tensorInfo.byteSize, MADV_WILLNEED)
    }
}

func prefetchLmHead() {
    for info in [lmHeadWInfo, lmHeadSInfo, lmHeadBInfo] {
        let ptr = mmapMut[info.fileIdx] + info.byteOffset
        madvise(ptr, info.byteSize, MADV_WILLNEED)
    }
}

// ============================================================================
// MARK: - Layer Loading Functions
// ============================================================================

func loadAttnLayer(_ layerIdx: Int) -> AttnLayerWeights {
    let li = layersInfo[layerIdx]
    let tensors = li["tensors"] as! [String: Any]
    func loadT(_ name: String) -> MTLBuffer {
        return loadTensorToBuffer(device, mmaps, parseTensorInfo(tensors[name] as! [String: Any]))
    }
    var lw = AttnLayerWeights()
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
    return lw
}

func loadGDNLayer(_ layerIdx: Int) -> GDNLayerWeights {
    let li = layersInfo[layerIdx]
    let tensors = li["tensors"] as! [String: Any]
    func loadT(_ name: String) -> MTLBuffer {
        return loadTensorToBuffer(device, mmaps, parseTensorInfo(tensors[name] as! [String: Any]))
    }
    var lw = GDNLayerWeights()
    lw.inputNormWeight = loadT("input_layernorm.weight")
    lw.qkvProjW = loadT("linear_attn.in_proj_qkv.weight")
    lw.qkvProjS = loadT("linear_attn.in_proj_qkv.scales")
    lw.qkvProjB = loadT("linear_attn.in_proj_qkv.biases")
    lw.zProjW = loadT("linear_attn.in_proj_z.weight")
    lw.zProjS = loadT("linear_attn.in_proj_z.scales")
    lw.zProjB = loadT("linear_attn.in_proj_z.biases")
    lw.bProjW = loadT("linear_attn.in_proj_b.weight")
    lw.bProjS = loadT("linear_attn.in_proj_b.scales")
    lw.bProjB = loadT("linear_attn.in_proj_b.biases")
    lw.aProjW = loadT("linear_attn.in_proj_a.weight")
    lw.aProjS = loadT("linear_attn.in_proj_a.scales")
    lw.aProjB = loadT("linear_attn.in_proj_a.biases")
    lw.convWeight = loadT("linear_attn.conv1d.weight")
    lw.ALog = loadT("linear_attn.A_log")
    lw.dtBias = loadT("linear_attn.dt_bias")
    lw.normWeight = loadT("linear_attn.norm.weight")
    lw.outProjW = loadT("linear_attn.out_proj.weight")
    lw.outProjS = loadT("linear_attn.out_proj.scales")
    lw.outProjB = loadT("linear_attn.out_proj.biases")
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
    return lw
}

// ============================================================================
// MARK: - Activation Buffers
// ============================================================================

let hiddenBuf = device.makeBuffer(length: HIDDEN_SIZE * 2, options: .storageModeShared)!
let normedBuf = device.makeBuffer(length: HIDDEN_SIZE * 2, options: .storageModeShared)!
let residualBuf = device.makeBuffer(length: HIDDEN_SIZE * 2, options: .storageModeShared)!

let qBuf = device.makeBuffer(length: NUM_HEADS * HEAD_DIM * 2 * 2, options: .storageModeShared)!
let queriesBuf = device.makeBuffer(length: NUM_HEADS * HEAD_DIM * 2, options: .storageModeShared)!
let attnGateBuf = device.makeBuffer(length: NUM_HEADS * HEAD_DIM * 2, options: .storageModeShared)!
let kBuf = device.makeBuffer(length: NUM_KV_HEADS * HEAD_DIM * 2, options: .storageModeShared)!
let vBuf = device.makeBuffer(length: NUM_KV_HEADS * HEAD_DIM * 2, options: .storageModeShared)!
let qNormedBuf = device.makeBuffer(length: NUM_HEADS * HEAD_DIM * 2, options: .storageModeShared)!
let kNormedBuf = device.makeBuffer(length: NUM_KV_HEADS * HEAD_DIM * 2, options: .storageModeShared)!
let attnOutBuf = device.makeBuffer(length: NUM_HEADS * HEAD_DIM * 2, options: .storageModeShared)!
let gatedBuf = device.makeBuffer(length: NUM_HEADS * HEAD_DIM * 2, options: .storageModeShared)!
let oOutBuf = device.makeBuffer(length: HIDDEN_SIZE * 2, options: .storageModeShared)!

let qkvBuf = device.makeBuffer(length: GDN_CONV_DIM * 2, options: .storageModeShared)!
let zBuf = device.makeBuffer(length: GDN_VALUE_DIM * 2, options: .storageModeShared)!
let aBuf = device.makeBuffer(length: GDN_NUM_V_HEADS * 2, options: .storageModeShared)!
let bBuf = device.makeBuffer(length: GDN_NUM_V_HEADS * 2, options: .storageModeShared)!
let convOutBuf = device.makeBuffer(length: GDN_CONV_DIM * 2, options: .storageModeShared)!
let convSiluBuf = device.makeBuffer(length: GDN_CONV_DIM * 2, options: .storageModeShared)!
let gdnQNormedBuf = device.makeBuffer(length: GDN_KEY_DIM * 2, options: .storageModeShared)!
let gdnKNormedBuf = device.makeBuffer(length: GDN_KEY_DIM * 2, options: .storageModeShared)!
let gdnQExpandedBuf = device.makeBuffer(length: GDN_NUM_V_HEADS * GDN_HEAD_K_DIM * 2, options: .storageModeShared)!
let gdnKExpandedBuf = device.makeBuffer(length: GDN_NUM_V_HEADS * GDN_HEAD_K_DIM * 2, options: .storageModeShared)!
let gBuf = device.makeBuffer(length: GDN_NUM_V_HEADS * 2, options: .storageModeShared)!
let betaBuf = device.makeBuffer(length: GDN_NUM_V_HEADS * 2, options: .storageModeShared)!
let gdnYBuf = device.makeBuffer(length: GDN_VALUE_DIM * 2, options: .storageModeShared)!
let gdnNormedBuf = device.makeBuffer(length: GDN_VALUE_DIM * 2, options: .storageModeShared)!
let gdnOutProjBuf = device.makeBuffer(length: HIDDEN_SIZE * 2, options: .storageModeShared)!

let gateProjBuf = device.makeBuffer(length: INTERMEDIATE_SIZE * 2, options: .storageModeShared)!
let upProjBuf = device.makeBuffer(length: INTERMEDIATE_SIZE * 2, options: .storageModeShared)!
let mlpHiddenBuf = device.makeBuffer(length: INTERMEDIATE_SIZE * 2, options: .storageModeShared)!
let downProjBuf = device.makeBuffer(length: HIDDEN_SIZE * 2, options: .storageModeShared)!

let logitsBuf = device.makeBuffer(length: VOCAB_SIZE * 2, options: .storageModeShared)!
let argmaxBuf = device.makeBuffer(length: 4, options: .storageModeShared)!

let ropeCosBuf = device.makeBuffer(length: ROTARY_DIM / 2 * 4, options: .storageModeShared)!
let ropeSinBuf = device.makeBuffer(length: ROTARY_DIM / 2 * 4, options: .storageModeShared)!

// ============================================================================
// MARK: - GDN Persistent State
// ============================================================================

let convStateBytes = (GDN_CONV_KERNEL - 1) * GDN_CONV_DIM * 2
let deltaStateBytes = GDN_STATE_SIZE * 4

let numGDNLayers = 48
let convStateA = device.makeBuffer(length: convStateBytes * numGDNLayers, options: .storageModeShared)!
let convStateB = device.makeBuffer(length: convStateBytes * numGDNLayers, options: .storageModeShared)!
let deltaState = device.makeBuffer(length: deltaStateBytes * numGDNLayers, options: .storageModeShared)!

memset(convStateA.contents(), 0, convStateA.length)
memset(convStateB.contents(), 0, convStateB.length)
memset(deltaState.contents(), 0, deltaState.length)
print("GDN state: \((convStateA.length * 2 + deltaState.length) / 1024 / 1024) MB")

var gdnLayerIndex: [Int: Int] = [:]
var gdnIdx = 0
for i in 0..<NUM_LAYERS { if !isFullAttn(i) { gdnLayerIndex[i] = gdnIdx; gdnIdx += 1 } }

var attnLayerIndex: [Int: Int] = [:]
var attnIdx = 0
for i in 0..<NUM_LAYERS { if isFullAttn(i) { attnLayerIndex[i] = attnIdx; attnIdx += 1 } }

// ============================================================================
// MARK: - KV Cache
// ============================================================================

let MAX_SEQ: Int = 1024
let kvTokenBytes = NUM_KV_HEADS * HEAD_DIM * 2
let kvLayerBytes = MAX_SEQ * kvTokenBytes
let numAttnLayers = 16

let kCache = device.makeBuffer(length: kvLayerBytes * numAttnLayers, options: .storageModeShared)!
let vCache = device.makeBuffer(length: kvLayerBytes * numAttnLayers, options: .storageModeShared)!
memset(kCache.contents(), 0, kCache.length)
memset(vCache.contents(), 0, vCache.length)
print("KV cache: \(kCache.length * 2 / 1024) KB")

// ============================================================================
// MARK: - Helpers
// ============================================================================

func dispatchQMatmul(_ enc: MTLComputeCommandEncoder,
                     _ wBuf: MTLBuffer, _ sBuf: MTLBuffer, _ bBuf_: MTLBuffer,
                     _ xBuf: MTLBuffer, _ yBuf: MTLBuffer,
                     K_dim: Int, N_dim: Int, M_dim: Int) {
    enc.setComputePipelineState(naxPSO)
    enc.setBuffer(wBuf, offset: 0, index: 0)
    enc.setBuffer(sBuf, offset: 0, index: 1)
    enc.setBuffer(bBuf_, offset: 0, index: 2)
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

func embedFromMmap(tokenId: Int) {
    let wSrc = mmaps[embedWeightInfo.fileIdx] + embedWeightInfo.byteOffset + tokenId * embedWeightRowBytes
    let sSrc = mmaps[embedScalesInfo.fileIdx] + embedScalesInfo.byteOffset + tokenId * embedScaleRowBytes
    let bSrc = mmaps[embedBiasesInfo.fileIdx] + embedBiasesInfo.byteOffset + tokenId * embedBiasRowBytes
    memcpy(embedRowWBuf.contents(), wSrc, embedWeightRowBytes)
    memcpy(embedRowSBuf.contents(), sSrc, embedScaleRowBytes)
    memcpy(embedRowBBuf.contents(), bSrc, embedBiasRowBytes)
}

// ============================================================================
// MARK: - Dispatch: Attention Layer
// ============================================================================

func dispatchAttnLayer(_ enc: MTLComputeCommandEncoder, _ lw: AttnLayerWeights,
                       attnIdx: Int, position: Int) {
    var dim = UInt32(HIDDEN_SIZE); var eps = RMS_NORM_EPS; var count = UInt32(HIDDEN_SIZE)

    enc.setComputePipelineState(rmsNormPSO)
    enc.setBuffer(hiddenBuf, offset: 0, index: 0)
    enc.setBuffer(lw.inputNormWeight!, offset: 0, index: 1)
    enc.setBuffer(normedBuf, offset: 0, index: 2)
    enc.setBytes(&dim, length: 4, index: 3)
    enc.setBytes(&eps, length: 4, index: 4)
    enc.dispatchThreads(MTLSize(width: 256, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

    dispatchQMatmul(enc, lw.qProjW!, lw.qProjS!, lw.qProjB!,
                    normedBuf, qBuf, K_dim: HIDDEN_SIZE, N_dim: NUM_HEADS * HEAD_DIM * 2, M_dim: 1)

    enc.setComputePipelineState(splitQGatePSO)
    enc.setBuffer(qBuf, offset: 0, index: 0)
    enc.setBuffer(queriesBuf, offset: 0, index: 1)
    enc.setBuffer(attnGateBuf, offset: 0, index: 2)
    var nh = UInt32(NUM_HEADS); var hd = UInt32(HEAD_DIM)
    enc.setBytes(&nh, length: 4, index: 3); enc.setBytes(&hd, length: 4, index: 4)
    enc.dispatchThreads(MTLSize(width: HEAD_DIM, height: NUM_HEADS, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

    dispatchQMatmul(enc, lw.kProjW!, lw.kProjS!, lw.kProjB!,
                    normedBuf, kBuf, K_dim: HIDDEN_SIZE, N_dim: NUM_KV_HEADS * HEAD_DIM, M_dim: 1)
    dispatchQMatmul(enc, lw.vProjW!, lw.vProjS!, lw.vProjB!,
                    normedBuf, vBuf, K_dim: HIDDEN_SIZE, N_dim: NUM_KV_HEADS * HEAD_DIM, M_dim: 1)

    enc.setComputePipelineState(perHeadRmsNormPSO)
    enc.setBuffer(queriesBuf, offset: 0, index: 0); enc.setBuffer(lw.qNormWeight!, offset: 0, index: 1)
    enc.setBuffer(qNormedBuf, offset: 0, index: 2)
    enc.setBytes(&nh, length: 4, index: 3); enc.setBytes(&hd, length: 4, index: 4); enc.setBytes(&eps, length: 4, index: 5)
    enc.dispatchThreads(MTLSize(width: HEAD_DIM, height: NUM_HEADS, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: HEAD_DIM, height: 1, depth: 1))

    var nkv = UInt32(NUM_KV_HEADS)
    enc.setBuffer(kBuf, offset: 0, index: 0); enc.setBuffer(lw.kNormWeight!, offset: 0, index: 1)
    enc.setBuffer(kNormedBuf, offset: 0, index: 2)
    enc.setBytes(&nkv, length: 4, index: 3); enc.setBytes(&hd, length: 4, index: 4); enc.setBytes(&eps, length: 4, index: 5)
    enc.dispatchThreads(MTLSize(width: HEAD_DIM, height: NUM_KV_HEADS, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: HEAD_DIM, height: 1, depth: 1))

    enc.setComputePipelineState(ropePSO)
    enc.setBuffer(qNormedBuf, offset: 0, index: 0)
    enc.setBuffer(ropeCosBuf, offset: 0, index: 1); enc.setBuffer(ropeSinBuf, offset: 0, index: 2)
    enc.setBytes(&nh, length: 4, index: 3); enc.setBytes(&hd, length: 4, index: 4)
    var rd = UInt32(ROTARY_DIM); enc.setBytes(&rd, length: 4, index: 5)
    enc.dispatchThreads(MTLSize(width: ROTARY_DIM / 2, height: NUM_HEADS, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.setBuffer(kNormedBuf, offset: 0, index: 0); enc.setBytes(&nkv, length: 4, index: 3)
    enc.dispatchThreads(MTLSize(width: ROTARY_DIM / 2, height: NUM_KV_HEADS, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))

    let cacheLayerOffset = attnIdx * kvLayerBytes
    let cacheTokenOffset = cacheLayerOffset + position * kvTokenBytes
    enc.setComputePipelineState(copyPSO)
    enc.setBuffer(kNormedBuf, offset: 0, index: 0); enc.setBuffer(kCache, offset: cacheTokenOffset, index: 1)
    var kvCount = UInt32(NUM_KV_HEADS * HEAD_DIM); enc.setBytes(&kvCount, length: 4, index: 2)
    enc.dispatchThreads(MTLSize(width: NUM_KV_HEADS * HEAD_DIM, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    enc.setBuffer(vBuf, offset: 0, index: 0); enc.setBuffer(vCache, offset: cacheTokenOffset, index: 1)
    enc.dispatchThreads(MTLSize(width: NUM_KV_HEADS * HEAD_DIM, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

    enc.setComputePipelineState(sdpaPSO)
    enc.setBuffer(qNormedBuf, offset: 0, index: 0)
    enc.setBuffer(kCache, offset: cacheLayerOffset, index: 1); enc.setBuffer(vCache, offset: cacheLayerOffset, index: 2)
    enc.setBuffer(attnOutBuf, offset: 0, index: 3)
    enc.setBytes(&nh, length: 4, index: 4); enc.setBytes(&nkv, length: 4, index: 5)
    enc.setBytes(&hd, length: 4, index: 6)
    var seqLen = UInt32(position + 1); enc.setBytes(&seqLen, length: 4, index: 7)
    enc.dispatchThreads(MTLSize(width: NUM_HEADS, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: NUM_HEADS, height: 1, depth: 1))

    enc.setComputePipelineState(sigmoidMulPSO)
    enc.setBuffer(attnOutBuf, offset: 0, index: 0); enc.setBuffer(attnGateBuf, offset: 0, index: 1)
    enc.setBuffer(gatedBuf, offset: 0, index: 2)
    var gateCount = UInt32(NUM_HEADS * HEAD_DIM); enc.setBytes(&gateCount, length: 4, index: 3)
    enc.dispatchThreads(MTLSize(width: NUM_HEADS * HEAD_DIM, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

    dispatchQMatmul(enc, lw.oProjW!, lw.oProjS!, lw.oProjB!,
                    gatedBuf, oOutBuf, K_dim: NUM_HEADS * HEAD_DIM, N_dim: HIDDEN_SIZE, M_dim: 1)

    enc.setComputePipelineState(residualPSO)
    enc.setBuffer(hiddenBuf, offset: 0, index: 0); enc.setBuffer(oOutBuf, offset: 0, index: 1)
    enc.setBuffer(residualBuf, offset: 0, index: 2); enc.setBytes(&count, length: 4, index: 3)
    enc.dispatchThreads(MTLSize(width: HIDDEN_SIZE, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

    enc.setComputePipelineState(rmsNormPSO)
    enc.setBuffer(residualBuf, offset: 0, index: 0); enc.setBuffer(lw.postNormWeight!, offset: 0, index: 1)
    enc.setBuffer(normedBuf, offset: 0, index: 2)
    enc.setBytes(&dim, length: 4, index: 3); enc.setBytes(&eps, length: 4, index: 4)
    enc.dispatchThreads(MTLSize(width: 256, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

    dispatchMLP(enc, lw.gateProjW!, lw.gateProjS!, lw.gateProjB!,
                lw.upProjW!, lw.upProjS!, lw.upProjB!,
                lw.downProjW!, lw.downProjS!, lw.downProjB!)

    enc.setComputePipelineState(residualPSO)
    enc.setBuffer(residualBuf, offset: 0, index: 0); enc.setBuffer(downProjBuf, offset: 0, index: 1)
    enc.setBuffer(hiddenBuf, offset: 0, index: 2); enc.setBytes(&count, length: 4, index: 3)
    enc.dispatchThreads(MTLSize(width: HIDDEN_SIZE, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
}

// ============================================================================
// MARK: - Dispatch: GDN Layer
// ============================================================================

func dispatchGDNLayer(_ enc: MTLComputeCommandEncoder, _ lw: GDNLayerWeights,
                      gdnIdx: Int, useConvA: Bool) {
    let convStateRead = useConvA ? convStateA : convStateB
    let convStateWrite = useConvA ? convStateB : convStateA
    let convOffset = gdnIdx * convStateBytes
    let deltaOffset = gdnIdx * deltaStateBytes

    var dim = UInt32(HIDDEN_SIZE); var eps = RMS_NORM_EPS; var count = UInt32(HIDDEN_SIZE)

    enc.setComputePipelineState(rmsNormPSO)
    enc.setBuffer(hiddenBuf, offset: 0, index: 0); enc.setBuffer(lw.inputNormWeight!, offset: 0, index: 1)
    enc.setBuffer(normedBuf, offset: 0, index: 2)
    enc.setBytes(&dim, length: 4, index: 3); enc.setBytes(&eps, length: 4, index: 4)
    enc.dispatchThreads(MTLSize(width: 256, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

    dispatchQMatmul(enc, lw.qkvProjW!, lw.qkvProjS!, lw.qkvProjB!,
                    normedBuf, qkvBuf, K_dim: HIDDEN_SIZE, N_dim: GDN_CONV_DIM, M_dim: 1)
    dispatchQMatmul(enc, lw.zProjW!, lw.zProjS!, lw.zProjB!,
                    normedBuf, zBuf, K_dim: HIDDEN_SIZE, N_dim: GDN_VALUE_DIM, M_dim: 1)
    dispatchQMatmul(enc, lw.bProjW!, lw.bProjS!, lw.bProjB!,
                    normedBuf, bBuf, K_dim: HIDDEN_SIZE, N_dim: GDN_NUM_V_HEADS, M_dim: 1)
    dispatchQMatmul(enc, lw.aProjW!, lw.aProjS!, lw.aProjB!,
                    normedBuf, aBuf, K_dim: HIDDEN_SIZE, N_dim: GDN_NUM_V_HEADS, M_dim: 1)

    enc.setComputePipelineState(depthwiseConvPSO)
    enc.setBuffer(convStateRead, offset: convOffset, index: 0); enc.setBuffer(qkvBuf, offset: 0, index: 1)
    enc.setBuffer(lw.convWeight!, offset: 0, index: 2); enc.setBuffer(convOutBuf, offset: 0, index: 3)
    enc.setBuffer(convStateWrite, offset: convOffset, index: 4)
    var cdim = UInt32(GDN_CONV_DIM); enc.setBytes(&cdim, length: 4, index: 5)
    enc.dispatchThreads(MTLSize(width: GDN_CONV_DIM, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

    enc.setComputePipelineState(siluPSO)
    enc.setBuffer(convOutBuf, offset: 0, index: 0); enc.setBuffer(convSiluBuf, offset: 0, index: 1)
    enc.setBytes(&cdim, length: 4, index: 2)
    enc.dispatchThreads(MTLSize(width: GDN_CONV_DIM, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

    enc.setComputePipelineState(perHeadRmsNormScaledPSO)
    enc.setBuffer(convSiluBuf, offset: 0, index: 0); enc.setBuffer(gdnQNormedBuf, offset: 0, index: 1)
    var nkh = UInt32(GDN_NUM_K_HEADS); var hkd = UInt32(GDN_HEAD_K_DIM); var normEps: Float = 1e-6
    enc.setBytes(&nkh, length: 4, index: 2); enc.setBytes(&hkd, length: 4, index: 3)
    enc.setBytes(&normEps, length: 4, index: 4)
    var qScale: Float = GDN_INV_SCALE * GDN_INV_SCALE; enc.setBytes(&qScale, length: 4, index: 5)
    enc.dispatchThreads(MTLSize(width: GDN_HEAD_K_DIM, height: GDN_NUM_K_HEADS, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: GDN_HEAD_K_DIM, height: 1, depth: 1))

    enc.setBuffer(convSiluBuf, offset: GDN_KEY_DIM * 2, index: 0); enc.setBuffer(gdnKNormedBuf, offset: 0, index: 1)
    var kScale: Float = GDN_INV_SCALE; enc.setBytes(&kScale, length: 4, index: 5)
    enc.dispatchThreads(MTLSize(width: GDN_HEAD_K_DIM, height: GDN_NUM_K_HEADS, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: GDN_HEAD_K_DIM, height: 1, depth: 1))

    var nvh = UInt32(GDN_NUM_V_HEADS)
    enc.setComputePipelineState(expandKvPSO)
    enc.setBuffer(gdnQNormedBuf, offset: 0, index: 0); enc.setBuffer(gdnQExpandedBuf, offset: 0, index: 1)
    enc.setBytes(&nvh, length: 4, index: 2); enc.setBytes(&nkh, length: 4, index: 3); enc.setBytes(&hkd, length: 4, index: 4)
    enc.dispatchThreads(MTLSize(width: GDN_HEAD_K_DIM, height: GDN_NUM_V_HEADS, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    enc.setBuffer(gdnKNormedBuf, offset: 0, index: 0); enc.setBuffer(gdnKExpandedBuf, offset: 0, index: 1)
    enc.dispatchThreads(MTLSize(width: GDN_HEAD_K_DIM, height: GDN_NUM_V_HEADS, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))

    enc.setComputePipelineState(computeGBetaPSO)
    enc.setBuffer(aBuf, offset: 0, index: 0); enc.setBuffer(bBuf, offset: 0, index: 1)
    enc.setBuffer(lw.ALog!, offset: 0, index: 2); enc.setBuffer(lw.dtBias!, offset: 0, index: 3)
    enc.setBuffer(gBuf, offset: 0, index: 4); enc.setBuffer(betaBuf, offset: 0, index: 5)
    enc.setBytes(&nvh, length: 4, index: 6)
    enc.dispatchThreads(MTLSize(width: GDN_NUM_V_HEADS, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 48, height: 1, depth: 1))

    enc.setComputePipelineState(gdnStepPSO)
    enc.setBuffer(gdnQExpandedBuf, offset: 0, index: 0); enc.setBuffer(gdnKExpandedBuf, offset: 0, index: 1)
    enc.setBuffer(convSiluBuf, offset: GDN_KEY_DIM * 2 * 2, index: 2)
    enc.setBuffer(gBuf, offset: 0, index: 3); enc.setBuffer(betaBuf, offset: 0, index: 4)
    enc.setBuffer(deltaState, offset: deltaOffset, index: 5); enc.setBuffer(gdnYBuf, offset: 0, index: 6)
    var hvd = UInt32(GDN_HEAD_V_DIM)
    enc.setBytes(&nvh, length: 4, index: 7); enc.setBytes(&hvd, length: 4, index: 8); enc.setBytes(&hkd, length: 4, index: 9)
    enc.dispatchThreads(MTLSize(width: GDN_HEAD_V_DIM, height: GDN_NUM_V_HEADS, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))

    enc.setComputePipelineState(rmsNormGatedPSO)
    enc.setBuffer(gdnYBuf, offset: 0, index: 0); enc.setBuffer(zBuf, offset: 0, index: 1)
    enc.setBuffer(lw.normWeight!, offset: 0, index: 2); enc.setBuffer(gdnNormedBuf, offset: 0, index: 3)
    enc.setBytes(&nvh, length: 4, index: 4); enc.setBytes(&hvd, length: 4, index: 5); enc.setBytes(&eps, length: 4, index: 6)
    enc.dispatchThreads(MTLSize(width: GDN_HEAD_V_DIM, height: GDN_NUM_V_HEADS, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))

    dispatchQMatmul(enc, lw.outProjW!, lw.outProjS!, lw.outProjB!,
                    gdnNormedBuf, gdnOutProjBuf, K_dim: GDN_VALUE_DIM, N_dim: HIDDEN_SIZE, M_dim: 1)

    enc.setComputePipelineState(residualPSO)
    enc.setBuffer(hiddenBuf, offset: 0, index: 0); enc.setBuffer(gdnOutProjBuf, offset: 0, index: 1)
    enc.setBuffer(residualBuf, offset: 0, index: 2); enc.setBytes(&count, length: 4, index: 3)
    enc.dispatchThreads(MTLSize(width: HIDDEN_SIZE, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

    enc.setComputePipelineState(rmsNormPSO)
    enc.setBuffer(residualBuf, offset: 0, index: 0); enc.setBuffer(lw.postNormWeight!, offset: 0, index: 1)
    enc.setBuffer(normedBuf, offset: 0, index: 2)
    enc.setBytes(&dim, length: 4, index: 3); enc.setBytes(&eps, length: 4, index: 4)
    enc.dispatchThreads(MTLSize(width: 256, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

    dispatchMLP(enc, lw.gateProjW!, lw.gateProjS!, lw.gateProjB!,
                lw.upProjW!, lw.upProjS!, lw.upProjB!,
                lw.downProjW!, lw.downProjS!, lw.downProjB!)

    enc.setComputePipelineState(residualPSO)
    enc.setBuffer(residualBuf, offset: 0, index: 0); enc.setBuffer(downProjBuf, offset: 0, index: 1)
    enc.setBuffer(hiddenBuf, offset: 0, index: 2); enc.setBytes(&count, length: 4, index: 3)
    enc.dispatchThreads(MTLSize(width: HIDDEN_SIZE, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
}

// ============================================================================
// MARK: - Dispatch: MLP
// ============================================================================

func dispatchMLP(_ enc: MTLComputeCommandEncoder,
                 _ gateW: MTLBuffer, _ gateS: MTLBuffer, _ gateB: MTLBuffer,
                 _ upW: MTLBuffer, _ upS: MTLBuffer, _ upB: MTLBuffer,
                 _ downW: MTLBuffer, _ downS: MTLBuffer, _ downB: MTLBuffer) {
    dispatchQMatmul(enc, gateW, gateS, gateB,
                    normedBuf, gateProjBuf, K_dim: HIDDEN_SIZE, N_dim: INTERMEDIATE_SIZE, M_dim: 1)
    dispatchQMatmul(enc, upW, upS, upB,
                    normedBuf, upProjBuf, K_dim: HIDDEN_SIZE, N_dim: INTERMEDIATE_SIZE, M_dim: 1)
    enc.setComputePipelineState(siluMulPSO)
    enc.setBuffer(gateProjBuf, offset: 0, index: 0); enc.setBuffer(upProjBuf, offset: 0, index: 1)
    enc.setBuffer(mlpHiddenBuf, offset: 0, index: 2)
    var isize = UInt32(INTERMEDIATE_SIZE); enc.setBytes(&isize, length: 4, index: 3)
    enc.dispatchThreads(MTLSize(width: INTERMEDIATE_SIZE, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    dispatchQMatmul(enc, downW, downS, downB,
                    mlpHiddenBuf, downProjBuf, K_dim: INTERMEDIATE_SIZE, N_dim: HIDDEN_SIZE, M_dim: 1)
}

// ============================================================================
// MARK: - Warm Forward Pass with Prefetch + Overlap
// ============================================================================

var convStateFlip = true

func warmForwardPass(tokenId: Int, position: Int, verbose: Bool = false) -> (tokenId: Int, timeMs: Double) {
    let tStart = CFAbsoluteTimeGetCurrent()
    precomputeRoPE(position: position)

    // Prefetch first layer
    prefetchLayer(0)

    // Embedding (targeted row from mmap)
    embedFromMmap(tokenId: tokenId)
    let cbEmbed = queue.makeCommandBuffer()!
    let encEmbed = cbEmbed.makeComputeCommandEncoder()!
    encEmbed.setComputePipelineState(embedRowPSO)
    encEmbed.setBuffer(embedRowWBuf, offset: 0, index: 0); encEmbed.setBuffer(embedRowSBuf, offset: 0, index: 1)
    encEmbed.setBuffer(embedRowBBuf, offset: 0, index: 2); encEmbed.setBuffer(hiddenBuf, offset: 0, index: 3)
    var hdim = UInt32(HIDDEN_SIZE); encEmbed.setBytes(&hdim, length: 4, index: 4)
    var gsz = UInt32(GROUP_SIZE); encEmbed.setBytes(&gsz, length: 4, index: 5)
    encEmbed.dispatchThreads(MTLSize(width: HIDDEN_SIZE, height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    encEmbed.endEncoding()
    cbEmbed.commit()
    cbEmbed.waitUntilCompleted()

    // 64 Layers with prefetch: load layer N, prefetch N+1, dispatch, wait
    for layerIdx in 0..<NUM_LAYERS {
        let tLayer = CFAbsoluteTimeGetCurrent()

        // Prefetch NEXT layer's weights (async kernel readahead)
        if layerIdx + 1 < NUM_LAYERS {
            prefetchLayer(layerIdx + 1)
        } else {
            prefetchLmHead()
        }

        // Load current layer weights into Metal buffers
        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!

        if isFullAttn(layerIdx) {
            let lw = loadAttnLayer(layerIdx)
            dispatchAttnLayer(enc, lw, attnIdx: attnLayerIndex[layerIdx]!, position: position)
        } else {
            let gIdx = gdnLayerIndex[layerIdx]!
            let lw = loadGDNLayer(layerIdx)
            dispatchGDNLayer(enc, lw, gdnIdx: gIdx, useConvA: convStateFlip)
        }

        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        if cb.status == .error {
            print("ERROR at layer \(layerIdx): \(cb.error?.localizedDescription ?? "unknown")")
            return (-1, 0)
        }

        if verbose {
            let layerMs = (CFAbsoluteTimeGetCurrent() - tLayer) * 1000
            let layerType = isFullAttn(layerIdx) ? "ATTN" : "GDN"
            if layerIdx < 4 || layerIdx >= 60 || layerIdx % 16 == 0 {
                print("  Layer \(String(format: "%2d", layerIdx)) [\(layerType)]: \(String(format: "%.1f", layerMs))ms")
            } else if layerIdx == 4 {
                print("  ...")
            }
        }
    }

    convStateFlip = !convStateFlip

    // Final norm
    let cbFinal = queue.makeCommandBuffer()!
    let encFinal = cbFinal.makeComputeCommandEncoder()!
    encFinal.setComputePipelineState(rmsNormPSO)
    encFinal.setBuffer(hiddenBuf, offset: 0, index: 0); encFinal.setBuffer(finalNormW, offset: 0, index: 1)
    encFinal.setBuffer(normedBuf, offset: 0, index: 2)
    var dim2 = UInt32(HIDDEN_SIZE); encFinal.setBytes(&dim2, length: 4, index: 3)
    var eps2 = RMS_NORM_EPS; encFinal.setBytes(&eps2, length: 4, index: 4)
    encFinal.dispatchThreads(MTLSize(width: 256, height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    encFinal.endEncoding()
    cbFinal.commit()
    cbFinal.waitUntilCompleted()

    // lm_head (streamed)
    let lmW = loadTensorToBuffer(device, mmaps, lmHeadWInfo)
    let lmS = loadTensorToBuffer(device, mmaps, lmHeadSInfo)
    let lmB = loadTensorToBuffer(device, mmaps, lmHeadBInfo)

    let cbLm = queue.makeCommandBuffer()!
    let encLm = cbLm.makeComputeCommandEncoder()!
    dispatchQMatmul(encLm, lmW, lmS, lmB,
                    normedBuf, logitsBuf, K_dim: HIDDEN_SIZE, N_dim: VOCAB_SIZE, M_dim: 1)
    encLm.setComputePipelineState(argmaxPSO)
    encLm.setBuffer(logitsBuf, offset: 0, index: 0); encLm.setBuffer(argmaxBuf, offset: 0, index: 1)
    var vocabSize = UInt32(VOCAB_SIZE); encLm.setBytes(&vocabSize, length: 4, index: 2)
    encLm.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                          threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
    encLm.endEncoding()
    cbLm.commit()
    cbLm.waitUntilCompleted()

    let result = argmaxBuf.contents().bindMemory(to: UInt32.self, capacity: 1)
    let totalMs = (CFAbsoluteTimeGetCurrent() - tStart) * 1000
    return (Int(result[0]), totalMs)
}

// ============================================================================
// MARK: - Load Prompt from File (or fallback to Hello token)
// ============================================================================

var promptTokens: [Int] = []
var maxGenerate = 400

// Accept prompt path as command-line argument, or use default
let promptPath: String
if CommandLine.arguments.count > 1 {
    promptPath = CommandLine.arguments[1]
} else {
    promptPath = "/Users/midas/Desktop/cowork/inference-across-metal/overnight_prompt_tokens.json"
}

if FileManager.default.fileExists(atPath: promptPath) {
    let data = try! Data(contentsOf: URL(fileURLWithPath: promptPath))
    let obj = try! JSONSerialization.jsonObject(with: data) as! [String: Any]
    promptTokens = (obj["tokens"] as! [Any]).map { ($0 as! NSNumber).intValue }
    maxGenerate = (obj["max_generate"] as? NSNumber)?.intValue ?? 400
    print("\nLoaded prompt: \(promptTokens.count) tokens, max_generate=\(maxGenerate)")
    print("Source: \(obj["source"] as? String ?? "unknown")")
} else {
    promptTokens = [9419]
    maxGenerate = 50
    print("\nNo prompt file found, using Hello token")
}

// Qwen3.5 EOS token IDs
let EOS_IM_END: Int = 248046      // <|im_end|>
let EOS_ENDOFTEXT: Int = 248044   // <|endoftext|>

// ============================================================================
// MARK: - Generation
// ============================================================================

print("\n" + String(repeating: "=", count: 60))
print("WARM PATH Generation — Qwen3.5-27B on M5 Air 16GB")
print("Strategy: mmap + MADV_WILLNEED prefetch + per-layer streaming")
print("Prompt: \(promptTokens.count) tokens, Generate: \(maxGenerate) tokens")
print("Estimated time: \((promptTokens.count + maxGenerate) * 8 / 60) min")
print(String(repeating: "=", count: 60))

var generatedTokens: [Int] = []
let tGenStart = CFAbsoluteTimeGetCurrent()

// --- Phase 1: Prefill (process prompt tokens) ---
var lastPrefillToken: Int = -1
if promptTokens.count > 1 {
    print("\nPhase 1: Prefill (\(promptTokens.count) tokens)...")
    for (i, token) in promptTokens.enumerated() {
        let position = i
        let (nextToken, stepMs) = warmForwardPass(tokenId: token, position: position)
        lastPrefillToken = nextToken

        if i < 3 || i == promptTokens.count - 1 || i % 50 == 0 {
            let elapsed = CFAbsoluteTimeGetCurrent() - tGenStart
            let pct = Double(i + 1) / Double(promptTokens.count) * 100
            print("  Prefill \(String(format: "%3d", i+1))/\(promptTokens.count) " +
                  "(\(String(format: "%.0f", pct))%) " +
                  "[\(String(format: "%.1f", stepMs))ms, " +
                  "\(String(format: "%.0f", elapsed))s elapsed]")
        }
    }
    let prefillTime = CFAbsoluteTimeGetCurrent() - tGenStart
    print("  Prefill complete: \(String(format: "%.0f", prefillTime))s " +
          "(\(String(format: "%.1f", prefillTime / 60))min)")
}

// --- Phase 2: Generate ---
print("\nPhase 2: Generate (\(maxGenerate) tokens)...")
let tGenPhase = CFAbsoluteTimeGetCurrent()

var currentToken: Int
if promptTokens.count > 1 {
    currentToken = lastPrefillToken
} else {
    let (firstGen, firstMs) = warmForwardPass(tokenId: promptTokens[0], position: 0, verbose: true)
    if firstGen < 0 { print("FATAL"); exit(1) }
    currentToken = firstGen
    print("  Token 1: \(firstGen) [\(String(format: "%.1f", firstMs))ms]")
}
generatedTokens.append(currentToken)

// Check if the very first generated token is EOS
var earlyEOS = false
if currentToken == EOS_IM_END || currentToken == EOS_ENDOFTEXT {
    print("  EOS on first token (\(currentToken)). Nothing to generate.")
    earlyEOS = true
}

let genStartPos = promptTokens.count

for step in 1..<maxGenerate where !earlyEOS {
    let position = genStartPos + step - 1
    if position >= MAX_SEQ - 1 { print("  KV cache full at position \(position)"); break }
    let (nextToken, stepMs) = warmForwardPass(tokenId: currentToken, position: position)
    if nextToken < 0 { print("ERROR at step \(step)"); break }
    generatedTokens.append(nextToken)
    currentToken = nextToken
    let elapsed = CFAbsoluteTimeGetCurrent() - tGenPhase
    let tokPerSec = Double(step + 1) / elapsed
    print("  Gen \(String(format: "%3d", step+1))/\(maxGenerate): " +
          "token \(String(format: "%6d", nextToken)) " +
          "[\(String(format: "%.1f", stepMs))ms, \(String(format: "%.3f", tokPerSec)) tok/s]")
    // Qwen3.5 EOS: 248046 = <|im_end|>, 248044 = <|endoftext|>
    if nextToken == EOS_IM_END || nextToken == EOS_ENDOFTEXT {
        print("  EOS detected (token \(nextToken)). Generation complete.")
        break
    }
}

let tGenTotal = CFAbsoluteTimeGetCurrent() - tGenStart
let genPhaseTime = CFAbsoluteTimeGetCurrent() - tGenPhase
let prefillTime = tGenPhase - tGenStart

print("\n" + String(repeating: "=", count: 60))
print("GENERATION COMPLETE")
print(String(repeating: "=", count: 60))
print("Prompt tokens:    \(promptTokens.count)")
print("Generated tokens: \(generatedTokens.count)")
print("Total time:       \(String(format: "%.0f", tGenTotal))s (\(String(format: "%.1f", tGenTotal / 60))min)")
if promptTokens.count > 1 {
    print("  Prefill:        \(String(format: "%.0f", prefillTime))s")
    print("  Generation:     \(String(format: "%.0f", genPhaseTime))s")
}
print("Gen tok/s:        \(String(format: "%.4f", Double(generatedTokens.count) / genPhaseTime))")

// Determine output path: use --output arg or derive from prompt path
let basePath = "/Users/midas/Desktop/cowork/inference-across-metal"
let resultPath: String
let outputTokensPath: String
if let outputIdx = CommandLine.arguments.firstIndex(of: "--output"), outputIdx + 1 < CommandLine.arguments.count {
    resultPath = CommandLine.arguments[outputIdx + 1]
    outputTokensPath = resultPath.replacingOccurrences(of: ".json", with: "_tokens.json")
} else {
    resultPath = basePath + "/warm_result.json"
    outputTokensPath = basePath + "/warm_output_tokens.json"
}

// Check if generation ended with EOS
let hitEOS = !generatedTokens.isEmpty &&
    (generatedTokens.last == EOS_IM_END || generatedTokens.last == EOS_ENDOFTEXT)

// Strip EOS token from output for cleaner decoding
var outputTokens = generatedTokens
if hitEOS { outputTokens.removeLast() }

let resultDict: [String: Any] = [
    "mode": "warm_path_prefetch",
    "prompt_tokens": promptTokens.count,
    "generated_tokens": generatedTokens.count,
    "output_tokens": outputTokens.count,
    "hit_eos": hitEOS,
    "total_time_s": tGenTotal,
    "prefill_time_s": prefillTime,
    "gen_time_s": genPhaseTime,
    "avg_tok_per_s": Double(generatedTokens.count) / genPhaseTime,
    "avg_ms_per_token": genPhaseTime * 1000 / Double(generatedTokens.count),
    "generated_token_ids": generatedTokens,
]
let resultData = try! JSONSerialization.data(withJSONObject: resultDict, options: .prettyPrinted)
try! resultData.write(to: URL(fileURLWithPath: resultPath))
print("Results saved to: \(resultPath)")

// Save output token IDs separately for easy Python decoding
let tokensDict: [String: Any] = [
    "tokens": outputTokens,
    "hit_eos": hitEOS,
    "prompt_tokens": promptTokens.count,
    "generated_tokens": outputTokens.count,
]
let tokensData = try! JSONSerialization.data(withJSONObject: tokensDict, options: .prettyPrinted)
try! tokensData.write(to: URL(fileURLWithPath: outputTokensPath))
print("Output tokens saved to: \(outputTokensPath)")

print("\nDecode with:")
print("  python3 \(basePath)/tokenize_prompt.py --decode \(outputTokensPath)")