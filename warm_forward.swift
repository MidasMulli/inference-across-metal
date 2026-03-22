// warm_forward.swift — Warm-path weight wiring for Qwen3.5-27B
// Pre-loads ALL 64 transformer layers + lm_head into Metal shared buffers at startup.
// Eliminates per-token SSD page faults that caused 14-16s/token in the cold path.
//
// Memory budget (16GB M5 Air):
//   64 layers: ~13,068 MB (48 GDN @ 205.7 MB + 16 Attn @ 199.7 MB)
//   lm_head:   ~682 MB
//   GDN state: ~149 MB (conv + delta)
//   KV cache:  ~64 MB (16 layers x 1024 tokens)
//   Activations: ~100 MB
//   Total:     ~14,063 MB → fits in 16GB with ~1.5GB for OS
//
//   Embedding (682 MB) is NOT pre-loaded during decode.
//   Instead: targeted 10KB memcpy per token from mmap (token_id * row_size).
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

// Layer type: every 4th layer (3, 7, 11, ..., 63) is full attention
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
    uint packed_dim = hidden_dim / 8;
    uint groups_per_row = hidden_dim / group_sz;
    uint pack_idx = tid / 8;
    uint nibble_idx = tid % 8;
    uint packed_val = weight[token_id * packed_dim + pack_idx];
    uint nibble = (packed_val >> (nibble_idx * 4)) & 0xF;
    uint group_idx = tid / group_sz;
    float scale = float(scales[token_id * groups_per_row + group_idx]);
    float bias = float(biases[token_id * groups_per_row + group_idx]);
    out[tid] = bfloat(float(nibble) * scale + bias);
}

// CPU-side embedding lookup for warm path (writes bf16 directly)
// This kernel dequantizes a single row from small buffers holding just one row
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

// Copy bf16 buffer (for KV cache append)
kernel void copy_bf16(
    device const bfloat* src [[buffer(0)]],
    device bfloat* dst       [[buffer(1)]],
    constant uint& count     [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= count) return;
    dst[tid] = src[tid];
}

// Scaled dot-product attention for M=1 decode with KV cache
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

// Load metallib (NAX kernel)
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

// Custom kernels
let customLib = try! device.makeLibrary(source: metalSource, options: nil)
func makePSO(_ name: String) -> MTLComputePipelineState {
    let fn = customLib.makeFunction(name: name)!
    return try! device.makeComputePipelineState(function: fn)
}
let rmsNormPSO = makePSO("rms_norm_bf16")
let siluMulPSO = makePSO("silu_multiply_bf16")
let residualPSO = makePSO("residual_add_bf16")
let embedPSO = makePSO("embed_lookup_bf16")
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
// MARK: - Embedding: mmap pointers for targeted row lookup (no full load)
// ============================================================================

let embedInfo = index["embed_tokens"] as! [String: Any]
let embedWeightInfo = parseTensorInfo(embedInfo["weight"] as! [String: Any])
let embedScalesInfo = parseTensorInfo(embedInfo["scales"] as! [String: Any])
let embedBiasesInfo = parseTensorInfo(embedInfo["biases"] as! [String: Any])

// Row sizes for targeted lookup
let embedPackedDim = HIDDEN_SIZE / 8  // 640 uint32s per row
let embedWeightRowBytes = embedPackedDim * 4  // 2560 bytes
let embedGroupsPerRow = HIDDEN_SIZE / GROUP_SIZE  // 80 groups per row
let embedScaleRowBytes = embedGroupsPerRow * 2  // 160 bytes (bf16)
let embedBiasRowBytes = embedGroupsPerRow * 2   // 160 bytes (bf16)
let embedTotalRowBytes = embedWeightRowBytes + embedScaleRowBytes + embedBiasRowBytes  // 2880 bytes

// Small Metal buffers for single-row embedding lookup during decode
let embedRowWBuf = device.makeBuffer(length: embedWeightRowBytes, options: .storageModeShared)!
let embedRowSBuf = device.makeBuffer(length: embedScaleRowBytes, options: .storageModeShared)!
let embedRowBBuf = device.makeBuffer(length: embedBiasRowBytes, options: .storageModeShared)!

print("\nEmbedding: targeted row lookup (\(embedTotalRowBytes) bytes/token, NOT pre-loaded)")

// Final norm (tiny, always loaded)
let finalNormInfo = index["final_norm"] as! [String: Any]
let finalNormW = loadTensorToBuffer(device, mmaps, parseTensorInfo(finalNormInfo["weight"] as! [String: Any]))
print("Final norm: \(finalNormW.length / 1024) KB")

// ============================================================================
// MARK: - WARM PATH: Pre-load ALL 64 layers + lm_head
// ============================================================================

// Memory budget: leave ~2.5GB for OS + GDN state (149MB) + KV cache (64MB) + activations (100MB)
// On 16GB: 16384 - 2500 - 149 - 64 - 100 = 13571 MB available for weights
// But Metal overhead and fragmentation eat ~500MB more, so budget 12800 MB for layers
// lm_head is streamed (682MB saved), embed is row-lookup (682MB saved)
let WEIGHT_BUDGET_MB: Int = 12800  // Conservative: leaves ~3.2GB for everything else

print("\n" + String(repeating: "=", count: 60))
print("WARM PATH: Pre-loading weights into Metal buffers")
print("Budget: \(WEIGHT_BUDGET_MB) MB for layer weights")
print(String(repeating: "=", count: 60))
let tWarmStart = CFAbsoluteTimeGetCurrent()

// Pre-allocated weight arrays indexed by layer position
var warmAttnLayers: [Int: AttnLayerWeights] = [:]
var warmGDNLayers: [Int: GDNLayerWeights] = [:]

var totalWiredBytes: Int = 0
var wiredLayerCount: Int = 0
var lastWiredLayer: Int = -1

for layerIdx in 0..<NUM_LAYERS {
    // Check budget before loading
    let li = layersInfo[layerIdx]
    let tensors = li["tensors"] as! [String: Any]
    var layerBytes = 0
    for (_, v) in tensors {
        let d = v as! [String: Any]
        layerBytes += d["byte_size"] as! Int
    }
    if (totalWiredBytes + layerBytes) / 1024 / 1024 > WEIGHT_BUDGET_MB {
        print("  Budget limit reached at layer \(layerIdx) " +
              "(\(totalWiredBytes / 1024 / 1024) MB wired, " +
              "\(layerBytes / 1024 / 1024) MB needed)")
        break
    }
    let tLayer = CFAbsoluteTimeGetCurrent()

    func loadT(_ name: String) -> MTLBuffer {
        let info = parseTensorInfo(tensors[name] as! [String: Any])
        totalWiredBytes += info.byteSize
        return loadTensorToBuffer(device, mmaps, info)
    }

    if isFullAttn(layerIdx) {
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
        warmAttnLayers[layerIdx] = lw
    } else {
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
        warmGDNLayers[layerIdx] = lw
    }

    wiredLayerCount += 1
    lastWiredLayer = layerIdx

    let layerMs = (CFAbsoluteTimeGetCurrent() - tLayer) * 1000
    let layerType = isFullAttn(layerIdx) ? "ATTN" : "GDN"
    let wiredMB = totalWiredBytes / 1024 / 1024
    if layerIdx < 3 || layerIdx == NUM_LAYERS - 1 || layerIdx % 16 == 0 {
        print("  Layer \(String(format: "%2d", layerIdx)) [\(layerType)]: " +
              "\(String(format: "%.0f", layerMs))ms  " +
              "(cumulative: \(wiredMB) MB)")
    } else if layerIdx == 3 {
        print("  ...")
    }
}

// lm_head: stream from mmap (saves 682MB — critical on 16GB)
let lmHeadInfo = index["lm_head"] as! [String: Any]
let lmHeadWInfo = parseTensorInfo(lmHeadInfo["weight"] as! [String: Any])
let lmHeadSInfo = parseTensorInfo(lmHeadInfo["scales"] as! [String: Any])
let lmHeadBInfo = parseTensorInfo(lmHeadInfo["biases"] as! [String: Any])
print("  lm_head: streamed from mmap (682 MB saved)")

let tWarmTotal = CFAbsoluteTimeGetCurrent() - tWarmStart
let coldLayerCount = NUM_LAYERS - wiredLayerCount
print("\nWarm wiring complete:")
print("  Total wired: \(totalWiredBytes / 1024 / 1024) MB")
print("  Time: \(String(format: "%.1f", tWarmTotal))s")
print("  Warm layers: \(wiredLayerCount)/\(NUM_LAYERS) (0-\(lastWiredLayer))")
print("  Cold layers: \(coldLayerCount) (\(lastWiredLayer+1)-\(NUM_LAYERS-1)) — streamed from mmap")
print("  lm_head: streamed from mmap (682 MB saved)")
print("  embed: row lookup from mmap (682 MB saved)")

// Cold-path layer loading (same as full_forward.swift)
func loadAttnLayerCold(_ layerIdx: Int) -> AttnLayerWeights {
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

func loadGDNLayerCold(_ layerIdx: Int) -> GDNLayerWeights {
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
// MARK: - Unmap safetensors (free virtual address space, weights are in Metal buffers now)
// ============================================================================
// We keep mmaps alive for embedding row lookups during decode.
// The mmap pages for layer weights should be evicted from RAM since Metal buffers
// now hold copies. The OS will reclaim those pages under memory pressure.

// ============================================================================
// MARK: - Activation Buffers
// ============================================================================

let hiddenBuf = device.makeBuffer(length: HIDDEN_SIZE * 2, options: .storageModeShared)!
let normedBuf = device.makeBuffer(length: HIDDEN_SIZE * 2, options: .storageModeShared)!
let residualBuf = device.makeBuffer(length: HIDDEN_SIZE * 2, options: .storageModeShared)!

// Full attention buffers
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

// GDN buffers
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

// MLP buffers (shared between attn and GDN layers)
let gateProjBuf = device.makeBuffer(length: INTERMEDIATE_SIZE * 2, options: .storageModeShared)!
let upProjBuf = device.makeBuffer(length: INTERMEDIATE_SIZE * 2, options: .storageModeShared)!
let mlpHiddenBuf = device.makeBuffer(length: INTERMEDIATE_SIZE * 2, options: .storageModeShared)!
let downProjBuf = device.makeBuffer(length: HIDDEN_SIZE * 2, options: .storageModeShared)!

// Logits + argmax
let logitsBuf = device.makeBuffer(length: VOCAB_SIZE * 2, options: .storageModeShared)!
let argmaxBuf = device.makeBuffer(length: 4, options: .storageModeShared)!

// RoPE cache
let ropeCosBuf = device.makeBuffer(length: ROTARY_DIM / 2 * 4, options: .storageModeShared)!
let ropeSinBuf = device.makeBuffer(length: ROTARY_DIM / 2 * 4, options: .storageModeShared)!

// ============================================================================
// MARK: - GDN Persistent State (48 layers)
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

let stateMB = (convStateA.length * 2 + deltaState.length) / 1024 / 1024
print("\nGDN state allocated: \(stateMB) MB (\(numGDNLayers) layers)")

var gdnLayerIndex: [Int: Int] = [:]
var gdnIdx = 0
for i in 0..<NUM_LAYERS {
    if !isFullAttn(i) {
        gdnLayerIndex[i] = gdnIdx
        gdnIdx += 1
    }
}

var attnLayerIndex: [Int: Int] = [:]
var attnIdx = 0
for i in 0..<NUM_LAYERS {
    if isFullAttn(i) {
        attnLayerIndex[i] = attnIdx
        attnIdx += 1
    }
}

// ============================================================================
// MARK: - KV Cache (16 attention layers)
// ============================================================================

let MAX_SEQ: Int = 1024
let kvTokenBytes = NUM_KV_HEADS * HEAD_DIM * 2
let kvLayerBytes = MAX_SEQ * kvTokenBytes
let numAttnLayers = 16

let kCache = device.makeBuffer(length: kvLayerBytes * numAttnLayers, options: .storageModeShared)!
let vCache = device.makeBuffer(length: kvLayerBytes * numAttnLayers, options: .storageModeShared)!
memset(kCache.contents(), 0, kCache.length)
memset(vCache.contents(), 0, vCache.length)
print("KV cache: \(kCache.length * 2 / 1024) KB (\(numAttnLayers) layers x \(MAX_SEQ) tokens)")

// ============================================================================
// MARK: - Helper: dispatch quantized matmul
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

// ============================================================================
// MARK: - Helper: RoPE precompute
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
// MARK: - Helper: Targeted embedding lookup from mmap
// ============================================================================

func embedFromMmap(tokenId: Int) {
    // Copy just one row of weight/scales/biases from mmap into small Metal buffers
    let wSrc = mmaps[embedWeightInfo.fileIdx] + embedWeightInfo.byteOffset + tokenId * embedWeightRowBytes
    let sSrc = mmaps[embedScalesInfo.fileIdx] + embedScalesInfo.byteOffset + tokenId * embedScaleRowBytes
    let bSrc = mmaps[embedBiasesInfo.fileIdx] + embedBiasesInfo.byteOffset + tokenId * embedBiasRowBytes

    memcpy(embedRowWBuf.contents(), wSrc, embedWeightRowBytes)
    memcpy(embedRowSBuf.contents(), sSrc, embedScaleRowBytes)
    memcpy(embedRowBBuf.contents(), bSrc, embedBiasRowBytes)
}

// ============================================================================
// MARK: - Dispatch: Full Attention Layer (uses pre-wired weights)
// ============================================================================

func dispatchAttnLayer(_ enc: MTLComputeCommandEncoder, _ lw: AttnLayerWeights,
                       attnIdx: Int, position: Int) {
    var dim = UInt32(HIDDEN_SIZE)
    var eps = RMS_NORM_EPS
    var count = UInt32(HIDDEN_SIZE)

    // 1. Input RMS Norm
    enc.setComputePipelineState(rmsNormPSO)
    enc.setBuffer(hiddenBuf, offset: 0, index: 0)
    enc.setBuffer(lw.inputNormWeight!, offset: 0, index: 1)
    enc.setBuffer(normedBuf, offset: 0, index: 2)
    enc.setBytes(&dim, length: 4, index: 3)
    enc.setBytes(&eps, length: 4, index: 4)
    enc.dispatchThreads(MTLSize(width: 256, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

    // 2. Q projection
    dispatchQMatmul(enc, lw.qProjW!, lw.qProjS!, lw.qProjB!,
                    normedBuf, qBuf,
                    K_dim: HIDDEN_SIZE, N_dim: NUM_HEADS * HEAD_DIM * 2, M_dim: 1)

    // 2b. Split Q+gate
    enc.setComputePipelineState(splitQGatePSO)
    enc.setBuffer(qBuf, offset: 0, index: 0)
    enc.setBuffer(queriesBuf, offset: 0, index: 1)
    enc.setBuffer(attnGateBuf, offset: 0, index: 2)
    var nh = UInt32(NUM_HEADS)
    var hd = UInt32(HEAD_DIM)
    enc.setBytes(&nh, length: 4, index: 3)
    enc.setBytes(&hd, length: 4, index: 4)
    enc.dispatchThreads(MTLSize(width: HEAD_DIM, height: NUM_HEADS, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

    // 3. K projection
    dispatchQMatmul(enc, lw.kProjW!, lw.kProjS!, lw.kProjB!,
                    normedBuf, kBuf,
                    K_dim: HIDDEN_SIZE, N_dim: NUM_KV_HEADS * HEAD_DIM, M_dim: 1)

    // 4. V projection
    dispatchQMatmul(enc, lw.vProjW!, lw.vProjS!, lw.vProjB!,
                    normedBuf, vBuf,
                    K_dim: HIDDEN_SIZE, N_dim: NUM_KV_HEADS * HEAD_DIM, M_dim: 1)

    // 5. Per-head QK norm
    enc.setComputePipelineState(perHeadRmsNormPSO)
    enc.setBuffer(queriesBuf, offset: 0, index: 0)
    enc.setBuffer(lw.qNormWeight!, offset: 0, index: 1)
    enc.setBuffer(qNormedBuf, offset: 0, index: 2)
    enc.setBytes(&nh, length: 4, index: 3)
    enc.setBytes(&hd, length: 4, index: 4)
    enc.setBytes(&eps, length: 4, index: 5)
    enc.dispatchThreads(MTLSize(width: HEAD_DIM, height: NUM_HEADS, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: HEAD_DIM, height: 1, depth: 1))

    var nkv = UInt32(NUM_KV_HEADS)
    enc.setBuffer(kBuf, offset: 0, index: 0)
    enc.setBuffer(lw.kNormWeight!, offset: 0, index: 1)
    enc.setBuffer(kNormedBuf, offset: 0, index: 2)
    enc.setBytes(&nkv, length: 4, index: 3)
    enc.setBytes(&hd, length: 4, index: 4)
    enc.setBytes(&eps, length: 4, index: 5)
    enc.dispatchThreads(MTLSize(width: HEAD_DIM, height: NUM_KV_HEADS, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: HEAD_DIM, height: 1, depth: 1))

    // 6. RoPE
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

    enc.setBuffer(kNormedBuf, offset: 0, index: 0)
    enc.setBytes(&nkv, length: 4, index: 3)
    enc.dispatchThreads(MTLSize(width: ROTARY_DIM / 2, height: NUM_KV_HEADS, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))

    // 7a. Append K to cache
    let cacheLayerOffset = attnIdx * kvLayerBytes
    let cacheTokenOffset = cacheLayerOffset + position * kvTokenBytes
    enc.setComputePipelineState(copyPSO)
    enc.setBuffer(kNormedBuf, offset: 0, index: 0)
    enc.setBuffer(kCache, offset: cacheTokenOffset, index: 1)
    var kvCount = UInt32(NUM_KV_HEADS * HEAD_DIM)
    enc.setBytes(&kvCount, length: 4, index: 2)
    enc.dispatchThreads(MTLSize(width: NUM_KV_HEADS * HEAD_DIM, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

    // 7b. Append V to cache
    enc.setBuffer(vBuf, offset: 0, index: 0)
    enc.setBuffer(vCache, offset: cacheTokenOffset, index: 1)
    enc.dispatchThreads(MTLSize(width: NUM_KV_HEADS * HEAD_DIM, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

    // 7c. SDPA
    enc.setComputePipelineState(sdpaPSO)
    enc.setBuffer(qNormedBuf, offset: 0, index: 0)
    enc.setBuffer(kCache, offset: cacheLayerOffset, index: 1)
    enc.setBuffer(vCache, offset: cacheLayerOffset, index: 2)
    enc.setBuffer(attnOutBuf, offset: 0, index: 3)
    enc.setBytes(&nh, length: 4, index: 4)
    enc.setBytes(&nkv, length: 4, index: 5)
    enc.setBytes(&hd, length: 4, index: 6)
    var seqLen = UInt32(position + 1)
    enc.setBytes(&seqLen, length: 4, index: 7)
    enc.dispatchThreads(MTLSize(width: NUM_HEADS, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: NUM_HEADS, height: 1, depth: 1))

    // 8. Output gate
    enc.setComputePipelineState(sigmoidMulPSO)
    enc.setBuffer(attnOutBuf, offset: 0, index: 0)
    enc.setBuffer(attnGateBuf, offset: 0, index: 1)
    enc.setBuffer(gatedBuf, offset: 0, index: 2)
    var gateCount = UInt32(NUM_HEADS * HEAD_DIM)
    enc.setBytes(&gateCount, length: 4, index: 3)
    enc.dispatchThreads(MTLSize(width: NUM_HEADS * HEAD_DIM, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

    // 9. O projection
    dispatchQMatmul(enc, lw.oProjW!, lw.oProjS!, lw.oProjB!,
                    gatedBuf, oOutBuf,
                    K_dim: NUM_HEADS * HEAD_DIM, N_dim: HIDDEN_SIZE, M_dim: 1)

    // 10. Residual
    enc.setComputePipelineState(residualPSO)
    enc.setBuffer(hiddenBuf, offset: 0, index: 0)
    enc.setBuffer(oOutBuf, offset: 0, index: 1)
    enc.setBuffer(residualBuf, offset: 0, index: 2)
    enc.setBytes(&count, length: 4, index: 3)
    enc.dispatchThreads(MTLSize(width: HIDDEN_SIZE, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

    // 11. Post-attention norm
    enc.setComputePipelineState(rmsNormPSO)
    enc.setBuffer(residualBuf, offset: 0, index: 0)
    enc.setBuffer(lw.postNormWeight!, offset: 0, index: 1)
    enc.setBuffer(normedBuf, offset: 0, index: 2)
    enc.setBytes(&dim, length: 4, index: 3)
    enc.setBytes(&eps, length: 4, index: 4)
    enc.dispatchThreads(MTLSize(width: 256, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

    // 12-15. MLP
    dispatchMLP(enc, lw.gateProjW!, lw.gateProjS!, lw.gateProjB!,
                lw.upProjW!, lw.upProjS!, lw.upProjB!,
                lw.downProjW!, lw.downProjS!, lw.downProjB!)

    // 16. Final residual
    enc.setComputePipelineState(residualPSO)
    enc.setBuffer(residualBuf, offset: 0, index: 0)
    enc.setBuffer(downProjBuf, offset: 0, index: 1)
    enc.setBuffer(hiddenBuf, offset: 0, index: 2)
    enc.setBytes(&count, length: 4, index: 3)
    enc.dispatchThreads(MTLSize(width: HIDDEN_SIZE, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
}

// ============================================================================
// MARK: - Dispatch: GDN Layer (uses pre-wired weights)
// ============================================================================

func dispatchGDNLayer(_ enc: MTLComputeCommandEncoder, _ lw: GDNLayerWeights,
                      gdnIdx: Int, useConvA: Bool) {
    let convStateRead = useConvA ? convStateA : convStateB
    let convStateWrite = useConvA ? convStateB : convStateA
    let convOffset = gdnIdx * convStateBytes
    let deltaOffset = gdnIdx * deltaStateBytes

    var dim = UInt32(HIDDEN_SIZE)
    var eps = RMS_NORM_EPS
    var count = UInt32(HIDDEN_SIZE)

    // 1. Input norm
    enc.setComputePipelineState(rmsNormPSO)
    enc.setBuffer(hiddenBuf, offset: 0, index: 0)
    enc.setBuffer(lw.inputNormWeight!, offset: 0, index: 1)
    enc.setBuffer(normedBuf, offset: 0, index: 2)
    enc.setBytes(&dim, length: 4, index: 3)
    enc.setBytes(&eps, length: 4, index: 4)
    enc.dispatchThreads(MTLSize(width: 256, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

    // 2. Projections
    dispatchQMatmul(enc, lw.qkvProjW!, lw.qkvProjS!, lw.qkvProjB!,
                    normedBuf, qkvBuf, K_dim: HIDDEN_SIZE, N_dim: GDN_CONV_DIM, M_dim: 1)
    dispatchQMatmul(enc, lw.zProjW!, lw.zProjS!, lw.zProjB!,
                    normedBuf, zBuf, K_dim: HIDDEN_SIZE, N_dim: GDN_VALUE_DIM, M_dim: 1)
    dispatchQMatmul(enc, lw.bProjW!, lw.bProjS!, lw.bProjB!,
                    normedBuf, bBuf, K_dim: HIDDEN_SIZE, N_dim: GDN_NUM_V_HEADS, M_dim: 1)
    dispatchQMatmul(enc, lw.aProjW!, lw.aProjS!, lw.aProjB!,
                    normedBuf, aBuf, K_dim: HIDDEN_SIZE, N_dim: GDN_NUM_V_HEADS, M_dim: 1)

    // 3. Depthwise conv1d
    enc.setComputePipelineState(depthwiseConvPSO)
    enc.setBuffer(convStateRead, offset: convOffset, index: 0)
    enc.setBuffer(qkvBuf, offset: 0, index: 1)
    enc.setBuffer(lw.convWeight!, offset: 0, index: 2)
    enc.setBuffer(convOutBuf, offset: 0, index: 3)
    enc.setBuffer(convStateWrite, offset: convOffset, index: 4)
    var cdim = UInt32(GDN_CONV_DIM)
    enc.setBytes(&cdim, length: 4, index: 5)
    enc.dispatchThreads(MTLSize(width: GDN_CONV_DIM, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

    // 4. SiLU
    enc.setComputePipelineState(siluPSO)
    enc.setBuffer(convOutBuf, offset: 0, index: 0)
    enc.setBuffer(convSiluBuf, offset: 0, index: 1)
    enc.setBytes(&cdim, length: 4, index: 2)
    enc.dispatchThreads(MTLSize(width: GDN_CONV_DIM, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

    // 6. QK norm with scale
    enc.setComputePipelineState(perHeadRmsNormScaledPSO)
    enc.setBuffer(convSiluBuf, offset: 0, index: 0)
    enc.setBuffer(gdnQNormedBuf, offset: 0, index: 1)
    var nkh = UInt32(GDN_NUM_K_HEADS)
    var hkd = UInt32(GDN_HEAD_K_DIM)
    var normEps: Float = 1e-6
    enc.setBytes(&nkh, length: 4, index: 2)
    enc.setBytes(&hkd, length: 4, index: 3)
    enc.setBytes(&normEps, length: 4, index: 4)
    var qScale: Float = GDN_INV_SCALE * GDN_INV_SCALE
    enc.setBytes(&qScale, length: 4, index: 5)
    enc.dispatchThreads(MTLSize(width: GDN_HEAD_K_DIM, height: GDN_NUM_K_HEADS, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: GDN_HEAD_K_DIM, height: 1, depth: 1))

    // K norm
    enc.setBuffer(convSiluBuf, offset: GDN_KEY_DIM * 2, index: 0)
    enc.setBuffer(gdnKNormedBuf, offset: 0, index: 1)
    var kScale: Float = GDN_INV_SCALE
    enc.setBytes(&kScale, length: 4, index: 5)
    enc.dispatchThreads(MTLSize(width: GDN_HEAD_K_DIM, height: GDN_NUM_K_HEADS, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: GDN_HEAD_K_DIM, height: 1, depth: 1))

    // 7. Expand K heads
    var nvh = UInt32(GDN_NUM_V_HEADS)
    enc.setComputePipelineState(expandKvPSO)
    enc.setBuffer(gdnQNormedBuf, offset: 0, index: 0)
    enc.setBuffer(gdnQExpandedBuf, offset: 0, index: 1)
    enc.setBytes(&nvh, length: 4, index: 2)
    enc.setBytes(&nkh, length: 4, index: 3)
    enc.setBytes(&hkd, length: 4, index: 4)
    enc.dispatchThreads(MTLSize(width: GDN_HEAD_K_DIM, height: GDN_NUM_V_HEADS, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))

    enc.setBuffer(gdnKNormedBuf, offset: 0, index: 0)
    enc.setBuffer(gdnKExpandedBuf, offset: 0, index: 1)
    enc.dispatchThreads(MTLSize(width: GDN_HEAD_K_DIM, height: GDN_NUM_V_HEADS, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))

    // 8. Compute g and beta
    enc.setComputePipelineState(computeGBetaPSO)
    enc.setBuffer(aBuf, offset: 0, index: 0)
    enc.setBuffer(bBuf, offset: 0, index: 1)
    enc.setBuffer(lw.ALog!, offset: 0, index: 2)
    enc.setBuffer(lw.dtBias!, offset: 0, index: 3)
    enc.setBuffer(gBuf, offset: 0, index: 4)
    enc.setBuffer(betaBuf, offset: 0, index: 5)
    enc.setBytes(&nvh, length: 4, index: 6)
    enc.dispatchThreads(MTLSize(width: GDN_NUM_V_HEADS, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 48, height: 1, depth: 1))

    // 9. Gated delta step
    enc.setComputePipelineState(gdnStepPSO)
    enc.setBuffer(gdnQExpandedBuf, offset: 0, index: 0)
    enc.setBuffer(gdnKExpandedBuf, offset: 0, index: 1)
    enc.setBuffer(convSiluBuf, offset: GDN_KEY_DIM * 2 * 2, index: 2)
    enc.setBuffer(gBuf, offset: 0, index: 3)
    enc.setBuffer(betaBuf, offset: 0, index: 4)
    enc.setBuffer(deltaState, offset: deltaOffset, index: 5)
    enc.setBuffer(gdnYBuf, offset: 0, index: 6)
    var hvd = UInt32(GDN_HEAD_V_DIM)
    enc.setBytes(&nvh, length: 4, index: 7)
    enc.setBytes(&hvd, length: 4, index: 8)
    enc.setBytes(&hkd, length: 4, index: 9)
    enc.dispatchThreads(MTLSize(width: GDN_HEAD_V_DIM, height: GDN_NUM_V_HEADS, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))

    // 10. RMSNormGated
    enc.setComputePipelineState(rmsNormGatedPSO)
    enc.setBuffer(gdnYBuf, offset: 0, index: 0)
    enc.setBuffer(zBuf, offset: 0, index: 1)
    enc.setBuffer(lw.normWeight!, offset: 0, index: 2)
    enc.setBuffer(gdnNormedBuf, offset: 0, index: 3)
    enc.setBytes(&nvh, length: 4, index: 4)
    enc.setBytes(&hvd, length: 4, index: 5)
    enc.setBytes(&eps, length: 4, index: 6)
    enc.dispatchThreads(MTLSize(width: GDN_HEAD_V_DIM, height: GDN_NUM_V_HEADS, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))

    // 11. Out projection
    dispatchQMatmul(enc, lw.outProjW!, lw.outProjS!, lw.outProjB!,
                    gdnNormedBuf, gdnOutProjBuf, K_dim: GDN_VALUE_DIM, N_dim: HIDDEN_SIZE, M_dim: 1)

    // 12. Residual
    enc.setComputePipelineState(residualPSO)
    enc.setBuffer(hiddenBuf, offset: 0, index: 0)
    enc.setBuffer(gdnOutProjBuf, offset: 0, index: 1)
    enc.setBuffer(residualBuf, offset: 0, index: 2)
    enc.setBytes(&count, length: 4, index: 3)
    enc.dispatchThreads(MTLSize(width: HIDDEN_SIZE, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

    // 13. Post-attention norm
    enc.setComputePipelineState(rmsNormPSO)
    enc.setBuffer(residualBuf, offset: 0, index: 0)
    enc.setBuffer(lw.postNormWeight!, offset: 0, index: 1)
    enc.setBuffer(normedBuf, offset: 0, index: 2)
    enc.setBytes(&dim, length: 4, index: 3)
    enc.setBytes(&eps, length: 4, index: 4)
    enc.dispatchThreads(MTLSize(width: 256, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

    // 14. MLP
    dispatchMLP(enc, lw.gateProjW!, lw.gateProjS!, lw.gateProjB!,
                lw.upProjW!, lw.upProjS!, lw.upProjB!,
                lw.downProjW!, lw.downProjS!, lw.downProjB!)

    // 15. Final residual
    enc.setComputePipelineState(residualPSO)
    enc.setBuffer(residualBuf, offset: 0, index: 0)
    enc.setBuffer(downProjBuf, offset: 0, index: 1)
    enc.setBuffer(hiddenBuf, offset: 0, index: 2)
    enc.setBytes(&count, length: 4, index: 3)
    enc.dispatchThreads(MTLSize(width: HIDDEN_SIZE, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
}

// ============================================================================
// MARK: - Dispatch: MLP (shared between attn and GDN)
// ============================================================================

func dispatchMLP(_ enc: MTLComputeCommandEncoder,
                 _ gateW: MTLBuffer, _ gateS: MTLBuffer, _ gateB: MTLBuffer,
                 _ upW: MTLBuffer, _ upS: MTLBuffer, _ upB: MTLBuffer,
                 _ downW: MTLBuffer, _ downS: MTLBuffer, _ downB: MTLBuffer) {
    dispatchQMatmul(enc, gateW, gateS, gateB,
                    normedBuf, gateProjBuf,
                    K_dim: HIDDEN_SIZE, N_dim: INTERMEDIATE_SIZE, M_dim: 1)
    dispatchQMatmul(enc, upW, upS, upB,
                    normedBuf, upProjBuf,
                    K_dim: HIDDEN_SIZE, N_dim: INTERMEDIATE_SIZE, M_dim: 1)

    enc.setComputePipelineState(siluMulPSO)
    enc.setBuffer(gateProjBuf, offset: 0, index: 0)
    enc.setBuffer(upProjBuf, offset: 0, index: 1)
    enc.setBuffer(mlpHiddenBuf, offset: 0, index: 2)
    var isize = UInt32(INTERMEDIATE_SIZE)
    enc.setBytes(&isize, length: 4, index: 3)
    enc.dispatchThreads(MTLSize(width: INTERMEDIATE_SIZE, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

    dispatchQMatmul(enc, downW, downS, downB,
                    mlpHiddenBuf, downProjBuf,
                    K_dim: INTERMEDIATE_SIZE, N_dim: HIDDEN_SIZE, M_dim: 1)
}

// ============================================================================
// MARK: - Warm Forward Pass (all weights pre-loaded, no per-layer mmap)
// ============================================================================

var convStateFlip = true

func warmForwardPass(tokenId: Int, position: Int, verbose: Bool = false) -> (tokenId: Int, timeMs: Double) {
    let tStart = CFAbsoluteTimeGetCurrent()
    precomputeRoPE(position: position)

    // --- Embedding (targeted row lookup from mmap, ~3KB) ---
    embedFromMmap(tokenId: tokenId)

    let cbEmbed = queue.makeCommandBuffer()!
    let encEmbed = cbEmbed.makeComputeCommandEncoder()!
    encEmbed.setComputePipelineState(embedRowPSO)
    encEmbed.setBuffer(embedRowWBuf, offset: 0, index: 0)
    encEmbed.setBuffer(embedRowSBuf, offset: 0, index: 1)
    encEmbed.setBuffer(embedRowBBuf, offset: 0, index: 2)
    encEmbed.setBuffer(hiddenBuf, offset: 0, index: 3)
    var hdim = UInt32(HIDDEN_SIZE)
    encEmbed.setBytes(&hdim, length: 4, index: 4)
    var gsz = UInt32(GROUP_SIZE)
    encEmbed.setBytes(&gsz, length: 4, index: 5)
    encEmbed.dispatchThreads(MTLSize(width: HIDDEN_SIZE, height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    encEmbed.endEncoding()
    cbEmbed.commit()
    cbEmbed.waitUntilCompleted()

    // --- 64 Layers (warm layers from Metal buffers, cold layers streamed from mmap) ---
    for layerIdx in 0..<NUM_LAYERS {
        let tLayer = CFAbsoluteTimeGetCurrent()

        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!

        let isWarm = layerIdx <= lastWiredLayer

        if isFullAttn(layerIdx) {
            let lw = isWarm ? warmAttnLayers[layerIdx]! : loadAttnLayerCold(layerIdx)
            dispatchAttnLayer(enc, lw, attnIdx: attnLayerIndex[layerIdx]!, position: position)
        } else {
            let gIdx = gdnLayerIndex[layerIdx]!
            let lw = isWarm ? warmGDNLayers[layerIdx]! : loadGDNLayerCold(layerIdx)
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
            let pathTag = isWarm ? "WARM" : "COLD"
            if layerIdx < 4 || layerIdx >= 60 || layerIdx % 16 == 0 {
                print("  Layer \(String(format: "%2d", layerIdx)) [\(layerType)] \(pathTag): \(String(format: "%.1f", layerMs))ms")
            } else if layerIdx == 4 {
                print("  ...")
            }
        }
    }

    convStateFlip = !convStateFlip

    // --- Final norm ---
    let cbFinal = queue.makeCommandBuffer()!
    let encFinal = cbFinal.makeComputeCommandEncoder()!

    encFinal.setComputePipelineState(rmsNormPSO)
    encFinal.setBuffer(hiddenBuf, offset: 0, index: 0)
    encFinal.setBuffer(finalNormW, offset: 0, index: 1)
    encFinal.setBuffer(normedBuf, offset: 0, index: 2)
    var dim2 = UInt32(HIDDEN_SIZE)
    encFinal.setBytes(&dim2, length: 4, index: 3)
    var eps2 = RMS_NORM_EPS
    encFinal.setBytes(&eps2, length: 4, index: 4)
    encFinal.dispatchThreads(MTLSize(width: 256, height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    encFinal.endEncoding()
    cbFinal.commit()
    cbFinal.waitUntilCompleted()

    // --- lm_head (streamed from mmap each token — 682MB, not worth pre-loading on 16GB) ---
    let lmW = loadTensorToBuffer(device, mmaps, lmHeadWInfo)
    let lmS = loadTensorToBuffer(device, mmaps, lmHeadSInfo)
    let lmB = loadTensorToBuffer(device, mmaps, lmHeadBInfo)

    let cbLm = queue.makeCommandBuffer()!
    let encLm = cbLm.makeComputeCommandEncoder()!
    dispatchQMatmul(encLm, lmW, lmS, lmB,
                    normedBuf, logitsBuf,
                    K_dim: HIDDEN_SIZE, N_dim: VOCAB_SIZE, M_dim: 1)

    // Argmax
    encLm.setComputePipelineState(argmaxPSO)
    encLm.setBuffer(logitsBuf, offset: 0, index: 0)
    encLm.setBuffer(argmaxBuf, offset: 0, index: 1)
    var vocabSize = UInt32(VOCAB_SIZE)
    encLm.setBytes(&vocabSize, length: 4, index: 2)
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
// MARK: - Autoregressive Generation
// ============================================================================

// Single "Hello" token — no prompt file needed for warm path test
let promptTokens: [Int] = [9419]
let maxGenerate = 50

print("\n" + String(repeating: "=", count: 60))
print("WARM PATH Generation — Qwen3.5-27B on M5 Air 16GB")
print("Prompt: token 9419 (Hello), Generate: \(maxGenerate) tokens")
print("All weights pre-loaded in Metal shared buffers")
print(String(repeating: "=", count: 60))

var generatedTokens: [Int] = []

let tGenStart = CFAbsoluteTimeGetCurrent()

// First token
print("\nGenerating (verbose first token)...")
let (firstGen, firstMs) = warmForwardPass(tokenId: promptTokens[0], position: 0, verbose: true)
if firstGen < 0 {
    print("FATAL: First token failed")
    exit(1)
}
generatedTokens.append(firstGen)
print("  Token 1: \(firstGen) [\(String(format: "%.1f", firstMs))ms]")

var currentToken = firstGen

for step in 1..<maxGenerate {
    let position = step

    if position >= MAX_SEQ - 1 {
        print("  KV cache full at position \(position)")
        break
    }

    let (nextToken, stepMs) = warmForwardPass(tokenId: currentToken, position: position)

    if nextToken < 0 {
        print("ERROR at step \(step)")
        break
    }

    generatedTokens.append(nextToken)
    currentToken = nextToken

    let elapsed = CFAbsoluteTimeGetCurrent() - tGenStart
    let tokPerSec = Double(step + 1) / elapsed

    print("  Gen \(String(format: "%3d", step+1))/\(maxGenerate): " +
          "token \(String(format: "%6d", nextToken))  " +
          "[\(String(format: "%.1f", stepMs))ms, " +
          "\(String(format: "%.2f", tokPerSec)) tok/s]")

    if nextToken == 151643 || nextToken == 151645 {
        print("  (EOS)")
        break
    }
}

let tGenTotal = CFAbsoluteTimeGetCurrent() - tGenStart

print("\n" + String(repeating: "=", count: 60))
print("WARM PATH GENERATION COMPLETE")
print(String(repeating: "=", count: 60))
print("Generated tokens: \(generatedTokens.count)")
print("Total gen time:   \(String(format: "%.1f", tGenTotal))s")
print("Avg tok/s:        \(String(format: "%.3f", Double(generatedTokens.count) / tGenTotal))")
print("Avg ms/token:     \(String(format: "%.1f", tGenTotal * 1000 / Double(generatedTokens.count)))")
print("Wiring overhead:  \(String(format: "%.1f", tWarmTotal))s (one-time)")

// Save results
let resultPath = "/Users/midas/Desktop/cowork/inference-across-metal/warm_result.json"
let resultDict: [String: Any] = [
    "mode": "warm_path",
    "generated_tokens": generatedTokens.count,
    "total_gen_time_s": tGenTotal,
    "avg_tok_per_s": Double(generatedTokens.count) / tGenTotal,
    "avg_ms_per_token": tGenTotal * 1000 / Double(generatedTokens.count),
    "wiring_time_s": tWarmTotal,
    "wired_mb": totalWiredBytes / 1024 / 1024,
    "generated_token_ids": generatedTokens,
]
let resultData = try! JSONSerialization.data(withJSONObject: resultDict, options: .prettyPrinted)
try! resultData.write(to: URL(fileURLWithPath: resultPath))
print("\nResults saved to: \(resultPath)")

print("\nGenerated token IDs: \(generatedTokens)")
print("\nDecode with:")
print("python3 -c \"from transformers import AutoTokenizer; " +
      "t=AutoTokenizer.from_pretrained('/Users/midas/models/Qwen3.5-27B-MLX-4bit'); " +
      "print(t.decode(\(generatedTokens)))\"")

print("\nDone.")
