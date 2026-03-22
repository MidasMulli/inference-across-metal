// gdn_forward.swift — Session 2: GDN (GatedDeltaNet) M=1 decode kernels
// Tests one GDN layer against MLX reference, then wires into full 64-layer forward pass.
//
// Build: swiftc -O -framework Metal -framework MetalPerformanceShaders gdn_forward.swift -o gdn_forward
// Run:   ./gdn_forward

import Metal
import Foundation

// ============================================================================
// MARK: - Config
// ============================================================================

let HIDDEN_SIZE: Int = 5120
let INTERMEDIATE_SIZE: Int = 17408
let NUM_HEADS: Int = 24        // full attention heads
let NUM_KV_HEADS: Int = 4      // full attention KV heads
let HEAD_DIM: Int = 256         // full attention head dim
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
let GDN_KEY_DIM: Int = 2048     // 16 * 128
let GDN_VALUE_DIM: Int = 6144   // 48 * 128
let GDN_CONV_DIM: Int = 10240   // 2048*2 + 6144
let GDN_CONV_KERNEL: Int = 4
let GDN_KV_REPEAT: Int = 3      // 48 / 16
let GDN_INV_SCALE: Float = 0.08838834764  // 1/sqrt(128)
let GDN_STATE_SIZE: Int = 48 * 128 * 128  // per layer, in elements

// Layer type pattern
let FULL_ATTN_LAYERS = stride(from: 3, to: 64, by: 4).map { $0 }
let GDN_LAYERS = (0..<64).filter { ($0 + 1) % 4 != 0 }

// ============================================================================
// MARK: - Metal Kernel Source
// ============================================================================

let metalSource = """
#include <metal_stdlib>
using namespace metal;

// === Shared kernels (from Session 1) ===

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

// === NEW: GDN-specific kernels ===

// Depthwise Conv1d for M=1 decode
// conv_state: (3, conv_dim) — last 3 timesteps
// new_input: (conv_dim) — current timestep qkv
// conv_weight: (conv_dim, 4, 1) — depthwise weights
// out: (conv_dim) — conv output (before silu)
// new_state: (3, conv_dim) — updated state (last 3 of [state, new_input])
kernel void depthwise_conv1d_m1_bf16(
    device const bfloat* conv_state   [[buffer(0)]],  // (3, conv_dim)
    device const bfloat* new_input    [[buffer(1)]],  // (conv_dim)
    device const bfloat* conv_weight  [[buffer(2)]],  // (conv_dim, 4, 1)
    device bfloat* out                [[buffer(3)]],  // (conv_dim)
    device bfloat* new_state_out      [[buffer(4)]],  // (3, conv_dim)
    constant uint& conv_dim           [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= conv_dim) return;

    // Build the 4-element window for this channel: [state[0], state[1], state[2], new_input]
    float window[4];
    window[0] = float(conv_state[0 * conv_dim + tid]);
    window[1] = float(conv_state[1 * conv_dim + tid]);
    window[2] = float(conv_state[2 * conv_dim + tid]);
    window[3] = float(new_input[tid]);

    // Dot product with conv weights
    float sum = 0.0f;
    for (int i = 0; i < 4; i++) {
        sum += window[i] * float(conv_weight[tid * 4 + i]);
    }
    out[tid] = bfloat(sum);

    // Update state: shift left, new_input becomes last
    new_state_out[0 * conv_dim + tid] = conv_state[1 * conv_dim + tid];
    new_state_out[1 * conv_dim + tid] = conv_state[2 * conv_dim + tid];
    new_state_out[2 * conv_dim + tid] = new_input[tid];
}

// SiLU activation (standalone, for conv output)
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

// Per-head RMS norm WITHOUT weight, WITH scale factor
// For GDN QK norm: q = scale^2 * rms_norm(q), k = scale * rms_norm(k)
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

// Compute g = exp(-exp(A_log) * softplus(a + dt_bias))
// beta = sigmoid(b)
// All per-head scalars
kernel void compute_g_beta_bf16(
    device const bfloat* a       [[buffer(0)]],  // (num_v_heads)
    device const bfloat* b       [[buffer(1)]],  // (num_v_heads)
    device const float* A_log    [[buffer(2)]],  // (num_v_heads) float32
    device const bfloat* dt_bias [[buffer(3)]],  // (num_v_heads)
    device bfloat* g_out         [[buffer(4)]],  // (num_v_heads)
    device bfloat* beta_out      [[buffer(5)]],  // (num_v_heads)
    constant uint& num_v_heads   [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= num_v_heads) return;

    float a_val = float(a[tid]);
    float b_val = float(b[tid]);
    float A_val = A_log[tid];
    float dt_val = float(dt_bias[tid]);

    // softplus(a + dt_bias) = log(1 + exp(a + dt_bias))
    float sp_input = a_val + dt_val;
    float softplus = sp_input > 20.0f ? sp_input : log(1.0f + exp(sp_input));

    // g = exp(-exp(A_log) * softplus)
    float g = exp(-exp(A_val) * softplus);
    g_out[tid] = bfloat(g);

    // beta = sigmoid(b)
    beta_out[tid] = bfloat(1.0f / (1.0f + exp(-b_val)));
}

// Gated Delta Net M=1 step (the core recurrence)
// For each v_head: update state matrix and compute output
// State: (num_v_heads, head_v_dim, head_k_dim) in float32
// At M=1: single step, no time loop
//
// Algorithm per v_head h:
//   state[h] = g[h] * state[h]                     // decay
//   kv_mem = state[h] @ k_expanded[h]               // (Dv,) = (Dv, Dk) @ (Dk,)
//   delta = (v[h] - kv_mem) * beta[h]              // (Dv,)
//   state[h] += outer(delta, k_expanded[h])         // rank-1 update
//   y[h] = state[h] @ q_expanded[h]                 // (Dv,) = (Dv, Dk) @ (Dk,)
//
// Thread assignment: one thread per (v_head, dv_idx) — handles one row of state
kernel void gated_delta_step_bf16(
    device const bfloat* q       [[buffer(0)]],  // (num_v_heads, head_k_dim) — expanded
    device const bfloat* k       [[buffer(1)]],  // (num_v_heads, head_k_dim) — expanded
    device const bfloat* v       [[buffer(2)]],  // (num_v_heads, head_v_dim)
    device const bfloat* g       [[buffer(3)]],  // (num_v_heads)
    device const bfloat* beta    [[buffer(4)]],  // (num_v_heads)
    device float* state          [[buffer(5)]],  // (num_v_heads, head_v_dim, head_k_dim) float32
    device bfloat* y_out         [[buffer(6)]],  // (num_v_heads, head_v_dim)
    constant uint& num_v_heads   [[buffer(7)]],
    constant uint& head_v_dim    [[buffer(8)]],
    constant uint& head_k_dim    [[buffer(9)]],
    uint2 tid [[thread_position_in_grid]])
{
    uint vh = tid.y;   // v_head index
    uint dv = tid.x;   // dv_dim index (row of state matrix)
    if (vh >= num_v_heads || dv >= head_v_dim) return;

    float g_val = float(g[vh]);
    float beta_val = float(beta[vh]);
    float v_val = float(v[vh * head_v_dim + dv]);

    uint state_base = (vh * head_v_dim + dv) * head_k_dim;

    // Decay state row and compute kv_mem = state_row @ k
    float kv_mem = 0.0f;
    for (uint dk = 0; dk < head_k_dim; dk++) {
        float s = state[state_base + dk] * g_val;
        state[state_base + dk] = s;
        kv_mem += s * float(k[vh * head_k_dim + dk]);
    }

    // delta = (v - kv_mem) * beta
    float delta = (v_val - kv_mem) * beta_val;

    // Rank-1 update: state += outer(delta, k) and compute y = state @ q
    float y_val = 0.0f;
    for (uint dk = 0; dk < head_k_dim; dk++) {
        float new_s = state[state_base + dk] + float(k[vh * head_k_dim + dk]) * delta;
        state[state_base + dk] = new_s;
        y_val += new_s * float(q[vh * head_k_dim + dk]);
    }

    y_out[vh * head_v_dim + dv] = bfloat(y_val);
}

// RMSNormGated: output = silu(z) * rms_norm(x, weight)
// Per-head: x and z are (num_v_heads, head_v_dim)
kernel void rms_norm_gated_bf16(
    device const bfloat* x       [[buffer(0)]],  // (num_v_heads, head_v_dim)
    device const bfloat* z       [[buffer(1)]],  // (num_v_heads, head_v_dim)
    device const bfloat* weight  [[buffer(2)]],  // (head_v_dim)
    device bfloat* out           [[buffer(3)]],  // (num_v_heads, head_v_dim)
    constant uint& num_heads     [[buffer(4)]],
    constant uint& head_dim      [[buffer(5)]],
    constant float& eps          [[buffer(6)]],
    uint2 tid [[thread_position_in_grid]])
{
    uint head = tid.y;
    uint elem = tid.x;
    if (head >= num_heads || elem >= head_dim) return;

    uint base = head * head_dim;

    // RMS norm on x
    float sum_sq = 0.0f;
    for (uint i = 0; i < head_dim; i++) {
        float v = float(x[base + i]);
        sum_sq += v * v;
    }
    float rms = sqrt(sum_sq / float(head_dim) + eps);
    float x_normed = (float(x[base + elem]) / rms) * float(weight[elem]);

    // SiLU gate
    float z_val = float(z[base + elem]);
    float silu_z = z_val / (1.0f + exp(-z_val));

    out[base + elem] = bfloat(silu_z * x_normed);
}
"""

// ============================================================================
// MARK: - Safetensors & Weight Loading (same as Session 1)
// ============================================================================

struct TensorInfo {
    let fileIdx: Int
    let byteOffset: Int
    let byteSize: Int
    let shape: [Int]
    let dtype: String
}

struct FullAttnLayerWeights {
    var inputNormWeight: MTLBuffer?
    var qProjW: MTLBuffer?, qProjS: MTLBuffer?, qProjB: MTLBuffer?
    var kProjW: MTLBuffer?, kProjS: MTLBuffer?, kProjB: MTLBuffer?
    var vProjW: MTLBuffer?, vProjS: MTLBuffer?, vProjB: MTLBuffer?
    var oProjW: MTLBuffer?, oProjS: MTLBuffer?, oProjB: MTLBuffer?
    var qNormWeight: MTLBuffer?
    var kNormWeight: MTLBuffer?
    var postNormWeight: MTLBuffer?
    var gateProjW: MTLBuffer?, gateProjS: MTLBuffer?, gateProjB: MTLBuffer?
    var upProjW: MTLBuffer?, upProjS: MTLBuffer?, upProjB: MTLBuffer?
    var downProjW: MTLBuffer?, downProjS: MTLBuffer?, downProjB: MTLBuffer?
}

struct GDNLayerWeights {
    var inputNormWeight: MTLBuffer?
    // GDN projections
    var qkvProjW: MTLBuffer?, qkvProjS: MTLBuffer?, qkvProjB: MTLBuffer?
    var zProjW: MTLBuffer?, zProjS: MTLBuffer?, zProjB: MTLBuffer?
    var bProjW: MTLBuffer?, bProjS: MTLBuffer?, bProjB: MTLBuffer?
    var aProjW: MTLBuffer?, aProjS: MTLBuffer?, aProjB: MTLBuffer?
    // Conv1d
    var convWeight: MTLBuffer?
    // Delta net params
    var ALog: MTLBuffer?    // float32
    var dtBias: MTLBuffer?
    // Norm
    var normWeight: MTLBuffer?
    // Output
    var outProjW: MTLBuffer?, outProjS: MTLBuffer?, outProjB: MTLBuffer?
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

// Load metallib
let mlxMetalPath = String(cString: getenv("HOME")) + "/.mlx-env/lib/python3.11/site-packages/mlx/lib/mlx.metallib"
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
let sigmoidMulPSO = makePSO("sigmoid_multiply_bf16")
let ropePSO = makePSO("rope_bf16")
let argmaxPSO = makePSO("argmax_bf16")
let perHeadRmsNormPSO = makePSO("per_head_rms_norm_bf16")
let expandKvPSO = makePSO("expand_kv_heads_bf16")
let splitQGatePSO = makePSO("split_q_gate_bf16")
// GDN kernels
let depthwiseConvPSO = makePSO("depthwise_conv1d_m1_bf16")
let siluPSO = makePSO("silu_bf16")
let perHeadRmsNormScaledPSO = makePSO("per_head_rms_norm_scaled_bf16")
let computeGBetaPSO = makePSO("compute_g_beta_bf16")
let gdnStepPSO = makePSO("gated_delta_step_bf16")
let rmsNormGatedPSO = makePSO("rms_norm_gated_bf16")
print("All kernels compiled")

// Load weight index
let indexPath = "/Users/midas/Desktop/cowork/inference-across-metal/reference_data/27b_full_index.json"
let indexData = try! Data(contentsOf: URL(fileURLWithPath: indexPath))
let index = try! JSONSerialization.jsonObject(with: indexData) as! [String: Any]
let shardFiles = index["shard_files"] as! [String]
let layersInfo = index["layers"] as! [[String: Any]]

// mmap safetensors
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
// MARK: - Load GDN Layer 0 Weights (for verification)
// ============================================================================

print("\nLoading GDN layer 0 weights...")
let t0 = CFAbsoluteTimeGetCurrent()

// Embedding
let embedInfo = index["embed_tokens"] as! [String: Any]
let embedW = loadTensorToBuffer(device, mmaps, parseTensorInfo(embedInfo["weight"] as! [String: Any]))
let embedS = loadTensorToBuffer(device, mmaps, parseTensorInfo(embedInfo["scales"] as! [String: Any]))
let embedB = loadTensorToBuffer(device, mmaps, parseTensorInfo(embedInfo["biases"] as! [String: Any]))

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

let gdn0 = loadGDNLayer(0)
let tLoad = CFAbsoluteTimeGetCurrent() - t0
print("Loaded in \(String(format: "%.2f", tLoad))s")

// ============================================================================
// MARK: - Helper: dispatch quantized matmul
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
// MARK: - Activation Buffers
// ============================================================================

let hiddenBuf = device.makeBuffer(length: HIDDEN_SIZE * 2, options: .storageModeShared)!
let normedBuf = device.makeBuffer(length: HIDDEN_SIZE * 2, options: .storageModeShared)!
let residualBuf = device.makeBuffer(length: HIDDEN_SIZE * 2, options: .storageModeShared)!

// GDN buffers
let qkvBuf = device.makeBuffer(length: GDN_CONV_DIM * 2, options: .storageModeShared)!      // 10240 bf16
let zBuf = device.makeBuffer(length: GDN_VALUE_DIM * 2, options: .storageModeShared)!        // 6144 bf16
let aBuf = device.makeBuffer(length: GDN_NUM_V_HEADS * 2, options: .storageModeShared)!      // 48 bf16
let bBuf = device.makeBuffer(length: GDN_NUM_V_HEADS * 2, options: .storageModeShared)!      // 48 bf16
let convOutBuf = device.makeBuffer(length: GDN_CONV_DIM * 2, options: .storageModeShared)!   // 10240 bf16
let convSiluBuf = device.makeBuffer(length: GDN_CONV_DIM * 2, options: .storageModeShared)!  // after silu
let gdnQBuf = device.makeBuffer(length: GDN_KEY_DIM * 2, options: .storageModeShared)!       // 2048 bf16
let gdnKBuf = device.makeBuffer(length: GDN_KEY_DIM * 2, options: .storageModeShared)!
let gdnVBuf = device.makeBuffer(length: GDN_VALUE_DIM * 2, options: .storageModeShared)!     // 6144 bf16
let gdnQNormedBuf = device.makeBuffer(length: GDN_KEY_DIM * 2, options: .storageModeShared)!
let gdnKNormedBuf = device.makeBuffer(length: GDN_KEY_DIM * 2, options: .storageModeShared)!
let gdnQExpandedBuf = device.makeBuffer(length: GDN_NUM_V_HEADS * GDN_HEAD_K_DIM * 2, options: .storageModeShared)!
let gdnKExpandedBuf = device.makeBuffer(length: GDN_NUM_V_HEADS * GDN_HEAD_K_DIM * 2, options: .storageModeShared)!
let gBuf = device.makeBuffer(length: GDN_NUM_V_HEADS * 2, options: .storageModeShared)!
let betaBuf = device.makeBuffer(length: GDN_NUM_V_HEADS * 2, options: .storageModeShared)!
let gdnYBuf = device.makeBuffer(length: GDN_VALUE_DIM * 2, options: .storageModeShared)!     // 6144 bf16
let gdnNormedBuf = device.makeBuffer(length: GDN_VALUE_DIM * 2, options: .storageModeShared)!
let gdnOutProjBuf = device.makeBuffer(length: HIDDEN_SIZE * 2, options: .storageModeShared)!

// Conv state per GDN layer: (3, 10240) bf16 = 61440 bytes each
// State per GDN layer: (48, 128, 128) float32 = 3145728 bytes each
// For single-layer test, allocate just one
let convStateBuf = device.makeBuffer(length: (GDN_CONV_KERNEL - 1) * GDN_CONV_DIM * 2, options: .storageModeShared)!
let convStateNewBuf = device.makeBuffer(length: (GDN_CONV_KERNEL - 1) * GDN_CONV_DIM * 2, options: .storageModeShared)!
let gdnStateBuf = device.makeBuffer(length: GDN_STATE_SIZE * 4, options: .storageModeShared)!  // float32

// MLP buffers (reuse from before)
let gateProjBuf = device.makeBuffer(length: INTERMEDIATE_SIZE * 2, options: .storageModeShared)!
let upProjBuf = device.makeBuffer(length: INTERMEDIATE_SIZE * 2, options: .storageModeShared)!
let mlpHiddenBuf = device.makeBuffer(length: INTERMEDIATE_SIZE * 2, options: .storageModeShared)!
let downProjBuf = device.makeBuffer(length: HIDDEN_SIZE * 2, options: .storageModeShared)!

// Initialize conv state and gdn state to zeros
memset(convStateBuf.contents(), 0, convStateBuf.length)
memset(gdnStateBuf.contents(), 0, gdnStateBuf.length)

print("Activation buffers allocated")

// ============================================================================
// MARK: - Verification Helpers
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
    var within01 = 0, within001 = 0
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

func compareF32BufferToRef(_ buf: MTLBuffer, _ refName: String, _ count: Int, _ label: String) {
    let path = "/Users/midas/Desktop/cowork/inference-across-metal/reference_data/\(refName)"
    let data = try! Data(contentsOf: URL(fileURLWithPath: path))
    let ref = data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    let ptr = buf.contents().bindMemory(to: Float.self, capacity: count)
    var maxErr: Float = 0
    var within01 = 0, within001 = 0
    for i in 0..<min(count, ref.count) {
        let err = abs(ptr[i] - ref[i])
        let denom = max(abs(ref[i]), 1e-6)
        let relErr = err / denom
        if relErr < 0.1 { within01 += 1 }
        if relErr < 0.01 { within001 += 1 }
        if err > maxErr { maxErr = err }
    }
    let total = min(count, ref.count)
    let pct01 = Float(within01) / Float(total) * 100
    let status = pct01 > 90 ? "PASS" : "FAIL"
    print("  \(label): \(status) — \(String(format: "%.0f", pct01))% < 0.1, max_err=\(String(format: "%.6f", maxErr))")
}

// ============================================================================
// MARK: - GDN Layer 0 Verification
// ============================================================================

print("\n=== GDN Layer 0 Verification ===")

// Embed token 1234
let cb1 = queue.makeCommandBuffer()!
let enc1 = cb1.makeComputeCommandEncoder()!
enc1.setComputePipelineState(embedPSO)
enc1.setBuffer(embedW, offset: 0, index: 0)
enc1.setBuffer(embedS, offset: 0, index: 1)
enc1.setBuffer(embedB, offset: 0, index: 2)
enc1.setBuffer(hiddenBuf, offset: 0, index: 3)
var tokenIdV = UInt32(1234)
enc1.setBytes(&tokenIdV, length: 4, index: 4)
var hdimV = UInt32(HIDDEN_SIZE)
enc1.setBytes(&hdimV, length: 4, index: 5)
var gszV = UInt32(GROUP_SIZE)
enc1.setBytes(&gszV, length: 4, index: 6)
enc1.dispatchThreads(MTLSize(width: HIDDEN_SIZE, height: 1, depth: 1),
                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
enc1.endEncoding()
cb1.commit()
cb1.waitUntilCompleted()

// Now run GDN layer 0
let cb2 = queue.makeCommandBuffer()!
let enc = cb2.makeComputeCommandEncoder()!

// 1. Input norm
enc.setComputePipelineState(rmsNormPSO)
enc.setBuffer(hiddenBuf, offset: 0, index: 0)
enc.setBuffer(gdn0.inputNormWeight!, offset: 0, index: 1)
enc.setBuffer(normedBuf, offset: 0, index: 2)
var dim = UInt32(HIDDEN_SIZE)
enc.setBytes(&dim, length: 4, index: 3)
var eps = RMS_NORM_EPS
enc.setBytes(&eps, length: 4, index: 4)
enc.dispatchThreads(MTLSize(width: 256, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

// 2. Projections
dispatchQMatmul(enc, gdn0.qkvProjW!, gdn0.qkvProjS!, gdn0.qkvProjB!,
                normedBuf, qkvBuf, K_dim: HIDDEN_SIZE, N_dim: GDN_CONV_DIM, M_dim: 1)
dispatchQMatmul(enc, gdn0.zProjW!, gdn0.zProjS!, gdn0.zProjB!,
                normedBuf, zBuf, K_dim: HIDDEN_SIZE, N_dim: GDN_VALUE_DIM, M_dim: 1)
dispatchQMatmul(enc, gdn0.bProjW!, gdn0.bProjS!, gdn0.bProjB!,
                normedBuf, bBuf, K_dim: HIDDEN_SIZE, N_dim: GDN_NUM_V_HEADS, M_dim: 1)
dispatchQMatmul(enc, gdn0.aProjW!, gdn0.aProjS!, gdn0.aProjB!,
                normedBuf, aBuf, K_dim: HIDDEN_SIZE, N_dim: GDN_NUM_V_HEADS, M_dim: 1)

// 3. Depthwise conv1d (M=1)
enc.setComputePipelineState(depthwiseConvPSO)
enc.setBuffer(convStateBuf, offset: 0, index: 0)
enc.setBuffer(qkvBuf, offset: 0, index: 1)
enc.setBuffer(gdn0.convWeight!, offset: 0, index: 2)
enc.setBuffer(convOutBuf, offset: 0, index: 3)
enc.setBuffer(convStateNewBuf, offset: 0, index: 4)
var cdim = UInt32(GDN_CONV_DIM)
enc.setBytes(&cdim, length: 4, index: 5)
enc.dispatchThreads(MTLSize(width: GDN_CONV_DIM, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

// 4. SiLU on conv output
enc.setComputePipelineState(siluPSO)
enc.setBuffer(convOutBuf, offset: 0, index: 0)
enc.setBuffer(convSiluBuf, offset: 0, index: 1)
enc.setBytes(&cdim, length: 4, index: 2)
enc.dispatchThreads(MTLSize(width: GDN_CONV_DIM, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

// 5. Split Q/K/V from conv output
// Q: first 2048, K: next 2048, V: last 6144 — they're contiguous in convSiluBuf
// Just use offsets into convSiluBuf (no copy needed for the split)
// But we need separate buffers for the expand step. Copy them.
// Actually, we can set buffer offsets in Metal. Let's use a simple copy kernel.
// For simplicity, just read from convSiluBuf with offsets.
// gdnQBuf = convSiluBuf[0:2048], gdnKBuf = convSiluBuf[2048:4096], gdnVBuf = convSiluBuf[4096:10240]

// 6. Per-head RMS norm with scale (QK norm)
// Q: scale = inv_scale^2 = (1/sqrt(128))^2 = 1/128
// K: scale = inv_scale = 1/sqrt(128)
enc.setComputePipelineState(perHeadRmsNormScaledPSO)
enc.setBuffer(convSiluBuf, offset: 0, index: 0)  // Q portion starts at offset 0
enc.setBuffer(gdnQNormedBuf, offset: 0, index: 1)
var nkh = UInt32(GDN_NUM_K_HEADS)
enc.setBytes(&nkh, length: 4, index: 2)
var hkd = UInt32(GDN_HEAD_K_DIM)
enc.setBytes(&hkd, length: 4, index: 3)
var normEps: Float = 1e-6
enc.setBytes(&normEps, length: 4, index: 4)
var qScale: Float = GDN_INV_SCALE * GDN_INV_SCALE  // inv_scale^2
enc.setBytes(&qScale, length: 4, index: 5)
enc.dispatchThreads(MTLSize(width: GDN_HEAD_K_DIM, height: GDN_NUM_K_HEADS, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: GDN_HEAD_K_DIM, height: 1, depth: 1))

// K norm (offset by key_dim * 2 bytes into convSiluBuf)
enc.setBuffer(convSiluBuf, offset: GDN_KEY_DIM * 2, index: 0)  // K starts after Q
enc.setBuffer(gdnKNormedBuf, offset: 0, index: 1)
var kScale: Float = GDN_INV_SCALE  // inv_scale
enc.setBytes(&kScale, length: 4, index: 5)
enc.dispatchThreads(MTLSize(width: GDN_HEAD_K_DIM, height: GDN_NUM_K_HEADS, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: GDN_HEAD_K_DIM, height: 1, depth: 1))

// 7. Expand K heads to V heads (16 → 48, 3x repeat)
enc.setComputePipelineState(expandKvPSO)
enc.setBuffer(gdnQNormedBuf, offset: 0, index: 0)
enc.setBuffer(gdnQExpandedBuf, offset: 0, index: 1)
var nvh = UInt32(GDN_NUM_V_HEADS)
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
enc.setBuffer(gdn0.ALog!, offset: 0, index: 2)
enc.setBuffer(gdn0.dtBias!, offset: 0, index: 3)
enc.setBuffer(gBuf, offset: 0, index: 4)
enc.setBuffer(betaBuf, offset: 0, index: 5)
enc.setBytes(&nvh, length: 4, index: 6)
enc.dispatchThreads(MTLSize(width: GDN_NUM_V_HEADS, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 48, height: 1, depth: 1))

// 9. Gated delta step
enc.setComputePipelineState(gdnStepPSO)
enc.setBuffer(gdnQExpandedBuf, offset: 0, index: 0)
enc.setBuffer(gdnKExpandedBuf, offset: 0, index: 1)
enc.setBuffer(convSiluBuf, offset: GDN_KEY_DIM * 2 * 2, index: 2)  // V starts after Q+K
enc.setBuffer(gBuf, offset: 0, index: 3)
enc.setBuffer(betaBuf, offset: 0, index: 4)
enc.setBuffer(gdnStateBuf, offset: 0, index: 5)
enc.setBuffer(gdnYBuf, offset: 0, index: 6)
enc.setBytes(&nvh, length: 4, index: 7)
var hvd = UInt32(GDN_HEAD_V_DIM)
enc.setBytes(&hvd, length: 4, index: 8)
enc.setBytes(&hkd, length: 4, index: 9)
enc.dispatchThreads(MTLSize(width: GDN_HEAD_V_DIM, height: GDN_NUM_V_HEADS, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))

// 10. RMSNormGated: silu(z) * rms_norm(y, weight)
enc.setComputePipelineState(rmsNormGatedPSO)
enc.setBuffer(gdnYBuf, offset: 0, index: 0)
enc.setBuffer(zBuf, offset: 0, index: 1)
enc.setBuffer(gdn0.normWeight!, offset: 0, index: 2)
enc.setBuffer(gdnNormedBuf, offset: 0, index: 3)
enc.setBytes(&nvh, length: 4, index: 4)
enc.setBytes(&hvd, length: 4, index: 5)
enc.setBytes(&eps, length: 4, index: 6)
enc.dispatchThreads(MTLSize(width: GDN_HEAD_V_DIM, height: GDN_NUM_V_HEADS, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))

// 11. Out projection
dispatchQMatmul(enc, gdn0.outProjW!, gdn0.outProjS!, gdn0.outProjB!,
                gdnNormedBuf, gdnOutProjBuf, K_dim: GDN_VALUE_DIM, N_dim: HIDDEN_SIZE, M_dim: 1)

// 12. Residual
enc.setComputePipelineState(residualPSO)
enc.setBuffer(hiddenBuf, offset: 0, index: 0)
enc.setBuffer(gdnOutProjBuf, offset: 0, index: 1)
enc.setBuffer(residualBuf, offset: 0, index: 2)
var cnt = UInt32(HIDDEN_SIZE)
enc.setBytes(&cnt, length: 4, index: 3)
enc.dispatchThreads(MTLSize(width: HIDDEN_SIZE, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

// 13. Post-norm + MLP
enc.setComputePipelineState(rmsNormPSO)
enc.setBuffer(residualBuf, offset: 0, index: 0)
enc.setBuffer(gdn0.postNormWeight!, offset: 0, index: 1)
enc.setBuffer(normedBuf, offset: 0, index: 2)
enc.setBytes(&dim, length: 4, index: 3)
enc.setBytes(&eps, length: 4, index: 4)
enc.dispatchThreads(MTLSize(width: 256, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

dispatchQMatmul(enc, gdn0.gateProjW!, gdn0.gateProjS!, gdn0.gateProjB!,
                normedBuf, gateProjBuf, K_dim: HIDDEN_SIZE, N_dim: INTERMEDIATE_SIZE, M_dim: 1)
dispatchQMatmul(enc, gdn0.upProjW!, gdn0.upProjS!, gdn0.upProjB!,
                normedBuf, upProjBuf, K_dim: HIDDEN_SIZE, N_dim: INTERMEDIATE_SIZE, M_dim: 1)

enc.setComputePipelineState(siluMulPSO)
enc.setBuffer(gateProjBuf, offset: 0, index: 0)
enc.setBuffer(upProjBuf, offset: 0, index: 1)
enc.setBuffer(mlpHiddenBuf, offset: 0, index: 2)
var isz = UInt32(INTERMEDIATE_SIZE)
enc.setBytes(&isz, length: 4, index: 3)
enc.dispatchThreads(MTLSize(width: INTERMEDIATE_SIZE, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

dispatchQMatmul(enc, gdn0.downProjW!, gdn0.downProjS!, gdn0.downProjB!,
                mlpHiddenBuf, downProjBuf, K_dim: INTERMEDIATE_SIZE, N_dim: HIDDEN_SIZE, M_dim: 1)

// Final residual (layer output)
let layerOutBuf = device.makeBuffer(length: HIDDEN_SIZE * 2, options: .storageModeShared)!
enc.setComputePipelineState(residualPSO)
enc.setBuffer(residualBuf, offset: 0, index: 0)
enc.setBuffer(downProjBuf, offset: 0, index: 1)
enc.setBuffer(layerOutBuf, offset: 0, index: 2)
enc.setBytes(&cnt, length: 4, index: 3)
enc.dispatchThreads(MTLSize(width: HIDDEN_SIZE, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

enc.endEncoding()
cb2.commit()
cb2.waitUntilCompleted()

if cb2.status == .error {
    print("ERROR: Command buffer failed: \(cb2.error?.localizedDescription ?? "unknown")")
}

// ============================================================================
// MARK: - Verify each component
// ============================================================================

print("\nGDN Layer 0 component verification:")
compareBufferToRef(qkvBuf, "gdn_ref_qkv_proj.bin", GDN_CONV_DIM, "qkv_proj")
compareBufferToRef(zBuf, "gdn_ref_z_proj.bin", GDN_VALUE_DIM, "z_proj")
compareBufferToRef(bBuf, "gdn_ref_b_proj.bin", GDN_NUM_V_HEADS, "b_proj")
compareBufferToRef(aBuf, "gdn_ref_a_proj.bin", GDN_NUM_V_HEADS, "a_proj")
compareBufferToRef(convOutBuf, "gdn_ref_conv_out_raw.bin", GDN_CONV_DIM, "conv1d_raw")
compareBufferToRef(convSiluBuf, "gdn_ref_conv_out_silu.bin", GDN_CONV_DIM, "conv1d_silu")

// Q/K from conv split — compare against ref using offset into convSiluBuf
// Q is first 2048 of convSiluBuf, K is next 2048
let qSplitRef = loadRefBf16("gdn_ref_q_split.bin", count: GDN_KEY_DIM)
let kSplitRef = loadRefBf16("gdn_ref_k_split.bin", count: GDN_KEY_DIM)
let vSplitRef = loadRefBf16("gdn_ref_v_split.bin", count: GDN_VALUE_DIM)
// Verify by reading from convSiluBuf
let csPtr = convSiluBuf.contents().bindMemory(to: UInt16.self, capacity: GDN_CONV_DIM)
var qMatch = 0, kMatch = 0, vMatch = 0
for i in 0..<GDN_KEY_DIM {
    var bits = UInt32(csPtr[i]) << 16
    let val = withUnsafeBytes(of: &bits) { $0.load(as: Float.self) }
    if abs(val - qSplitRef[i]) / max(abs(qSplitRef[i]), 1e-6) < 0.1 { qMatch += 1 }
}
for i in 0..<GDN_KEY_DIM {
    var bits = UInt32(csPtr[GDN_KEY_DIM + i]) << 16
    let val = withUnsafeBytes(of: &bits) { $0.load(as: Float.self) }
    if abs(val - kSplitRef[i]) / max(abs(kSplitRef[i]), 1e-6) < 0.1 { kMatch += 1 }
}
for i in 0..<GDN_VALUE_DIM {
    var bits = UInt32(csPtr[2 * GDN_KEY_DIM + i]) << 16
    let val = withUnsafeBytes(of: &bits) { $0.load(as: Float.self) }
    if abs(val - vSplitRef[i]) / max(abs(vSplitRef[i]), 1e-6) < 0.1 { vMatch += 1 }
}
print("  q_split: \(qMatch * 100 / GDN_KEY_DIM)% < 0.1")
print("  k_split: \(kMatch * 100 / GDN_KEY_DIM)% < 0.1")
print("  v_split: \(vMatch * 100 / GDN_VALUE_DIM)% < 0.1")

compareBufferToRef(gdnQNormedBuf, "gdn_ref_q_normed.bin", GDN_KEY_DIM, "q_normed")
compareBufferToRef(gdnKNormedBuf, "gdn_ref_k_normed.bin", GDN_KEY_DIM, "k_normed")
compareBufferToRef(gBuf, "gdn_ref_g.bin", GDN_NUM_V_HEADS, "g_compute")
compareBufferToRef(betaBuf, "gdn_ref_beta.bin", GDN_NUM_V_HEADS, "beta_compute")
compareBufferToRef(gdnYBuf, "gdn_ref_delta_output.bin", GDN_VALUE_DIM, "delta_output")
compareF32BufferToRef(gdnStateBuf, "gdn_ref_new_state.bin", GDN_STATE_SIZE, "state_update")
compareBufferToRef(gdnNormedBuf, "gdn_ref_gated_norm.bin", GDN_VALUE_DIM, "rms_norm_gated")
compareBufferToRef(gdnOutProjBuf, "gdn_ref_out_proj.bin", HIDDEN_SIZE, "out_proj")
compareBufferToRef(layerOutBuf, "gdn_ref_layer0_output.bin", HIDDEN_SIZE, "layer_output")

// Timing
print("\n=== GDN Layer 0 Timing ===")
// Warmup
for _ in 0..<3 {
    memset(convStateBuf.contents(), 0, convStateBuf.length)
    memset(gdnStateBuf.contents(), 0, gdnStateBuf.length)
    let cb = queue.makeCommandBuffer()!
    let e = cb.makeComputeCommandEncoder()!
    // Just run the GDN step (the expensive part)
    e.setComputePipelineState(gdnStepPSO)
    e.setBuffer(gdnQExpandedBuf, offset: 0, index: 0)
    e.setBuffer(gdnKExpandedBuf, offset: 0, index: 1)
    e.setBuffer(convSiluBuf, offset: GDN_KEY_DIM * 2 * 2, index: 2)
    e.setBuffer(gBuf, offset: 0, index: 3)
    e.setBuffer(betaBuf, offset: 0, index: 4)
    e.setBuffer(gdnStateBuf, offset: 0, index: 5)
    e.setBuffer(gdnYBuf, offset: 0, index: 6)
    var nvh2 = UInt32(GDN_NUM_V_HEADS)
    e.setBytes(&nvh2, length: 4, index: 7)
    var hvd2 = UInt32(GDN_HEAD_V_DIM)
    e.setBytes(&hvd2, length: 4, index: 8)
    var hkd2 = UInt32(GDN_HEAD_K_DIM)
    e.setBytes(&hkd2, length: 4, index: 9)
    e.dispatchThreads(MTLSize(width: GDN_HEAD_V_DIM, height: GDN_NUM_V_HEADS, depth: 1),
                      threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    e.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
}

var gdnTimes: [Double] = []
for _ in 0..<10 {
    memset(gdnStateBuf.contents(), 0, gdnStateBuf.length)
    let t = CFAbsoluteTimeGetCurrent()
    let cb = queue.makeCommandBuffer()!
    let e = cb.makeComputeCommandEncoder()!
    e.setComputePipelineState(gdnStepPSO)
    e.setBuffer(gdnQExpandedBuf, offset: 0, index: 0)
    e.setBuffer(gdnKExpandedBuf, offset: 0, index: 1)
    e.setBuffer(convSiluBuf, offset: GDN_KEY_DIM * 2 * 2, index: 2)
    e.setBuffer(gBuf, offset: 0, index: 3)
    e.setBuffer(betaBuf, offset: 0, index: 4)
    e.setBuffer(gdnStateBuf, offset: 0, index: 5)
    e.setBuffer(gdnYBuf, offset: 0, index: 6)
    var nvh2 = UInt32(GDN_NUM_V_HEADS)
    e.setBytes(&nvh2, length: 4, index: 7)
    var hvd2 = UInt32(GDN_HEAD_V_DIM)
    e.setBytes(&hvd2, length: 4, index: 8)
    var hkd2 = UInt32(GDN_HEAD_K_DIM)
    e.setBytes(&hkd2, length: 4, index: 9)
    e.dispatchThreads(MTLSize(width: GDN_HEAD_V_DIM, height: GDN_NUM_V_HEADS, depth: 1),
                      threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    e.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    gdnTimes.append(CFAbsoluteTimeGetCurrent() - t)
}
let avgGdn = gdnTimes.reduce(0, +) / Double(gdnTimes.count) * 1000
let minGdn = gdnTimes.min()! * 1000
print("GDN step only: avg=\(String(format: "%.2f", avgGdn))ms, min=\(String(format: "%.2f", minGdn))ms")
print("State size: \(GDN_STATE_SIZE * 4 / 1024)KB per layer, \(GDN_STATE_SIZE * 4 * 48 / 1024 / 1024)MB for 48 layers")

print("\nGDN M=1 decode kernel built and verified.")
