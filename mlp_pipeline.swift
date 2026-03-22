// mlp_pipeline.swift — IAM Step 3: Double-buffered MLP pipeline from Swift
//
// Dispatches 3 quantized matmuls (gate_proj, up_proj, down_proj) + custom
// silu_multiply kernel for one full MLP layer. Then pipelines 32 "layers"
// with double-buffered command encoding.
//
// Gate: pipeline throughput within 2x of MLX MLP (1.35ms target)

import Metal
import Foundation

let device = MTLCreateSystemDefaultDevice()!
let commandQueue = device.makeCommandQueue()!
let mlxLib = try! device.makeLibrary(URL: URL(fileURLWithPath: "/Users/midas/.mlx-env/lib/python3.11/site-packages/mlx/lib/mlx.metallib"))
let refDir = "/Users/midas/Desktop/cowork/inference-across-metal/reference_data"

func loadBin(_ name: String) -> Data {
    try! Data(contentsOf: URL(fileURLWithPath: "\(refDir)/\(name)"))
}
func bf16ToFloat(_ val: UInt16) -> Float { Float(bitPattern: UInt32(val) << 16) }

print("Device: \(device.name)")
print()

// ── Custom silu_multiply kernel ─────────────────────────────────
// silu(gate) * up = gate * sigmoid(gate) * up
// Fused into one kernel to avoid intermediate buffers

let siluMulSource = """
#include <metal_stdlib>
using namespace metal;

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
    float s = g / (1.0f + exp(-g));  // silu = x * sigmoid(x)
    out[tid] = bfloat(s * u);
}
"""

let siluLib = try! device.makeLibrary(source: siluMulSource, options: nil)
let siluFn = siluLib.makeFunction(name: "silu_multiply_bf16")!
let siluPSO = try! device.makeComputePipelineState(function: siluFn)

// ── Load qmm kernel (NAX, bf16) ────────────────────────────────

let kernelName = "affine_qmm_t_nax_bfloat16_t_gs_64_b_4_bm64_bn64_bk64_wm2_wn2_alN_true_batch_0"
let qmmFn = mlxLib.makeFunction(name: kernelName)!
let qmmPSO = try! device.makeComputePipelineState(function: qmmFn)

// ── Load weights ────────────────────────────────────────────────

print("=== Loading MLP Weights ===")

// Input (1, 4096) bf16 — same as Step 2
let inputData = loadBin("qlinear_input_bf16.bin")

// gate_proj: (1, 4096) → (1, 12288), weight (12288, 512), scales/biases (12288, 64)
let gateW = loadBin("mlp_gate_weight_raw.bin")
let gateS = loadBin("mlp_gate_scales_bf16.bin")
let gateB = loadBin("mlp_gate_biases_bf16.bin")

// up_proj: same shapes as gate (already have from Step 2)
let upW = loadBin("qlinear_weight_raw.bin")
let upS = loadBin("qlinear_scales_bf16.bin")
let upB = loadBin("qlinear_biases_bf16.bin")

// down_proj: (1, 12288) → (1, 4096), weight (4096, 1536), scales/biases (4096, 192)
let downW = loadBin("mlp_down_weight_raw.bin")
let downS = loadBin("mlp_down_scales_bf16.bin")
let downB = loadBin("mlp_down_biases_bf16.bin")

// Reference outputs
let gateRefBF16 = loadBin("mlp_gate_output_bf16.bin")
let upRefBF16 = loadBin("mlp_up_output_bf16.bin")
let hiddenRefBF16 = loadBin("mlp_hidden_bf16.bin")
let downRefBF16 = loadBin("mlp_down_output_bf16.bin")

print("  Input: \(inputData.count) bytes")
print("  Gate weights: \(gateW.count + gateS.count + gateB.count) bytes")
print("  Up weights: \(upW.count + upS.count + upB.count) bytes")
print("  Down weights: \(downW.count + downS.count + downB.count) bytes")

// ── Create Metal buffers ────────────────────────────────────────

func makeBuf(_ data: Data) -> MTLBuffer {
    data.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: data.count, options: .storageModeShared)! }
}

let xBuf = makeBuf(inputData)

let gateWBuf = makeBuf(gateW), gateSBuf = makeBuf(gateS), gateBBuf = makeBuf(gateB)
let upWBuf = makeBuf(upW), upSBuf = makeBuf(upS), upBBuf = makeBuf(upB)
let downWBuf = makeBuf(downW), downSBuf = makeBuf(downS), downBBuf = makeBuf(downB)

// Intermediate/output buffers (bf16)
let dim12288 = 1 * 12288 * 2  // bf16
let dim4096 = 1 * 4096 * 2
let gateOutBuf = device.makeBuffer(length: dim12288, options: .storageModeShared)!
let upOutBuf = device.makeBuffer(length: dim12288, options: .storageModeShared)!
let hiddenBuf = device.makeBuffer(length: dim12288, options: .storageModeShared)!
let downOutBuf = device.makeBuffer(length: dim4096, options: .storageModeShared)!

// ── Dispatch helpers ────────────────────────────────────────────

// qmm dispatch: M=1, K=inDim, N=outDim
func encodeQMM(_ enc: MTLComputeCommandEncoder,
               w: MTLBuffer, s: MTLBuffer, b: MTLBuffer,
               x: MTLBuffer, y: MTLBuffer,
               inDim: Int32, outDim: Int32) {
    var K = inDim, N = outDim, M: Int32 = 1
    enc.setComputePipelineState(qmmPSO)
    enc.setBuffer(w, offset: 0, index: 0)
    enc.setBuffer(s, offset: 0, index: 1)
    enc.setBuffer(b, offset: 0, index: 2)
    enc.setBuffer(x, offset: 0, index: 3)
    enc.setBuffer(y, offset: 0, index: 4)
    enc.setBytes(&K, length: 4, index: 5)
    enc.setBytes(&N, length: 4, index: 6)
    enc.setBytes(&M, length: 4, index: 7)

    let BN = 64, BM = 64
    let gridW = (Int(N) + BN - 1) / BN
    let gridH = (Int(M) + BM - 1) / BM
    enc.dispatchThreadgroups(
        MTLSize(width: gridW, height: gridH, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 32, height: 2, depth: 2))
}

func encodeSiluMul(_ enc: MTLComputeCommandEncoder,
                   gate: MTLBuffer, up: MTLBuffer, out: MTLBuffer,
                   count: Int) {
    var n = UInt32(count)
    enc.setComputePipelineState(siluPSO)
    enc.setBuffer(gate, offset: 0, index: 0)
    enc.setBuffer(up, offset: 0, index: 1)
    enc.setBuffer(out, offset: 0, index: 2)
    enc.setBytes(&n, length: 4, index: 3)

    let tpg = siluPSO.maxTotalThreadsPerThreadgroup
    let gridSize = (count + tpg - 1) / tpg
    enc.dispatchThreadgroups(
        MTLSize(width: gridSize, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: tpg, height: 1, depth: 1))
}

// ── Phase 1: Single MLP forward pass with verification ──────────

print("\n=== Phase 1: Single MLP Forward Pass ===")

let cmd = commandQueue.makeCommandBuffer()!
let enc = cmd.makeComputeCommandEncoder()!

// 1. gate_proj(x): (1,4096) → (1,12288)
encodeQMM(enc, w: gateWBuf, s: gateSBuf, b: gateBBuf, x: xBuf, y: gateOutBuf, inDim: 4096, outDim: 12288)
// 2. up_proj(x): (1,4096) → (1,12288)
encodeQMM(enc, w: upWBuf, s: upSBuf, b: upBBuf, x: xBuf, y: upOutBuf, inDim: 4096, outDim: 12288)
// 3. silu(gate) * up → hidden (1,12288)
encodeSiluMul(enc, gate: gateOutBuf, up: upOutBuf, out: hiddenBuf, count: 12288)
// 4. down_proj(hidden): (1,12288) → (1,4096)
encodeQMM(enc, w: downWBuf, s: downSBuf, b: downBBuf, x: hiddenBuf, y: downOutBuf, inDim: 12288, outDim: 4096)

enc.endEncoding()
cmd.commit()
cmd.waitUntilCompleted()

if let error = cmd.error {
    print("  GPU ERROR: \(error)")
} else {
    print("  MLP dispatch completed successfully")
}

// Verify each stage
func verifyBF16(_ buf: MTLBuffer, ref: Data, name: String, count: Int) {
    let ptr = buf.contents().bindMemory(to: UInt16.self, capacity: count)
    let refPtr = ref.withUnsafeBytes { $0.bindMemory(to: UInt16.self) }

    var exact = 0, within001 = 0, within01 = 0, maxErr: Float = 0
    for i in 0..<count {
        if ptr[i] == refPtr[i] { exact += 1 }
        let got = bf16ToFloat(ptr[i])
        let exp = bf16ToFloat(refPtr[i])
        let err = abs(got - exp)
        maxErr = max(maxErr, err)
        if err < 0.01 { within001 += 1 }
        if err < 0.1 { within01 += 1 }
    }
    let pct = Float(within01) / Float(count) * 100
    let pct001 = Float(within001) / Float(count) * 100
    print("  \(name): exact=\(exact)/\(count) (\(String(format: "%.1f", Float(exact)/Float(count)*100))%), <0.01=\(String(format: "%.1f", pct001))%, <0.1=\(String(format: "%.1f", pct))%, max=\(String(format: "%.4f", maxErr))")

    // First 4 values
    print("    got: ", terminator: "")
    for i in 0..<min(4, count) { print(String(format: "%.4f ", bf16ToFloat(ptr[i])), terminator: "") }
    print()
    print("    ref: ", terminator: "")
    for i in 0..<min(4, count) { print(String(format: "%.4f ", bf16ToFloat(refPtr[i])), terminator: "") }
    print()
}

print("\n  Stage verification:")
verifyBF16(gateOutBuf, ref: gateRefBF16, name: "gate_proj", count: 12288)
verifyBF16(upOutBuf, ref: upRefBF16, name: "up_proj  ", count: 12288)
verifyBF16(hiddenBuf, ref: hiddenRefBF16, name: "silu*up  ", count: 12288)
verifyBF16(downOutBuf, ref: downRefBF16, name: "down_proj", count: 4096)

// ── Phase 2: Sequential timing (single command buffer) ──────────

print("\n=== Phase 2: Sequential MLP Timing ===")

var seqTimes: [Double] = []
for trial in 0..<23 {
    let start = CFAbsoluteTimeGetCurrent()

    let cmd = commandQueue.makeCommandBuffer()!
    let enc = cmd.makeComputeCommandEncoder()!
    encodeQMM(enc, w: gateWBuf, s: gateSBuf, b: gateBBuf, x: xBuf, y: gateOutBuf, inDim: 4096, outDim: 12288)
    encodeQMM(enc, w: upWBuf, s: upSBuf, b: upBBuf, x: xBuf, y: upOutBuf, inDim: 4096, outDim: 12288)
    encodeSiluMul(enc, gate: gateOutBuf, up: upOutBuf, out: hiddenBuf, count: 12288)
    encodeQMM(enc, w: downWBuf, s: downSBuf, b: downBBuf, x: hiddenBuf, y: downOutBuf, inDim: 12288, outDim: 4096)
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()

    let elapsed = CFAbsoluteTimeGetCurrent() - start
    if trial >= 3 { seqTimes.append(elapsed * 1000) }
}

let seqMean = seqTimes.reduce(0, +) / Double(seqTimes.count)
let seqStd = sqrt(seqTimes.map { ($0 - seqMean) * ($0 - seqMean) }.reduce(0, +) / Double(seqTimes.count))
print("  Single MLP (sequential): \(String(format: "%.3f", seqMean)) ms (std \(String(format: "%.3f", seqStd)))")
print("  MLX reference:           1.352 ms")
print("  Ratio:                   \(String(format: "%.2f", seqMean / 1.352))x")

// ── Phase 3: Double-buffered 32-layer pipeline ──────────────────
//
// Simulate a full 32-layer forward pass. Use two command buffers:
// encode layer N+1 while GPU executes layer N.
// All layers reuse same weights (structure test, not correctness).

print("\n=== Phase 3: Double-Buffered 32-Layer Pipeline ===")
let numLayers = 32

// Double-buffer output slots
// Layer reads from slot[n%2], writes to slot[(n+1)%2]
// For MLP: input is 4096, output is 4096 (residual ignored for pipeline test)
let slotA = device.makeBuffer(length: dim4096, options: .storageModeShared)!
let slotB = device.makeBuffer(length: dim4096, options: .storageModeShared)!

// Copy input to slot A
memcpy(slotA.contents(), xBuf.contents(), dim4096)

// Intermediate buffers (shared across layers, fine because sequential)
let gOut = device.makeBuffer(length: dim12288, options: .storageModeShared)!
let uOut = device.makeBuffer(length: dim12288, options: .storageModeShared)!
let hOut = device.makeBuffer(length: dim12288, options: .storageModeShared)!

// --- Single-buffer baseline (no overlap) ---
var singleTimes: [Double] = []
for trial in 0..<13 {
    memcpy(slotA.contents(), xBuf.contents(), dim4096)
    let start = CFAbsoluteTimeGetCurrent()

    for layer in 0..<numLayers {
        let inBuf = (layer % 2 == 0) ? slotA : slotB
        let outBuf = (layer % 2 == 0) ? slotB : slotA

        let cmd = commandQueue.makeCommandBuffer()!
        let enc = cmd.makeComputeCommandEncoder()!
        encodeQMM(enc, w: gateWBuf, s: gateSBuf, b: gateBBuf, x: inBuf, y: gOut, inDim: 4096, outDim: 12288)
        encodeQMM(enc, w: upWBuf, s: upSBuf, b: upBBuf, x: inBuf, y: uOut, inDim: 4096, outDim: 12288)
        encodeSiluMul(enc, gate: gOut, up: uOut, out: hOut, count: 12288)
        encodeQMM(enc, w: downWBuf, s: downSBuf, b: downBBuf, x: hOut, y: outBuf, inDim: 12288, outDim: 4096)
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }

    let elapsed = CFAbsoluteTimeGetCurrent() - start
    if trial >= 3 { singleTimes.append(elapsed * 1000) }
}

let singleMean = singleTimes.reduce(0, +) / Double(singleTimes.count)
let singlePerLayer = singleMean / Double(numLayers)
print("  Single-buffer 32 layers: \(String(format: "%.1f", singleMean)) ms (\(String(format: "%.3f", singlePerLayer)) ms/layer)")

// --- Double-buffer (overlap encoding with execution) ---
// Use completion handlers to overlap: while GPU runs layer N,
// CPU encodes layer N+1 into the next command buffer.
var doubleTimes: [Double] = []
for trial in 0..<13 {
    memcpy(slotA.contents(), xBuf.contents(), dim4096)
    let start = CFAbsoluteTimeGetCurrent()

    // We need separate intermediate buffers per pipeline stage to avoid WAR hazards
    let gOutA = device.makeBuffer(length: dim12288, options: .storageModeShared)!
    let uOutA = device.makeBuffer(length: dim12288, options: .storageModeShared)!
    let hOutA = device.makeBuffer(length: dim12288, options: .storageModeShared)!
    let gOutB = device.makeBuffer(length: dim12288, options: .storageModeShared)!
    let uOutB = device.makeBuffer(length: dim12288, options: .storageModeShared)!
    let hOutB = device.makeBuffer(length: dim12288, options: .storageModeShared)!

    var prevCmd: MTLCommandBuffer? = nil

    for layer in 0..<numLayers {
        let inBuf = (layer % 2 == 0) ? slotA : slotB
        let outBuf = (layer % 2 == 0) ? slotB : slotA
        let gBuf = (layer % 2 == 0) ? gOutA : gOutB
        let uBuf = (layer % 2 == 0) ? uOutA : uOutB
        let hBuf = (layer % 2 == 0) ? hOutA : hOutB

        let cmd = commandQueue.makeCommandBuffer()!
        let enc = cmd.makeComputeCommandEncoder()!
        encodeQMM(enc, w: gateWBuf, s: gateSBuf, b: gateBBuf, x: inBuf, y: gBuf, inDim: 4096, outDim: 12288)
        encodeQMM(enc, w: upWBuf, s: upSBuf, b: upBBuf, x: inBuf, y: uBuf, inDim: 4096, outDim: 12288)
        encodeSiluMul(enc, gate: gBuf, up: uBuf, out: hBuf, count: 12288)
        encodeQMM(enc, w: downWBuf, s: downSBuf, b: downBBuf, x: hBuf, y: outBuf, inDim: 12288, outDim: 4096)
        enc.endEncoding()

        // Wait for previous layer to finish before committing
        // (data dependency: this layer reads previous layer's output)
        if let prev = prevCmd {
            prev.waitUntilCompleted()
        }

        cmd.commit()
        prevCmd = cmd
    }

    // Wait for final layer
    prevCmd?.waitUntilCompleted()

    let elapsed = CFAbsoluteTimeGetCurrent() - start
    if trial >= 3 { doubleTimes.append(elapsed * 1000) }
}

let doubleMean = doubleTimes.reduce(0, +) / Double(doubleTimes.count)
let doublePerLayer = doubleMean / Double(numLayers)
print("  Double-buffer 32 layers: \(String(format: "%.1f", doubleMean)) ms (\(String(format: "%.3f", doublePerLayer)) ms/layer)")

// --- Batched: all 4 dispatches in one command buffer per layer ---
// This is the most efficient single-buffer approach
var batchedTimes: [Double] = []
for trial in 0..<13 {
    memcpy(slotA.contents(), xBuf.contents(), dim4096)
    let start = CFAbsoluteTimeGetCurrent()

    // All 32 layers in one command buffer (max throughput, no CB overhead per layer)
    let cmd = commandQueue.makeCommandBuffer()!

    for layer in 0..<numLayers {
        let inBuf = (layer % 2 == 0) ? slotA : slotB
        let outBuf = (layer % 2 == 0) ? slotB : slotA

        let enc = cmd.makeComputeCommandEncoder()!
        encodeQMM(enc, w: gateWBuf, s: gateSBuf, b: gateBBuf, x: inBuf, y: gOut, inDim: 4096, outDim: 12288)
        encodeQMM(enc, w: upWBuf, s: upSBuf, b: upBBuf, x: inBuf, y: uOut, inDim: 4096, outDim: 12288)
        encodeSiluMul(enc, gate: gOut, up: uOut, out: hOut, count: 12288)
        encodeQMM(enc, w: downWBuf, s: downSBuf, b: downBBuf, x: hOut, y: outBuf, inDim: 12288, outDim: 4096)
        enc.endEncoding()
    }

    cmd.commit()
    cmd.waitUntilCompleted()

    let elapsed = CFAbsoluteTimeGetCurrent() - start
    if trial >= 3 { batchedTimes.append(elapsed * 1000) }
}

let batchedMean = batchedTimes.reduce(0, +) / Double(batchedTimes.count)
let batchedPerLayer = batchedMean / Double(numLayers)
print("  Batched 32 layers:       \(String(format: "%.1f", batchedMean)) ms (\(String(format: "%.3f", batchedPerLayer)) ms/layer)")

// ── Summary ─────────────────────────────────────────────────────

print("\n═══════════════════════════════════════════")
print("SUMMARY")
print("═══════════════════════════════════════════")
print("Single MLP (Swift):       \(String(format: "%.3f", seqMean)) ms")
print("MLX MLP reference:        1.352 ms")
print("Swift/MLX ratio:          \(String(format: "%.2f", seqMean / 1.352))x")
print()
print("32-layer pipeline:")
print("  Single-buffer:          \(String(format: "%.1f", singleMean)) ms (\(String(format: "%.3f", singlePerLayer)) ms/layer)")
print("  Double-buffer:          \(String(format: "%.1f", doubleMean)) ms (\(String(format: "%.3f", doublePerLayer)) ms/layer)")
print("  Batched (1 CB):         \(String(format: "%.1f", batchedMean)) ms (\(String(format: "%.3f", batchedPerLayer)) ms/layer)")
print()
let bestPerLayer = min(singlePerLayer, doublePerLayer, batchedPerLayer)
let bestMethod = batchedPerLayer == bestPerLayer ? "Batched" : (doublePerLayer == bestPerLayer ? "Double-buffer" : "Single-buffer")
print("Best:                     \(bestMethod) at \(String(format: "%.3f", bestPerLayer)) ms/layer")
print("MLX 32-layer equivalent:  \(String(format: "%.1f", 1.352 * 32)) ms (\(String(format: "%.3f", 1.352)) ms/layer)")
print()

// Gate: pipeline per-layer within 2x of MLX
let ratio = bestPerLayer / 1.352
let gatePass = ratio <= 2.0
print("GATE: \(String(format: "%.3f", bestPerLayer)) ms/layer / 1.352 ms = \(String(format: "%.2f", ratio))x → \(gatePass ? "PASS" : "FAIL")")
if gatePass {
    print("  Swift MLP pipeline is within 2x of MLX.")
    print("  Double-buffering proves command encoding can overlap GPU execution.")
} else {
    print("  Swift pipeline overhead too high. Investigate per-dispatch cost.")
}
