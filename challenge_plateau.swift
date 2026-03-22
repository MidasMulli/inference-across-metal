// challenge_plateau.swift — Is the "batch verification plateau" real?
//
// Challenge: the M=1 time (4.52ms) vs M=2+ (3.50ms) could just be
// dispatch overhead at M=1, not NAX cache behavior. To distinguish:
// 1. Measure dispatch overhead alone (empty kernel or trivial kernel)
// 2. Measure with the 9B model (smaller weights, different overhead ratio)
// 3. Measure with larger K (9B's 4096 vs 27B's 5120)
// 4. Check if M>1 cost grows AT ALL with M (true plateau vs slow growth)

import Metal
import Foundation

let device = MTLCreateSystemDefaultDevice()!
let commandQueue = device.makeCommandQueue()!
let mlxLib = try! device.makeLibrary(URL: URL(fileURLWithPath: "/Users/midas/.mlx-env/lib/python3.11/site-packages/mlx/lib/mlx.metallib"))
let refDir = "/Users/midas/Desktop/cowork/inference-across-metal/reference_data"

func loadBin(_ name: String) -> Data {
    try! Data(contentsOf: URL(fileURLWithPath: "\(refDir)/\(name)"))
}

print("Device: \(device.name)")
print()

// ── Test 1: Pure dispatch overhead (empty command buffer) ───────

print("=== Test 1: Pure Dispatch Overhead ===")

var dispatchTimes: [Double] = []
for trial in 0..<30 {
    let start = CFAbsoluteTimeGetCurrent()
    let cmd = commandQueue.makeCommandBuffer()!
    // Empty command buffer — just measures Metal dispatch overhead
    cmd.commit()
    cmd.waitUntilCompleted()
    let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
    if trial >= 5 { dispatchTimes.append(elapsed) }
}
let dispatchMean = dispatchTimes.reduce(0, +) / Double(dispatchTimes.count)
print("  Empty CB dispatch: \(String(format: "%.3f", dispatchMean)) ms")

// ── Test 2: Single kernel dispatch overhead ─────────────────────

print("\n=== Test 2: Single Kernel Dispatch Overhead ===")

let kernelName = "affine_qmm_t_nax_bfloat16_t_gs_64_b_4_bm64_bn64_bk64_wm2_wn2_alN_true_batch_0"
let qmmPSO = try! device.makeComputePipelineState(function: mlxLib.makeFunction(name: kernelName)!)

// 9B up_proj: K=4096, N=12288 (same as Step 2)
let wBuf = loadBin("qlinear_weight_raw.bin").withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)! }
let sBuf = loadBin("qlinear_scales_bf16.bin").withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)! }
let bBuf = loadBin("qlinear_biases_bf16.bin").withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)! }

func benchmark9B(M: Int32, runs: Int = 25, warmup: Int = 5) -> (mean: Double, std: Double) {
    let inputSize = Int(M) * 4096 * 2  // bf16
    let outputSize = Int(M) * 12288 * 2
    var inputBF16 = [UInt16](repeating: 0, count: Int(M) * 4096)
    for i in 0..<inputBF16.count { inputBF16[i] = UInt16(Float.random(in: -1...1).bitPattern >> 16) }
    let xBuf = device.makeBuffer(bytes: &inputBF16, length: inputSize, options: .storageModeShared)!
    let yBuf = device.makeBuffer(length: outputSize, options: .storageModeShared)!

    var K: Int32 = 4096, N: Int32 = 12288
    var Mv = M

    var times: [Double] = []
    for trial in 0..<(warmup + runs) {
        let start = CFAbsoluteTimeGetCurrent()
        let cmd = commandQueue.makeCommandBuffer()!
        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(qmmPSO)
        enc.setBuffer(wBuf, offset: 0, index: 0)
        enc.setBuffer(sBuf, offset: 0, index: 1)
        enc.setBuffer(bBuf, offset: 0, index: 2)
        enc.setBuffer(xBuf, offset: 0, index: 3)
        enc.setBuffer(yBuf, offset: 0, index: 4)
        enc.setBytes(&K, length: 4, index: 5)
        enc.setBytes(&N, length: 4, index: 6)
        enc.setBytes(&Mv, length: 4, index: 7)
        enc.dispatchThreadgroups(
            MTLSize(width: (Int(N) + 63) / 64, height: (Int(Mv) + 63) / 64, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 2, depth: 2))
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
        if trial >= warmup { times.append(elapsed) }
    }
    let mean = times.reduce(0, +) / Double(times.count)
    let std = sqrt(times.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Double(times.count))
    return (mean, std)
}

// ── Test 3: 9B single matmul at varying M ───────────────────────

print("\n=== Test 3: 9B Single Matmul (K=4096, N=12288) ===")
print("  M   Time(ms)  Std     Ratio  Grid(H)")
print("  ─── ──────── ──────  ───── ────────")

let batchSizes: [Int32] = [1, 2, 4, 8, 16, 32, 64, 128]
var results9B: [(M: Int32, mean: Double, std: Double)] = []

for M in batchSizes {
    let r = benchmark9B(M: M)
    results9B.append((M: M, mean: r.mean, std: r.std))
    let gridH = (Int(M) + 63) / 64
    print("  \(String(format: "%3d", M))  \(String(format: "%7.3f", r.mean))  \(String(format: "%5.3f", r.std))   \(String(format: "%5.2f", r.mean / results9B[0].mean))x  \(gridH)")
}

// ── Test 4: 4 kernels (full MLP) at varying M ──────────────────

print("\n=== Test 4: 9B Full MLP (4 kernels) at varying M ===")

// Load all MLP weights
let gWBuf = loadBin("mlp_gate_weight_raw.bin").withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)! }
let gSBuf = loadBin("mlp_gate_scales_bf16.bin").withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)! }
let gBBuf = loadBin("mlp_gate_biases_bf16.bin").withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)! }
let dWBuf = loadBin("mlp_down_weight_raw.bin").withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)! }
let dSBuf = loadBin("mlp_down_scales_bf16.bin").withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)! }
let dBBuf = loadBin("mlp_down_biases_bf16.bin").withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)! }

let siluSrc = """
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
    out[tid] = bfloat((g / (1.0f + exp(-g))) * u);
}
"""
let siluPSO = try! device.makeComputePipelineState(function: try! device.makeLibrary(source: siluSrc, options: nil).makeFunction(name: "silu_multiply_bf16")!)

print("  M   Time(ms)  Std     Ratio  Compute-only(ms)")
print("  ─── ──────── ──────  ───── ────────────────")

var resultsMLP: [(M: Int32, mean: Double)] = []

for M in batchSizes {
    let interCount = Int(M) * 12288
    let outCount = Int(M) * 4096
    var inputBF16 = [UInt16](repeating: 0, count: Int(M) * 4096)
    for i in 0..<inputBF16.count { inputBF16[i] = UInt16(Float.random(in: -1...1).bitPattern >> 16) }

    let xBuf = device.makeBuffer(bytes: &inputBF16, length: Int(M) * 4096 * 2, options: .storageModeShared)!
    let gateOut = device.makeBuffer(length: interCount * 2, options: .storageModeShared)!
    let upOut = device.makeBuffer(length: interCount * 2, options: .storageModeShared)!
    let hiddenOut = device.makeBuffer(length: interCount * 2, options: .storageModeShared)!
    let downOut = device.makeBuffer(length: outCount * 2, options: .storageModeShared)!

    var K4096: Int32 = 4096, K12288: Int32 = 12288
    var N12288: Int32 = 12288, N4096: Int32 = 4096
    var Mv = M

    // Warmup
    for _ in 0..<3 {
        let cmd = commandQueue.makeCommandBuffer()!
        let enc = cmd.makeComputeCommandEncoder()!
        // gate
        enc.setComputePipelineState(qmmPSO)
        enc.setBuffer(gWBuf, offset: 0, index: 0); enc.setBuffer(gSBuf, offset: 0, index: 1); enc.setBuffer(gBBuf, offset: 0, index: 2)
        enc.setBuffer(xBuf, offset: 0, index: 3); enc.setBuffer(gateOut, offset: 0, index: 4)
        enc.setBytes(&K4096, length: 4, index: 5); enc.setBytes(&N12288, length: 4, index: 6); enc.setBytes(&Mv, length: 4, index: 7)
        enc.dispatchThreadgroups(MTLSize(width: 192, height: max(1, (Int(M)+63)/64), depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 2, depth: 2))
        // up
        enc.setBuffer(wBuf, offset: 0, index: 0); enc.setBuffer(sBuf, offset: 0, index: 1); enc.setBuffer(bBuf, offset: 0, index: 2)
        enc.setBuffer(xBuf, offset: 0, index: 3); enc.setBuffer(upOut, offset: 0, index: 4)
        enc.dispatchThreadgroups(MTLSize(width: 192, height: max(1, (Int(M)+63)/64), depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 2, depth: 2))
        // silu_mul
        var n = UInt32(interCount)
        enc.setComputePipelineState(siluPSO)
        enc.setBuffer(gateOut, offset: 0, index: 0); enc.setBuffer(upOut, offset: 0, index: 1); enc.setBuffer(hiddenOut, offset: 0, index: 2)
        enc.setBytes(&n, length: 4, index: 3)
        let tpgW = siluPSO.maxTotalThreadsPerThreadgroup
        enc.dispatchThreadgroups(MTLSize(width: (interCount + tpgW - 1) / tpgW, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: tpgW, height: 1, depth: 1))
        // down
        enc.setComputePipelineState(qmmPSO)
        enc.setBuffer(dWBuf, offset: 0, index: 0); enc.setBuffer(dSBuf, offset: 0, index: 1); enc.setBuffer(dBBuf, offset: 0, index: 2)
        enc.setBuffer(hiddenOut, offset: 0, index: 3); enc.setBuffer(downOut, offset: 0, index: 4)
        enc.setBytes(&K12288, length: 4, index: 5); enc.setBytes(&N4096, length: 4, index: 6); enc.setBytes(&Mv, length: 4, index: 7)
        enc.dispatchThreadgroups(MTLSize(width: 64, height: max(1, (Int(M)+63)/64), depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 2, depth: 2))
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }

    let tpg2 = siluPSO.maxTotalThreadsPerThreadgroup

    // Measure
    var times: [Double] = []
    for _ in 0..<20 {
        let start = CFAbsoluteTimeGetCurrent()
        let cmd = commandQueue.makeCommandBuffer()!
        let enc = cmd.makeComputeCommandEncoder()!
        // gate
        enc.setComputePipelineState(qmmPSO)
        enc.setBuffer(gWBuf, offset: 0, index: 0); enc.setBuffer(gSBuf, offset: 0, index: 1); enc.setBuffer(gBBuf, offset: 0, index: 2)
        enc.setBuffer(xBuf, offset: 0, index: 3); enc.setBuffer(gateOut, offset: 0, index: 4)
        enc.setBytes(&K4096, length: 4, index: 5); enc.setBytes(&N12288, length: 4, index: 6); enc.setBytes(&Mv, length: 4, index: 7)
        enc.dispatchThreadgroups(MTLSize(width: 192, height: max(1, (Int(M)+63)/64), depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 2, depth: 2))
        // up
        enc.setBuffer(wBuf, offset: 0, index: 0); enc.setBuffer(sBuf, offset: 0, index: 1); enc.setBuffer(bBuf, offset: 0, index: 2)
        enc.setBuffer(xBuf, offset: 0, index: 3); enc.setBuffer(upOut, offset: 0, index: 4)
        enc.dispatchThreadgroups(MTLSize(width: 192, height: max(1, (Int(M)+63)/64), depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 2, depth: 2))
        // silu_mul
        var n = UInt32(interCount)
        enc.setComputePipelineState(siluPSO)
        enc.setBuffer(gateOut, offset: 0, index: 0); enc.setBuffer(upOut, offset: 0, index: 1); enc.setBuffer(hiddenOut, offset: 0, index: 2)
        enc.setBytes(&n, length: 4, index: 3)
        enc.dispatchThreadgroups(MTLSize(width: (interCount + tpg2 - 1) / tpg2, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: tpg2, height: 1, depth: 1))
        // down
        enc.setComputePipelineState(qmmPSO)
        enc.setBuffer(dWBuf, offset: 0, index: 0); enc.setBuffer(dSBuf, offset: 0, index: 1); enc.setBuffer(dBBuf, offset: 0, index: 2)
        enc.setBuffer(hiddenOut, offset: 0, index: 3); enc.setBuffer(downOut, offset: 0, index: 4)
        enc.setBytes(&K12288, length: 4, index: 5); enc.setBytes(&N4096, length: 4, index: 6); enc.setBytes(&Mv, length: 4, index: 7)
        enc.dispatchThreadgroups(MTLSize(width: 64, height: max(1, (Int(M)+63)/64), depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 2, depth: 2))
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        times.append((CFAbsoluteTimeGetCurrent() - start) * 1000)
    }

    let mean = times.reduce(0, +) / Double(times.count)
    let std = sqrt(times.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Double(times.count))
    resultsMLP.append((M: M, mean: mean))
    let computeOnly = mean - dispatchMean  // subtract dispatch overhead
    print("  \(String(format: "%3d", M))  \(String(format: "%7.3f", mean))  \(String(format: "%5.3f", std))   \(String(format: "%5.2f", mean / resultsMLP[0].mean))x  \(String(format: "%.3f", computeOnly))")
}

// ── Summary ─────────────────────────────────────────────────────

print("\n═══════════════════════════════════════════")
print("PLATEAU CHALLENGE")
print("═══════════════════════════════════════════")
print()
print("Dispatch overhead: \(String(format: "%.3f", dispatchMean)) ms (empty CB)")
print()
print("Single matmul (9B, K=4096 N=12288):")
let r1_1 = results9B[0].mean
let r1_8 = results9B.first(where: { $0.M == 8 })!.mean
let r1_32 = results9B.first(where: { $0.M == 32 })!.mean
let r1_128 = results9B.last!.mean
print("  M=1: \(String(format: "%.3f", r1_1))ms, M=8: \(String(format: "%.3f", r1_8))ms (\(String(format: "%.2f", r1_8/r1_1))x), M=32: \(String(format: "%.3f", r1_32))ms (\(String(format: "%.2f", r1_32/r1_1))x), M=128: \(String(format: "%.3f", r1_128))ms (\(String(format: "%.2f", r1_128/r1_1))x)")
print()
print("Full MLP (9B, 4 kernels):")
let r2_1 = resultsMLP[0].mean
let r2_8 = resultsMLP.first(where: { $0.M == 8 })!.mean
let r2_32 = resultsMLP.first(where: { $0.M == 32 })!.mean
let r2_128 = resultsMLP.last!.mean
print("  M=1: \(String(format: "%.3f", r2_1))ms, M=8: \(String(format: "%.3f", r2_8))ms (\(String(format: "%.2f", r2_8/r2_1))x), M=32: \(String(format: "%.3f", r2_32))ms (\(String(format: "%.2f", r2_32/r2_1))x), M=128: \(String(format: "%.3f", r2_128))ms (\(String(format: "%.2f", r2_128/r2_1))x)")
print()

// Is it plateau or dispatch overhead?
let computeM1 = r2_1 - dispatchMean
let computeM8 = r2_8 - dispatchMean
let computeM32 = r2_32 - dispatchMean
print("After subtracting dispatch overhead (\(String(format: "%.3f", dispatchMean))ms):")
print("  M=1: \(String(format: "%.3f", computeM1))ms, M=8: \(String(format: "%.3f", computeM8))ms (\(String(format: "%.2f", computeM8/computeM1))x), M=32: \(String(format: "%.3f", computeM32))ms (\(String(format: "%.2f", computeM32/computeM1))x)")
print()

if computeM8 / computeM1 < 1.2 {
    print("VERDICT: The plateau is REAL (compute-only M=8/M=1 = \(String(format: "%.2f", computeM8/computeM1))x)")
    print("  Even after removing dispatch overhead, batch compute is flat.")
    print("  NAX weight caching is the mechanism, not just overhead amortization.")
} else {
    print("VERDICT: The plateau is partially dispatch overhead")
    print("  Compute-only ratio M=8/M=1 = \(String(format: "%.2f", computeM8/computeM1))x")
    print("  Some real compute scaling exists, but still sublinear.")
}
