// batch_verify_27b.swift — IAM Step 5: Batch verification through 27B MLP
//
// Core spec decode insight: verifying K draft tokens costs ~same as K=1
// because NAX loads weights once and processes all tokens from cache.
//
// Test: dispatch 27B MLP with M=1,2,4,8,16,32 (simulating batch verification
// of K draft tokens). Measure wall-clock per-layer time.
//
// Gate: M=8 time within 1.5x of M=1

import Metal
import Foundation

let device = MTLCreateSystemDefaultDevice()!
let commandQueue = device.makeCommandQueue()!
let mlxLib = try! device.makeLibrary(URL: URL(fileURLWithPath: "/Users/midas/.mlx-env/lib/python3.11/site-packages/mlx/lib/mlx.metallib"))
let refDir = "/Users/midas/Desktop/cowork/inference-across-metal/reference_data"

print("Device: \(device.name)")
print()

// ── Load 27B layer 0 weights ────────────────────────────────────

struct TensorRef { let fileIdx: Int, offset: Int, size: Int }

let indexData = try! Data(contentsOf: URL(fileURLWithPath: "\(refDir)/27b_mlp_index.json"))
let index = try! JSONSerialization.jsonObject(with: indexData) as! [String: Any]
let files = index["files"] as! [String]
let hiddenSize = index["hidden_size"] as! Int      // 5120
let intermediateSize = index["intermediate_size"] as! Int  // 17408
let layerDicts = index["layers"] as! [[String: Any]]

func parseRef(_ dict: [String: Any], _ key: String) -> TensorRef {
    let d = dict[key] as! [String: Any]
    return TensorRef(fileIdx: d["file"] as! Int, offset: d["offset"] as! Int, size: d["size"] as! Int)
}

// mmap files
var filePtrs: [UnsafeMutablePointer<UInt8>] = []
var fileSizes: [Int] = []
var fileFds: [Int32] = []

for path in files {
    let fd = open(path, O_RDONLY)
    guard fd >= 0 else { fatalError("Cannot open \(path)") }
    var st = stat()
    fstat(fd, &st)
    let size = Int(st.st_size)
    let ptr = mmap(nil, size, PROT_READ, MAP_PRIVATE, fd, 0)
    guard ptr != MAP_FAILED else { fatalError("mmap failed") }
    filePtrs.append(ptr!.assumingMemoryBound(to: UInt8.self))
    fileSizes.append(size)
    fileFds.append(fd)
}

func bufFromMmap(_ ref: TensorRef) -> MTLBuffer {
    device.makeBuffer(bytes: filePtrs[ref.fileIdx] + ref.offset, length: ref.size, options: .storageModeShared)!
}

let l0 = layerDicts[0] as [String: Any]
let gWBuf = bufFromMmap(parseRef(l0, "gate_proj_weight"))
let gSBuf = bufFromMmap(parseRef(l0, "gate_proj_scales"))
let gBBuf = bufFromMmap(parseRef(l0, "gate_proj_biases"))
let uWBuf = bufFromMmap(parseRef(l0, "up_proj_weight"))
let uSBuf = bufFromMmap(parseRef(l0, "up_proj_scales"))
let uBBuf = bufFromMmap(parseRef(l0, "up_proj_biases"))
let dWBuf = bufFromMmap(parseRef(l0, "down_proj_weight"))
let dSBuf = bufFromMmap(parseRef(l0, "down_proj_scales"))
let dBBuf = bufFromMmap(parseRef(l0, "down_proj_biases"))

print("Loaded layer 0 weights from mmap")

// ── Setup kernels ───────────────────────────────────────────────

let kernelName = "affine_qmm_t_nax_bfloat16_t_gs_64_b_4_bm64_bn64_bk64_wm2_wn2_alN_true_batch_0"
let qmmFn = mlxLib.makeFunction(name: kernelName)!
let qmmPSO = try! device.makeComputePipelineState(function: qmmFn)

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
let siluLib = try! device.makeLibrary(source: siluSrc, options: nil)
let siluPSO = try! device.makeComputePipelineState(function: siluLib.makeFunction(name: "silu_multiply_bf16")!)

// ── Dispatch helpers ────────────────────────────────────────────

func encodeQMM(_ enc: MTLComputeCommandEncoder,
               w: MTLBuffer, s: MTLBuffer, b: MTLBuffer,
               x: MTLBuffer, y: MTLBuffer,
               M: Int32, K: Int32, N: Int32) {
    var K = K, N = N, M = M
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
    enc.dispatchThreadgroups(
        MTLSize(width: (Int(N) + BN - 1) / BN, height: (Int(M) + BM - 1) / BM, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 32, height: 2, depth: 2))
}

func encodeSiluMul(_ enc: MTLComputeCommandEncoder,
                   gate: MTLBuffer, up: MTLBuffer, out: MTLBuffer, count: Int) {
    var n = UInt32(count)
    enc.setComputePipelineState(siluPSO)
    enc.setBuffer(gate, offset: 0, index: 0)
    enc.setBuffer(up, offset: 0, index: 1)
    enc.setBuffer(out, offset: 0, index: 2)
    enc.setBytes(&n, length: 4, index: 3)
    let tpg = siluPSO.maxTotalThreadsPerThreadgroup
    enc.dispatchThreadgroups(
        MTLSize(width: (count + tpg - 1) / tpg, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: tpg, height: 1, depth: 1))
}

// ── Benchmark: vary M (batch size = number of draft tokens) ─────

print("\n=== Batch Verification: 27B MLP Layer 0 ===")
print("Varying M (draft tokens) while K=\(hiddenSize), N=\(intermediateSize)")
print()

let batchSizes: [Int32] = [1, 2, 4, 8, 16, 32]
var results: [(M: Int32, mean: Double, std: Double)] = []

for batchM in batchSizes {
    let M = batchM
    let inputCount = Int(M) * hiddenSize
    let interCount = Int(M) * intermediateSize
    let outputCount = Int(M) * hiddenSize

    // Create random input
    var inputBF16 = [UInt16](repeating: 0, count: inputCount)
    for i in 0..<inputCount {
        let f = Float.random(in: -1...1)
        inputBF16[i] = UInt16(f.bitPattern >> 16)
    }
    let xBuf = device.makeBuffer(bytes: &inputBF16, length: inputCount * 2, options: .storageModeShared)!
    let gateOut = device.makeBuffer(length: interCount * 2, options: .storageModeShared)!
    let upOut = device.makeBuffer(length: interCount * 2, options: .storageModeShared)!
    let hiddenOut = device.makeBuffer(length: interCount * 2, options: .storageModeShared)!
    let downOut = device.makeBuffer(length: outputCount * 2, options: .storageModeShared)!

    // Warmup
    for _ in 0..<3 {
        let cmd = commandQueue.makeCommandBuffer()!
        let enc = cmd.makeComputeCommandEncoder()!
        encodeQMM(enc, w: gWBuf, s: gSBuf, b: gBBuf, x: xBuf, y: gateOut, M: M, K: Int32(hiddenSize), N: Int32(intermediateSize))
        encodeQMM(enc, w: uWBuf, s: uSBuf, b: uBBuf, x: xBuf, y: upOut, M: M, K: Int32(hiddenSize), N: Int32(intermediateSize))
        encodeSiluMul(enc, gate: gateOut, up: upOut, out: hiddenOut, count: interCount)
        encodeQMM(enc, w: dWBuf, s: dSBuf, b: dBBuf, x: hiddenOut, y: downOut, M: M, K: Int32(intermediateSize), N: Int32(hiddenSize))
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }

    // Benchmark
    var times: [Double] = []
    for _ in 0..<20 {
        let start = CFAbsoluteTimeGetCurrent()
        let cmd = commandQueue.makeCommandBuffer()!
        let enc = cmd.makeComputeCommandEncoder()!
        encodeQMM(enc, w: gWBuf, s: gSBuf, b: gBBuf, x: xBuf, y: gateOut, M: M, K: Int32(hiddenSize), N: Int32(intermediateSize))
        encodeQMM(enc, w: uWBuf, s: uSBuf, b: uBBuf, x: xBuf, y: upOut, M: M, K: Int32(hiddenSize), N: Int32(intermediateSize))
        encodeSiluMul(enc, gate: gateOut, up: upOut, out: hiddenOut, count: interCount)
        encodeQMM(enc, w: dWBuf, s: dSBuf, b: dBBuf, x: hiddenOut, y: downOut, M: M, K: Int32(intermediateSize), N: Int32(hiddenSize))
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        times.append((CFAbsoluteTimeGetCurrent() - start) * 1000)
    }

    let mean = times.reduce(0, +) / Double(times.count)
    let std = sqrt(times.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Double(times.count))
    results.append((M: M, mean: mean, std: std))

    // Verify non-zero
    let ptr = downOut.contents().bindMemory(to: UInt16.self, capacity: outputCount)
    let bf16ToFloat: (UInt16) -> Float = { Float(bitPattern: UInt32($0) << 16) }
    var nz = 0
    for i in 0..<outputCount { if abs(bf16ToFloat(ptr[i])) > 1e-6 { nz += 1 } }

    print("  M=\(String(format: "%2d", M)): \(String(format: "%6.2f", mean)) ms (std \(String(format: "%.2f", std))) | non-zero=\(nz)/\(outputCount) | ratio=\(String(format: "%.2f", mean / results[0].mean))x")
}

// ── Multi-layer batch verify (64 layers, M=1 vs M=8) ───────────

print("\n=== Full 64-Layer Batch Verify ===")

// Pre-load all layer weights
var allW: [(gW: MTLBuffer, gS: MTLBuffer, gB: MTLBuffer,
            uW: MTLBuffer, uS: MTLBuffer, uB: MTLBuffer,
            dW: MTLBuffer, dS: MTLBuffer, dB: MTLBuffer)] = []
for ld in layerDicts {
    let d = ld as [String: Any]
    allW.append((
        gW: bufFromMmap(parseRef(d, "gate_proj_weight")), gS: bufFromMmap(parseRef(d, "gate_proj_scales")), gB: bufFromMmap(parseRef(d, "gate_proj_biases")),
        uW: bufFromMmap(parseRef(d, "up_proj_weight")), uS: bufFromMmap(parseRef(d, "up_proj_scales")), uB: bufFromMmap(parseRef(d, "up_proj_biases")),
        dW: bufFromMmap(parseRef(d, "down_proj_weight")), dS: bufFromMmap(parseRef(d, "down_proj_scales")), dB: bufFromMmap(parseRef(d, "down_proj_biases"))
    ))
}

for testM in [Int32(1), Int32(8)] {
    let inputCount = Int(testM) * hiddenSize
    let interCount = Int(testM) * intermediateSize

    var inputBF16 = [UInt16](repeating: 0, count: inputCount)
    for i in 0..<inputCount { inputBF16[i] = UInt16(Float.random(in: -1...1).bitPattern >> 16) }

    let slotA = device.makeBuffer(bytes: &inputBF16, length: inputCount * 2, options: .storageModeShared)!
    let slotB = device.makeBuffer(length: inputCount * 2, options: .storageModeShared)!
    let gBuf = device.makeBuffer(length: interCount * 2, options: .storageModeShared)!
    let uBuf = device.makeBuffer(length: interCount * 2, options: .storageModeShared)!
    let hBuf = device.makeBuffer(length: interCount * 2, options: .storageModeShared)!

    // Warmup
    for _ in 0..<2 {
        let cmd = commandQueue.makeCommandBuffer()!
        for (li, w) in allW.prefix(4).enumerated() {
            let inB = (li % 2 == 0) ? slotA : slotB
            let outB = (li % 2 == 0) ? slotB : slotA
            let enc = cmd.makeComputeCommandEncoder()!
            encodeQMM(enc, w: w.gW, s: w.gS, b: w.gB, x: inB, y: gBuf, M: testM, K: Int32(hiddenSize), N: Int32(intermediateSize))
            encodeQMM(enc, w: w.uW, s: w.uS, b: w.uB, x: inB, y: uBuf, M: testM, K: Int32(hiddenSize), N: Int32(intermediateSize))
            encodeSiluMul(enc, gate: gBuf, up: uBuf, out: hBuf, count: interCount)
            encodeQMM(enc, w: w.dW, s: w.dS, b: w.dB, x: hBuf, y: outB, M: testM, K: Int32(intermediateSize), N: Int32(hiddenSize))
            enc.endEncoding()
        }
        cmd.commit()
        cmd.waitUntilCompleted()
    }

    // Benchmark
    var times: [Double] = []
    for _ in 0..<5 {
        memcpy(slotA.contents(), &inputBF16, inputCount * 2)
        let start = CFAbsoluteTimeGetCurrent()
        let cmd = commandQueue.makeCommandBuffer()!
        for (li, w) in allW.enumerated() {
            let inB = (li % 2 == 0) ? slotA : slotB
            let outB = (li % 2 == 0) ? slotB : slotA
            let enc = cmd.makeComputeCommandEncoder()!
            encodeQMM(enc, w: w.gW, s: w.gS, b: w.gB, x: inB, y: gBuf, M: testM, K: Int32(hiddenSize), N: Int32(intermediateSize))
            encodeQMM(enc, w: w.uW, s: w.uS, b: w.uB, x: inB, y: uBuf, M: testM, K: Int32(hiddenSize), N: Int32(intermediateSize))
            encodeSiluMul(enc, gate: gBuf, up: uBuf, out: hBuf, count: interCount)
            encodeQMM(enc, w: w.dW, s: w.dS, b: w.dB, x: hBuf, y: outB, M: testM, K: Int32(intermediateSize), N: Int32(hiddenSize))
            enc.endEncoding()
        }
        cmd.commit()
        cmd.waitUntilCompleted()
        times.append((CFAbsoluteTimeGetCurrent() - start) * 1000)
    }

    let mean = times.reduce(0, +) / Double(times.count)
    print("  M=\(testM) (64 layers): \(String(format: "%.0f", mean)) ms (\(String(format: "%.1f", mean / 64)) ms/layer)")
}

// ── Summary ─────────────────────────────────────────────────────

print("\n═══════════════════════════════════════════")
print("SUMMARY — Batch Verification Plateau")
print("═══════════════════════════════════════════")

let m1 = results[0].mean  // M=1
let m8 = results.first(where: { $0.M == 8 })!.mean
let m32 = results.last!.mean

print("Single layer (27B MLP):")
for r in results {
    let bar = String(repeating: "█", count: Int(r.mean / m1 * 20))
    print("  M=\(String(format: "%2d", r.M)): \(String(format: "%6.2f", r.mean)) ms  \(String(format: "%.2f", r.mean / m1))x  \(bar)")
}

print()
print("M=1 → M=8  ratio: \(String(format: "%.2f", m8 / m1))x")
print("M=1 → M=32 ratio: \(String(format: "%.2f", m32 / m1))x")
print()

// Gate: M=8 within 1.5x of M=1
let gatePass = (m8 / m1) <= 1.5
print("GATE: M=8/M=1 = \(String(format: "%.2f", m8 / m1))x → \(gatePass ? "PASS" : "FAIL")")
if gatePass {
    print("  Batch verification plateau confirmed on 27B.")
    print("  Verifying 8 draft tokens costs ~same as 1.")
    print("  NAX loads weights once, processes all tokens from cache.")
    print("  This is the core insight enabling spec decode on Apple Silicon.")
} else {
    print("  Batch verification does NOT plateau — linear scaling.")
}

// Cleanup
for i in 0..<files.count {
    munmap(UnsafeMutableRawPointer(filePtrs[i]), fileSizes[i])
    close(fileFds[i])
}
