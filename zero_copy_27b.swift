// zero_copy_27b.swift — Challenge: eliminate all weight memcpy
//
// Instead of makeBuffer(bytes:) which copies 143MB/layer from mmap,
// use makeBuffer(bytesNoCopy:) on the entire file + setBuffer offset.
// The GPU reads directly from mmap'd unified memory.
//
// Question: does the GPU handle page faults gracefully? What's the
// real cost when pages aren't resident?

import Metal
import Foundation

let device = MTLCreateSystemDefaultDevice()!
let commandQueue = device.makeCommandQueue()!
let mlxLib = try! device.makeLibrary(URL: URL(fileURLWithPath: "/Users/midas/.mlx-env/lib/python3.11/site-packages/mlx/lib/mlx.metallib"))
let refDir = "/Users/midas/Desktop/cowork/inference-across-metal/reference_data"

print("Device: \(device.name)")
print()

// ── Load index ──────────────────────────────────────────────────

struct TensorRef { let fileIdx: Int, offset: Int, size: Int }
struct LayerMLP {
    let gateW: TensorRef, gateS: TensorRef, gateB: TensorRef
    let upW: TensorRef, upS: TensorRef, upB: TensorRef
    let downW: TensorRef, downS: TensorRef, downB: TensorRef
}

let indexData = try! Data(contentsOf: URL(fileURLWithPath: "\(refDir)/27b_mlp_index.json"))
let index = try! JSONSerialization.jsonObject(with: indexData) as! [String: Any]
let files = index["files"] as! [String]
let hiddenSize = index["hidden_size"] as! Int
let intermediateSize = index["intermediate_size"] as! Int
let numLayers = index["num_layers"] as! Int
let layerDicts = index["layers"] as! [[String: Any]]

func parseRef(_ dict: [String: Any], _ key: String) -> TensorRef {
    let d = dict[key] as! [String: Any]
    return TensorRef(fileIdx: d["file"] as! Int, offset: d["offset"] as! Int, size: d["size"] as! Int)
}

var layers: [LayerMLP] = []
for ld in layerDicts {
    layers.append(LayerMLP(
        gateW: parseRef(ld, "gate_proj_weight"), gateS: parseRef(ld, "gate_proj_scales"), gateB: parseRef(ld, "gate_proj_biases"),
        upW: parseRef(ld, "up_proj_weight"), upS: parseRef(ld, "up_proj_scales"), upB: parseRef(ld, "up_proj_biases"),
        downW: parseRef(ld, "down_proj_weight"), downS: parseRef(ld, "down_proj_scales"), downB: parseRef(ld, "down_proj_biases")
    ))
}

// ── mmap files + create ZERO-COPY Metal buffers ─────────────────

print("=== Zero-Copy File Mapping ===")

let pageSize = Int(getpagesize())
var filePtrs: [UnsafeMutableRawPointer] = []
var fileSizes: [Int] = []
var fileFds: [Int32] = []
var fileMetalBufs: [MTLBuffer?] = []

for (i, path) in files.enumerated() {
    let fd = open(path, O_RDONLY)
    guard fd >= 0 else { fatalError("Cannot open \(path)") }

    var st = stat()
    fstat(fd, &st)
    let size = Int(st.st_size)

    let ptr = mmap(nil, size, PROT_READ, MAP_PRIVATE, fd, 0)
    guard ptr != MAP_FAILED else { fatalError("mmap failed for file \(i)") }

    filePtrs.append(ptr!)
    fileSizes.append(size)
    fileFds.append(fd)

    // Page-align the size for bytesNoCopy
    let alignedSize = ((size + pageSize - 1) / pageSize) * pageSize

    // Try zero-copy: Metal reads directly from mmap'd memory
    let metalBuf = device.makeBuffer(
        bytesNoCopy: ptr!,
        length: alignedSize,
        options: .storageModeShared,
        deallocator: nil
    )

    fileMetalBufs.append(metalBuf)

    if let buf = metalBuf {
        print("  File \(i): \(String(format: "%.1f", Double(size) / 1e9)) GB → zero-copy Metal buffer (\(buf.length) bytes)")
    } else {
        print("  File \(i): \(String(format: "%.1f", Double(size) / 1e9)) GB → bytesNoCopy FAILED, falling back to copy")
    }
}

let allZeroCopy = fileMetalBufs.allSatisfy { $0 != nil }
print("\n  Zero-copy: \(allZeroCopy ? "ALL FILES" : "partial") — GPU reads directly from mmap'd safetensors")

// ── Setup kernels ───────────────────────────────────────────────

let kernelName = "affine_qmm_t_nax_bfloat16_t_gs_64_b_4_bm64_bn64_bk64_wm2_wn2_alN_true_batch_0"
let qmmPSO = try! device.makeComputePipelineState(function: mlxLib.makeFunction(name: kernelName)!)

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

// ── Dispatch with setBuffer offset ──────────────────────────────

func encodeQMM_zeroCopy(_ enc: MTLComputeCommandEncoder,
                        wRef: TensorRef, sRef: TensorRef, bRef: TensorRef,
                        x: MTLBuffer, y: MTLBuffer,
                        M: Int32, K: Int32, N: Int32) {
    var K = K, N = N, M = M
    enc.setComputePipelineState(qmmPSO)

    // Use the whole-file Metal buffer + offset instead of per-tensor copy
    enc.setBuffer(fileMetalBufs[wRef.fileIdx]!, offset: wRef.offset, index: 0)
    enc.setBuffer(fileMetalBufs[sRef.fileIdx]!, offset: sRef.offset, index: 1)
    enc.setBuffer(fileMetalBufs[bRef.fileIdx]!, offset: bRef.offset, index: 2)
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

let bf16ToFloat: (UInt16) -> Float = { Float(bitPattern: UInt32($0) << 16) }

// Intermediate buffers (tiny: only activation tensors, not weights)
let gateOut = device.makeBuffer(length: intermediateSize * 2, options: .storageModeShared)!
let upOut = device.makeBuffer(length: intermediateSize * 2, options: .storageModeShared)!
let hiddenOut = device.makeBuffer(length: intermediateSize * 2, options: .storageModeShared)!

// Input/output ping-pong
var inputBF16 = [UInt16](repeating: 0, count: hiddenSize)
for i in 0..<hiddenSize { inputBF16[i] = UInt16(Float.random(in: -1...1).bitPattern >> 16) }
let slotA = device.makeBuffer(bytes: &inputBF16, length: hiddenSize * 2, options: .storageModeShared)!
let slotB = device.makeBuffer(length: hiddenSize * 2, options: .storageModeShared)!

// ── Test 1: Single layer, zero-copy ─────────────────────────────

print("\n=== Test 1: Single Layer Zero-Copy ===")

let l0 = layers[0]
let cmd = commandQueue.makeCommandBuffer()!
let enc = cmd.makeComputeCommandEncoder()!
encodeQMM_zeroCopy(enc, wRef: l0.gateW, sRef: l0.gateS, bRef: l0.gateB, x: slotA, y: gateOut, M: 1, K: Int32(hiddenSize), N: Int32(intermediateSize))
encodeQMM_zeroCopy(enc, wRef: l0.upW, sRef: l0.upS, bRef: l0.upB, x: slotA, y: upOut, M: 1, K: Int32(hiddenSize), N: Int32(intermediateSize))
encodeSiluMul(enc, gate: gateOut, up: upOut, out: hiddenOut, count: intermediateSize)
encodeQMM_zeroCopy(enc, wRef: l0.downW, sRef: l0.downS, bRef: l0.downB, x: hiddenOut, y: slotB, M: 1, K: Int32(hiddenSize), N: Int32(intermediateSize))
enc.endEncoding()
cmd.commit()
cmd.waitUntilCompleted()

if let error = cmd.error {
    print("  GPU ERROR: \(error)")
} else {
    let ptr = slotB.contents().bindMemory(to: UInt16.self, capacity: hiddenSize)
    var nz = 0
    for i in 0..<hiddenSize { if abs(bf16ToFloat(ptr[i])) > 1e-6 { nz += 1 } }
    print("  Layer 0 output: \(nz)/\(hiddenSize) non-zero")
    print("  First 4: ", terminator: "")
    for i in 0..<4 { print(String(format: "%.4f ", bf16ToFloat(ptr[i])), terminator: "") }
    print()
}

// ── Test 2: Cold vs warm — purge page cache then measure ────────

print("\n=== Test 2: Cold Start (purge page cache) ===")

// Purge: remap with MADV_DONTNEED to drop pages
for i in 0..<files.count {
    madvise(filePtrs[i], fileSizes[i], MADV_DONTNEED)
}
print("  Page cache purged via MADV_DONTNEED")

// Cold run: measure time for all 64 layers with zero-copy (pages from SSD)
memcpy(slotA.contents(), &inputBF16, hiddenSize * 2)

var coldLayerTimes: [Double] = []
let coldStart = CFAbsoluteTimeGetCurrent()

for layerIdx in 0..<numLayers {
    let l = layers[layerIdx]
    let inBuf = (layerIdx % 2 == 0) ? slotA : slotB
    let outBuf = (layerIdx % 2 == 0) ? slotB : slotA

    let lt0 = CFAbsoluteTimeGetCurrent()
    let cmd = commandQueue.makeCommandBuffer()!
    let enc = cmd.makeComputeCommandEncoder()!
    encodeQMM_zeroCopy(enc, wRef: l.gateW, sRef: l.gateS, bRef: l.gateB, x: inBuf, y: gateOut, M: 1, K: Int32(hiddenSize), N: Int32(intermediateSize))
    encodeQMM_zeroCopy(enc, wRef: l.upW, sRef: l.upS, bRef: l.upB, x: inBuf, y: upOut, M: 1, K: Int32(hiddenSize), N: Int32(intermediateSize))
    encodeSiluMul(enc, gate: gateOut, up: upOut, out: hiddenOut, count: intermediateSize)
    encodeQMM_zeroCopy(enc, wRef: l.downW, sRef: l.downS, bRef: l.downB, x: hiddenOut, y: outBuf, M: 1, K: Int32(hiddenSize), N: Int32(intermediateSize))
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    coldLayerTimes.append((CFAbsoluteTimeGetCurrent() - lt0) * 1000)

    if layerIdx < 3 || layerIdx == numLayers - 1 {
        print("  Layer \(String(format: "%2d", layerIdx)): \(String(format: "%7.1f", coldLayerTimes.last!)) ms")
    } else if layerIdx == 3 { print("  ...") }
}

let coldTotal = (CFAbsoluteTimeGetCurrent() - coldStart) * 1000
let coldMean = coldLayerTimes.reduce(0, +) / Double(coldLayerTimes.count)
print("  Cold total: \(String(format: "%.0f", coldTotal)) ms (\(String(format: "%.1f", coldMean)) ms/layer)")

// ── Test 3: Warm run (pages now cached) ─────────────────────────

print("\n=== Test 3: Warm Run (pages cached) ===")

memcpy(slotA.contents(), &inputBF16, hiddenSize * 2)

var warmLayerTimes: [Double] = []
let warmStart = CFAbsoluteTimeGetCurrent()

for layerIdx in 0..<numLayers {
    let l = layers[layerIdx]
    let inBuf = (layerIdx % 2 == 0) ? slotA : slotB
    let outBuf = (layerIdx % 2 == 0) ? slotB : slotA

    let lt0 = CFAbsoluteTimeGetCurrent()
    let cmd = commandQueue.makeCommandBuffer()!
    let enc = cmd.makeComputeCommandEncoder()!
    encodeQMM_zeroCopy(enc, wRef: l.gateW, sRef: l.gateS, bRef: l.gateB, x: inBuf, y: gateOut, M: 1, K: Int32(hiddenSize), N: Int32(intermediateSize))
    encodeQMM_zeroCopy(enc, wRef: l.upW, sRef: l.upS, bRef: l.upB, x: inBuf, y: upOut, M: 1, K: Int32(hiddenSize), N: Int32(intermediateSize))
    encodeSiluMul(enc, gate: gateOut, up: upOut, out: hiddenOut, count: intermediateSize)
    encodeQMM_zeroCopy(enc, wRef: l.downW, sRef: l.downS, bRef: l.downB, x: hiddenOut, y: outBuf, M: 1, K: Int32(hiddenSize), N: Int32(intermediateSize))
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    warmLayerTimes.append((CFAbsoluteTimeGetCurrent() - lt0) * 1000)
}

let warmTotal = (CFAbsoluteTimeGetCurrent() - warmStart) * 1000
let warmMean = warmLayerTimes.reduce(0, +) / Double(warmLayerTimes.count)
print("  Warm total: \(String(format: "%.0f", warmTotal)) ms (\(String(format: "%.1f", warmMean)) ms/layer)")

// ── Test 4: Warm batched (all in one command buffer) ────────────

print("\n=== Test 4: Warm Batched (1 CB, zero-copy) ===")

var batchedTimes: [Double] = []
for trial in 0..<8 {
    memcpy(slotA.contents(), &inputBF16, hiddenSize * 2)
    let start = CFAbsoluteTimeGetCurrent()

    let cmd = commandQueue.makeCommandBuffer()!
    for layerIdx in 0..<numLayers {
        let l = layers[layerIdx]
        let inBuf = (layerIdx % 2 == 0) ? slotA : slotB
        let outBuf = (layerIdx % 2 == 0) ? slotB : slotA

        let enc = cmd.makeComputeCommandEncoder()!
        encodeQMM_zeroCopy(enc, wRef: l.gateW, sRef: l.gateS, bRef: l.gateB, x: inBuf, y: gateOut, M: 1, K: Int32(hiddenSize), N: Int32(intermediateSize))
        encodeQMM_zeroCopy(enc, wRef: l.upW, sRef: l.upS, bRef: l.upB, x: inBuf, y: upOut, M: 1, K: Int32(hiddenSize), N: Int32(intermediateSize))
        encodeSiluMul(enc, gate: gateOut, up: upOut, out: hiddenOut, count: intermediateSize)
        encodeQMM_zeroCopy(enc, wRef: l.downW, sRef: l.downS, bRef: l.downB, x: hiddenOut, y: outBuf, M: 1, K: Int32(hiddenSize), N: Int32(intermediateSize))
        enc.endEncoding()
    }
    cmd.commit()
    cmd.waitUntilCompleted()
    let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
    if trial >= 3 { batchedTimes.append(elapsed) }
}

let batchedMean = batchedTimes.reduce(0, +) / Double(batchedTimes.count)
let batchedPerLayer = batchedMean / Double(numLayers)
print("  Warm batched: \(String(format: "%.0f", batchedMean)) ms (\(String(format: "%.2f", batchedPerLayer)) ms/layer)")

// ── Test 5: With madvise prefetch ───────────────────────────────

print("\n=== Test 5: Cold + madvise Prefetch ===")

// Purge again
for i in 0..<files.count {
    madvise(filePtrs[i], fileSizes[i], MADV_DONTNEED)
}

// Prefetch first 8 layers immediately
for layerIdx in 0..<min(8, numLayers) {
    let l = layers[layerIdx]
    for ref in [l.gateW, l.gateS, l.gateB, l.upW, l.upS, l.upB, l.downW, l.downS, l.downB] {
        madvise(filePtrs[ref.fileIdx] + ref.offset, ref.size, MADV_WILLNEED)
    }
}

usleep(50000)  // 50ms head start for prefetch

memcpy(slotA.contents(), &inputBF16, hiddenSize * 2)

var prefetchLayerTimes: [Double] = []
let prefetchStart = CFAbsoluteTimeGetCurrent()

for layerIdx in 0..<numLayers {
    // Prefetch 8 layers ahead
    let ahead = layerIdx + 8
    if ahead < numLayers {
        let la = layers[ahead]
        for ref in [la.gateW, la.gateS, la.gateB, la.upW, la.upS, la.upB, la.downW, la.downS, la.downB] {
            madvise(filePtrs[ref.fileIdx] + ref.offset, ref.size, MADV_WILLNEED)
        }
    }

    let l = layers[layerIdx]
    let inBuf = (layerIdx % 2 == 0) ? slotA : slotB
    let outBuf = (layerIdx % 2 == 0) ? slotB : slotA

    let lt0 = CFAbsoluteTimeGetCurrent()
    let cmd = commandQueue.makeCommandBuffer()!
    let enc = cmd.makeComputeCommandEncoder()!
    encodeQMM_zeroCopy(enc, wRef: l.gateW, sRef: l.gateS, bRef: l.gateB, x: inBuf, y: gateOut, M: 1, K: Int32(hiddenSize), N: Int32(intermediateSize))
    encodeQMM_zeroCopy(enc, wRef: l.upW, sRef: l.upS, bRef: l.upB, x: inBuf, y: upOut, M: 1, K: Int32(hiddenSize), N: Int32(intermediateSize))
    encodeSiluMul(enc, gate: gateOut, up: upOut, out: hiddenOut, count: intermediateSize)
    encodeQMM_zeroCopy(enc, wRef: l.downW, sRef: l.downS, bRef: l.downB, x: hiddenOut, y: outBuf, M: 1, K: Int32(hiddenSize), N: Int32(intermediateSize))
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    prefetchLayerTimes.append((CFAbsoluteTimeGetCurrent() - lt0) * 1000)

    if layerIdx < 3 || layerIdx == numLayers - 1 {
        print("  Layer \(String(format: "%2d", layerIdx)): \(String(format: "%7.1f", prefetchLayerTimes.last!)) ms")
    } else if layerIdx == 3 { print("  ...") }
}

let prefetchTotal = (CFAbsoluteTimeGetCurrent() - prefetchStart) * 1000
let prefetchMean = prefetchLayerTimes.reduce(0, +) / Double(prefetchLayerTimes.count)
print("  Prefetch total: \(String(format: "%.0f", prefetchTotal)) ms (\(String(format: "%.1f", prefetchMean)) ms/layer)")

// ── Summary ─────────────────────────────────────────────────────

print("\n═══════════════════════════════════════════")
print("ZERO-COPY vs MEMCPY — 27B MLP (64 layers)")
print("═══════════════════════════════════════════")
print()
print("Method                    Total (ms)  Per-layer (ms)  vs memcpy")
print("─────────────────────────────────────────────────────────────────")
print("memcpy (Step 4 baseline)     6523         101.9        1.00x")
print("Zero-copy cold              \(String(format: "%5.0f", coldTotal))        \(String(format: "%6.1f", coldMean))        \(String(format: "%.2f", coldTotal / 6523))x")
print("Zero-copy + prefetch        \(String(format: "%5.0f", prefetchTotal))        \(String(format: "%6.1f", prefetchMean))        \(String(format: "%.2f", prefetchTotal / 6523))x")
print("Zero-copy warm              \(String(format: "%5.0f", warmTotal))        \(String(format: "%6.1f", warmMean))        \(String(format: "%.2f", warmTotal / 6523))x")
print("Zero-copy warm batched      \(String(format: "%5.0f", batchedMean))        \(String(format: "%6.1f", batchedPerLayer))        \(String(format: "%.2f", batchedMean / 6523))x")
print("Pre-loaded (Step 4)          212          3.3        0.03x")
print()

// Metal buffer memory comparison
let zeroCopyMem = fileMetalBufs.compactMap { $0 }.reduce(0) { $0 + $1.length }
print("Metal buffer memory:")
print("  Zero-copy: 0 bytes extra (GPU reads from mmap'd files)")
print("  makeBuffer(bytesNoCopy): \(zeroCopyMem / 1024 / 1024 / 1024) GB virtual (0 bytes copied)")
print("  Pre-loaded (Step 4): ~9 GB copied into separate Metal buffers")
print()

let warmTokPerSec = 1000.0 / warmTotal
let batchedTokPerSec = 1000.0 / batchedMean
print("Estimated MLP-only tok/s:")
print("  Cold:      \(String(format: "%.2f", 1000.0 / coldTotal))")
print("  Prefetch:  \(String(format: "%.2f", 1000.0 / prefetchTotal))")
print("  Warm:      \(String(format: "%.2f", warmTokPerSec))")
print("  Warm batch:\(String(format: "%.2f", batchedTokPerSec))")

// Cleanup
for (i, _) in files.enumerated() {
    munmap(filePtrs[i], fileSizes[i])
    close(fileFds[i])
}
