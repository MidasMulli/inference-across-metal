// weight_loader.swift — IAM Kill Test
// Measures the cost of filling Metal buffers directly from mmap'd safetensors.
//
// Test: mmap the 9B safetensors file, gather 30 tensor chunks (one layer's
// weights, scattered across the file), copy into a single Metal buffer.
// Compare against mx.load's 6.6ms from B4.
//
// Gate: Metal buffer fill < 3ms → IAM architecture is viable.
//       Metal buffer fill >= 3ms → IAM is Pro-only, pivot to extraction hardening.

import Foundation
import Metal

// ── Config ──────────────────────────────────────────────────────

let safetensorsPath = "/Users/midas/.cache/huggingface/hub/models--mlx-community--Qwen3.5-9B-MLX-4bit/blobs/a68b87558c6ef43f74c2bd63ce7e9092ceddc3101f3def0030774bae5f42aadd"

// Layer 0: 30 tensors, 122,913,216 bytes (117.2 MB), scattered across 4GB span
let tensorOffsets: [(Int, Int)] = [
    (1157315190, 1572864),
    (1709974966, 128),
    (1712598838, 4096),
    (2033082038, 4096),
    (2748772438, 1572864),
    (2812529174, 25165824),
    (2838237974, 25165824),
    (2863403798, 8192),
    (2881295126, 524288),
    (2881819414, 8388608),
    (2890208022, 256),
    (2890210582, 64),
    (2901699926, 65536),
    (2903344726, 524288),
    (2903871318, 524288),
    (2904395606, 8388608),
    (2914885462, 16777216),
    (2931662678, 65536),
    (3440295766, 1572864),
    (3576599382, 8192),
    (3675457366, 4096),
    (3723034742, 1572864),
    (4055578870, 4096),
    (4126352886, 1048576),
    (4510353622, 25165824),
    (4795739414, 524288),
    (4994223094, 1572864),
    (5304768982, 1572864),
    (5316008406, 1048576),
    (5331806422, 65536),
]

let totalLayerBytes = tensorOffsets.reduce(0) { $0 + $1.1 }
let numTrials = 20
let warmupTrials = 3

// ── Setup ───────────────────────────────────────────────────────

guard let device = MTLCreateSystemDefaultDevice() else {
    fatalError("No Metal device")
}
print("Metal device: \(device.name)")
print("Safetensors: \(safetensorsPath)")
print("Layer 0: \(tensorOffsets.count) tensors, \(totalLayerBytes) bytes (\(Double(totalLayerBytes) / 1024 / 1024):.1f MB)")
print("Trials: \(numTrials) (+ \(warmupTrials) warmup)")
print()

// ── mmap the file ───────────────────────────────────────────────

let fd = open(safetensorsPath, O_RDONLY)
guard fd >= 0 else {
    fatalError("Cannot open file: \(String(cString: strerror(errno)))")
}

var st = stat()
fstat(fd, &st)
let fileSize = Int(st.st_size)
print("File size: \(Double(fileSize) / 1024 / 1024 / 1024):.2f GB")

let mmapPtr = mmap(nil, fileSize, PROT_READ, MAP_PRIVATE, fd, 0)
guard mmapPtr != MAP_FAILED else {
    fatalError("mmap failed: \(String(cString: strerror(errno)))")
}
let basePtr = mmapPtr!.assumingMemoryBound(to: UInt8.self)
print("mmap'd successfully")
print()

// ── Test 1: Gather-copy into a single Metal buffer ──────────────
// This simulates loading one layer's weights for a forward pass.
// The tensors are scattered across the file — we gather them into
// one contiguous Metal buffer.

print("TEST 1: Gather \(tensorOffsets.count) tensors → single Metal buffer")
print("─────────────────────────────────────────────────")

var gatherTimes: [Double] = []

for trial in 0..<(warmupTrials + numTrials) {
    let start = CFAbsoluteTimeGetCurrent()

    // Allocate Metal buffer for entire layer
    guard let buffer = device.makeBuffer(length: totalLayerBytes, options: .storageModeShared) else {
        fatalError("Cannot allocate Metal buffer")
    }

    // Gather: copy each tensor from its mmap'd location into the buffer
    let destPtr = buffer.contents().assumingMemoryBound(to: UInt8.self)
    var destOffset = 0
    for (fileOffset, size) in tensorOffsets {
        memcpy(destPtr + destOffset, basePtr + fileOffset, size)
        destOffset += size
    }

    let elapsed = CFAbsoluteTimeGetCurrent() - start

    if trial >= warmupTrials {
        gatherTimes.append(elapsed * 1000) // ms
    }
}

let gatherMean = gatherTimes.reduce(0, +) / Double(gatherTimes.count)
let gatherStd = sqrt(gatherTimes.map { ($0 - gatherMean) * ($0 - gatherMean) }.reduce(0, +) / Double(gatherTimes.count))
let gatherMin = gatherTimes.min()!
let gatherMax = gatherTimes.max()!

print("  Mean: \(String(format: "%.3f", gatherMean)) ms")
print("  Std:  \(String(format: "%.3f", gatherStd)) ms")
print("  Min:  \(String(format: "%.3f", gatherMin)) ms")
print("  Max:  \(String(format: "%.3f", gatherMax)) ms")
print("  Bandwidth: \(String(format: "%.1f", Double(totalLayerBytes) / gatherMean / 1e6)) GB/s")
print()

// ── Test 2: Single contiguous read (best case baseline) ─────────
// What if the layer WAS contiguous? Measures pure memcpy speed.

print("TEST 2: Single contiguous memcpy (117.2 MB, best case)")
print("─────────────────────────────────────────────────")

// Use the largest contiguous block (mlp.up_proj.weight, 25MB) as source
let contigOffset = tensorOffsets[5].0  // 25MB block start
var contigTimes: [Double] = []

for trial in 0..<(warmupTrials + numTrials) {
    let start = CFAbsoluteTimeGetCurrent()

    guard let buffer = device.makeBuffer(length: totalLayerBytes, options: .storageModeShared) else {
        fatalError("Cannot allocate Metal buffer")
    }
    let destPtr = buffer.contents().assumingMemoryBound(to: UInt8.self)
    memcpy(destPtr, basePtr + contigOffset, totalLayerBytes)

    let elapsed = CFAbsoluteTimeGetCurrent() - start

    if trial >= warmupTrials {
        contigTimes.append(elapsed * 1000)
    }
}

let contigMean = contigTimes.reduce(0, +) / Double(contigTimes.count)
let contigStd = sqrt(contigTimes.map { ($0 - contigMean) * ($0 - contigMean) }.reduce(0, +) / Double(contigTimes.count))

print("  Mean: \(String(format: "%.3f", contigMean)) ms")
print("  Std:  \(String(format: "%.3f", contigStd)) ms")
print("  Bandwidth: \(String(format: "%.1f", Double(totalLayerBytes) / contigMean / 1e6)) GB/s")
print()

// ── Test 3: makeBuffer(bytesNoCopy:) — zero-copy for contiguous ──
// If a tensor block is large enough, Metal can use the mmap'd pointer
// directly without copying. Only works for page-aligned, contiguous data.

print("TEST 3: makeBuffer(bytesNoCopy:) — zero-copy (largest tensor, 25MB)")
print("─────────────────────────────────────────────────")

// Use the 25MB mlp.up_proj.weight tensor
let largestOffset = tensorOffsets[5].0
let largestSize = tensorOffsets[5].1
// Page-align the offset
let pageSize = Int(getpagesize())
let alignedOffset = (largestOffset / pageSize) * pageSize
let adjustment = largestOffset - alignedOffset
let alignedSize = ((largestSize + adjustment + pageSize - 1) / pageSize) * pageSize

var noCopyTimes: [Double] = []
var noCopySuccess = false

for trial in 0..<(warmupTrials + numTrials) {
    let start = CFAbsoluteTimeGetCurrent()

    let rawPtr = UnsafeMutableRawPointer(mutating: basePtr + alignedOffset)
    let buffer = device.makeBuffer(
        bytesNoCopy: rawPtr,
        length: alignedSize,
        options: .storageModeShared,
        deallocator: nil
    )

    let elapsed = CFAbsoluteTimeGetCurrent() - start

    if buffer != nil {
        noCopySuccess = true
        if trial >= warmupTrials {
            noCopyTimes.append(elapsed * 1000)
        }
    }
}

if noCopySuccess && !noCopyTimes.isEmpty {
    let noCopyMean = noCopyTimes.reduce(0, +) / Double(noCopyTimes.count)
    let noCopyStd = sqrt(noCopyTimes.map { ($0 - noCopyMean) * ($0 - noCopyMean) }.reduce(0, +) / Double(noCopyTimes.count))
    print("  Mean: \(String(format: "%.4f", noCopyMean)) ms")
    print("  Std:  \(String(format: "%.4f", noCopyStd)) ms")
    print("  (Zero-copy: Metal uses mmap'd pointer directly, no data movement)")
} else {
    print("  FAILED: makeBuffer(bytesNoCopy:) returned nil")
    print("  (Likely alignment issue — pointer must be page-aligned)")
}
print()

// ── Test 4: Per-tensor individual Metal buffers ─────────────────
// Alternative: create separate Metal buffers per tensor (no gather).
// This avoids the gather copy but means more buffer management.

print("TEST 4: Per-tensor Metal buffers (\(tensorOffsets.count) buffers, no gather)")
print("─────────────────────────────────────────────────")

var perTensorTimes: [Double] = []

for trial in 0..<(warmupTrials + numTrials) {
    let start = CFAbsoluteTimeGetCurrent()

    var buffers: [MTLBuffer] = []
    buffers.reserveCapacity(tensorOffsets.count)
    for (fileOffset, size) in tensorOffsets {
        guard let buffer = device.makeBuffer(bytes: basePtr + fileOffset, length: size, options: .storageModeShared) else {
            fatalError("Cannot allocate Metal buffer for tensor")
        }
        buffers.append(buffer)
    }

    let elapsed = CFAbsoluteTimeGetCurrent() - start

    if trial >= warmupTrials {
        perTensorTimes.append(elapsed * 1000)
    }
}

let perTensorMean = perTensorTimes.reduce(0, +) / Double(perTensorTimes.count)
let perTensorStd = sqrt(perTensorTimes.map { ($0 - perTensorMean) * ($0 - perTensorMean) }.reduce(0, +) / Double(perTensorTimes.count))

print("  Mean: \(String(format: "%.3f", perTensorMean)) ms")
print("  Std:  \(String(format: "%.3f", perTensorStd)) ms")
print("  Bandwidth: \(String(format: "%.1f", Double(totalLayerBytes) / perTensorMean / 1e6)) GB/s")
print()

// ── Summary ─────────────────────────────────────────────────────

print("═══════════════════════════════════════════════════")
print("SUMMARY")
print("═══════════════════════════════════════════════════")
print("Layer size: \(String(format: "%.1f", Double(totalLayerBytes) / 1024 / 1024)) MB (\(tensorOffsets.count) tensors)")
print()
print("Method                    Mean (ms)   Bandwidth")
print("────────────────────────  ─────────   ─────────")
print("Gather → single buffer    \(String(format: "%7.3f", gatherMean))     \(String(format: "%.1f", Double(totalLayerBytes) / gatherMean / 1e6)) GB/s")
print("Contiguous memcpy         \(String(format: "%7.3f", contigMean))     \(String(format: "%.1f", Double(totalLayerBytes) / contigMean / 1e6)) GB/s")
if noCopySuccess && !noCopyTimes.isEmpty {
    let noCopyMean = noCopyTimes.reduce(0, +) / Double(noCopyTimes.count)
    print("Zero-copy (25MB tensor)   \(String(format: "%7.4f", noCopyMean))     ∞ (no copy)")
}
print("Per-tensor buffers        \(String(format: "%7.3f", perTensorMean))     \(String(format: "%.1f", Double(totalLayerBytes) / perTensorMean / 1e6)) GB/s")
print("mx.load (B4 baseline)       6.600     \(String(format: "%.1f", Double(totalLayerBytes) / 6.6 / 1e6)) GB/s")
print()

let gatePass = gatherMean < 3.0
print("GATE: Metal buffer fill \(String(format: "%.3f", gatherMean)) ms \(gatePass ? "<" : ">=") 3.0 ms → \(gatePass ? "PASS ✓" : "FAIL ✗")")
if gatePass {
    print("IAM architecture is VIABLE. Proceed to Step 2.")
} else {
    print("IAM is Pro-only. Pivot to extraction hardening.")
}

// Cleanup
munmap(mmapPtr, fileSize)
close(fd)
