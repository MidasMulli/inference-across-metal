// weight_loader_aligned.swift — IAM Kill Test v2
// Tests the "align once, zero-copy forever" approach.
//
// Strategy:
//   1. mmap the safetensors file
//   2. For each tensor, compute the page-aligned span that covers it
//   3. Use makeBuffer(bytesNoCopy:) with the aligned span
//   4. Track the intra-page offset to find the tensor within the buffer
//
// The key insight: bytesNoCopy doesn't need the TENSOR to be page-aligned —
// it needs the BUFFER to be page-aligned. We can align the buffer start
// to the previous page boundary and adjust the tensor offset within it.

import Foundation
import Metal

let safetensorsPath = "/Users/midas/.cache/huggingface/hub/models--mlx-community--Qwen3.5-9B-MLX-4bit/blobs/a68b87558c6ef43f74c2bd63ce7e9092ceddc3101f3def0030774bae5f42aadd"

// Layer 0 tensors: (file_offset, size)
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
let pageSize = Int(getpagesize()) // 16384 on Apple Silicon

guard let device = MTLCreateSystemDefaultDevice() else {
    fatalError("No Metal device")
}

print("Metal device: \(device.name)")
print("Page size: \(pageSize)")
print("Layer 0: \(tensorOffsets.count) tensors, \(totalLayerBytes) bytes (\(String(format: "%.1f", Double(totalLayerBytes) / 1024 / 1024)) MB)")
print()

// ── mmap ────────────────────────────────────────────────────────

let fd = open(safetensorsPath, O_RDONLY)
guard fd >= 0 else { fatalError("Cannot open file") }

var st = stat()
fstat(fd, &st)
let fileSize = Int(st.st_size)

let mmapPtr = mmap(nil, fileSize, PROT_READ, MAP_PRIVATE, fd, 0)
guard mmapPtr != MAP_FAILED else { fatalError("mmap failed") }
let basePtr = mmapPtr!

// ── Test A: Page-aligned zero-copy per tensor ───────────────────
// For each tensor, find the enclosing page-aligned range and
// create a zero-copy buffer. The tensor data starts at
// (buffer_start + intra_page_offset).

print("TEST A: Zero-copy per tensor (page-aligned spans)")
print("─────────────────────────────────────────────────")

struct AlignedTensor {
    let buffer: MTLBuffer
    let intraOffset: Int  // offset within buffer to tensor start
    let size: Int
}

var zeroCopyTimes: [Double] = []
var zeroCopyFailed = 0

for trial in 0..<(warmupTrials + numTrials) {
    let start = CFAbsoluteTimeGetCurrent()

    var tensors: [AlignedTensor] = []
    tensors.reserveCapacity(tensorOffsets.count)

    for (fileOffset, size) in tensorOffsets {
        let alignedStart = (fileOffset / pageSize) * pageSize
        let intraOffset = fileOffset - alignedStart
        let alignedEnd = ((fileOffset + size + pageSize - 1) / pageSize) * pageSize
        let alignedSize = alignedEnd - alignedStart

        let ptr = UnsafeMutableRawPointer(mutating: basePtr + alignedStart)
        if let buffer = device.makeBuffer(bytesNoCopy: ptr, length: alignedSize, options: .storageModeShared, deallocator: nil) {
            tensors.append(AlignedTensor(buffer: buffer, intraOffset: intraOffset, size: size))
        } else {
            zeroCopyFailed += 1
        }
    }

    let elapsed = CFAbsoluteTimeGetCurrent() - start

    if trial >= warmupTrials {
        zeroCopyTimes.append(elapsed * 1000)
    }
}

if !zeroCopyTimes.isEmpty {
    let mean = zeroCopyTimes.reduce(0, +) / Double(zeroCopyTimes.count)
    let std = sqrt(zeroCopyTimes.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Double(zeroCopyTimes.count))
    let minT = zeroCopyTimes.min()!
    let maxT = zeroCopyTimes.max()!
    print("  Mean: \(String(format: "%.4f", mean)) ms")
    print("  Std:  \(String(format: "%.4f", std)) ms")
    print("  Min:  \(String(format: "%.4f", minT)) ms")
    print("  Max:  \(String(format: "%.4f", maxT)) ms")
    if zeroCopyFailed > 0 {
        print("  Failed: \(zeroCopyFailed / (warmupTrials + numTrials)) per trial")
    } else {
        print("  All \(tensorOffsets.count) tensors: zero-copy SUCCESS")
    }
} else {
    print("  FAILED: No successful zero-copy")
}
print()

// ── Test B: Coalesced zero-copy (group nearby tensors) ──────────
// Instead of 30 buffers, group tensors that fall within the same
// page-aligned region into a single buffer.

print("TEST B: Coalesced zero-copy (merge adjacent tensors)")
print("─────────────────────────────────────────────────")

// Sort tensors by offset, then greedily merge overlapping page spans
var sorted_tensors = tensorOffsets.sorted { $0.0 < $1.0 }
var groups: [(Int, Int)] = [] // (alignedStart, alignedEnd)

for (offset, size) in sorted_tensors {
    let aStart = (offset / pageSize) * pageSize
    let aEnd = ((offset + size + pageSize - 1) / pageSize) * pageSize

    if let last = groups.last, aStart <= last.1 + pageSize {
        // Merge: extend the last group
        groups[groups.count - 1] = (last.0, max(last.1, aEnd))
    } else {
        groups.append((aStart, aEnd))
    }
}

print("  \(tensorOffsets.count) tensors coalesced into \(groups.count) page-aligned groups")
for (i, (start, end)) in groups.enumerated() {
    print("    Group \(i): \(String(format: "%.1f", Double(end - start) / 1024 / 1024)) MB")
}

var coalescedTimes: [Double] = []
var coalescedFailed = 0

for trial in 0..<(warmupTrials + numTrials) {
    let start = CFAbsoluteTimeGetCurrent()

    var buffers: [MTLBuffer] = []
    buffers.reserveCapacity(groups.count)

    for (aStart, aEnd) in groups {
        let ptr = UnsafeMutableRawPointer(mutating: basePtr + aStart)
        let size = aEnd - aStart
        if let buffer = device.makeBuffer(bytesNoCopy: ptr, length: size, options: .storageModeShared, deallocator: nil) {
            buffers.append(buffer)
        } else {
            coalescedFailed += 1
        }
    }

    let elapsed = CFAbsoluteTimeGetCurrent() - start

    if trial >= warmupTrials {
        coalescedTimes.append(elapsed * 1000)
    }
}

if !coalescedTimes.isEmpty {
    let mean = coalescedTimes.reduce(0, +) / Double(coalescedTimes.count)
    let std = sqrt(coalescedTimes.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Double(coalescedTimes.count))
    print("  Mean: \(String(format: "%.4f", mean)) ms")
    print("  Std:  \(String(format: "%.4f", std)) ms")
    if coalescedFailed > 0 {
        print("  Failed: \(coalescedFailed / (warmupTrials + numTrials)) per trial")
    }
} else {
    print("  FAILED: No successful coalesced zero-copy")
}
print()

// ── Test C: Full mmap zero-copy (one buffer for entire file) ────
// The ultimate: one Metal buffer wrapping the entire mmap'd file.

print("TEST C: Single buffer for entire 5GB file")
print("─────────────────────────────────────────────────")

let fileAlignedSize = ((fileSize + pageSize - 1) / pageSize) * pageSize

var fullFileTimes: [Double] = []
var fullFileSuccess = false

for trial in 0..<(warmupTrials + numTrials) {
    let start = CFAbsoluteTimeGetCurrent()

    let ptr = UnsafeMutableRawPointer(mutating: basePtr)
    let buffer = device.makeBuffer(bytesNoCopy: ptr, length: fileAlignedSize, options: .storageModeShared, deallocator: nil)

    let elapsed = CFAbsoluteTimeGetCurrent() - start

    if buffer != nil {
        fullFileSuccess = true
        if trial >= warmupTrials {
            fullFileTimes.append(elapsed * 1000)
        }
    }
}

if fullFileSuccess && !fullFileTimes.isEmpty {
    let mean = fullFileTimes.reduce(0, +) / Double(fullFileTimes.count)
    let std = sqrt(fullFileTimes.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Double(fullFileTimes.count))
    print("  Mean: \(String(format: "%.4f", mean)) ms")
    print("  Std:  \(String(format: "%.4f", std)) ms")
    print("  (Entire 5GB file as one Metal buffer — all tensors at known offsets)")
} else {
    print("  FAILED: Cannot wrap entire file")
}
print()

// ── Summary ─────────────────────────────────────────────────────

print("═══════════════════════════════════════════════════")
print("SUMMARY — Zero-Copy Approaches")
print("═══════════════════════════════════════════════════")
print()

let methods: [(String, [Double])] = [
    ("Per-tensor zero-copy (30 buffers)", zeroCopyTimes),
    ("Coalesced zero-copy (\(groups.count) buffers)", coalescedTimes),
    ("Full-file zero-copy (1 buffer)", fullFileTimes),
]

print("Method                              Mean (ms)")
print("──────────────────────────────────  ─────────")
for (name, times) in methods {
    if !times.isEmpty {
        let mean = times.reduce(0, +) / Double(times.count)
        print("\(name.padding(toLength: 36, withPad: " ", startingAt: 0))\(String(format: "%7.4f", mean))")
    } else {
        print("\(name.padding(toLength: 36, withPad: " ", startingAt: 0))  FAILED")
    }
}
print("v1 gather (from weight_loader)                    27.454")
print("mx.load (B4 Python baseline)                       6.600")
print()

// Gate check with best zero-copy result
var bestMean = 999.0
let allTimeSets: [[Double]] = [zeroCopyTimes, coalescedTimes, fullFileTimes]
for times in allTimeSets {
    if !times.isEmpty {
        let m: Double = times.reduce(0, +) / Double(times.count)
        if m < bestMean { bestMean = m }
    }
}
if bestMean < 999.0 {
    let gatePass = bestMean < 3.0
    let gateStr = gatePass ? "PASS" : "FAIL"
    print("GATE: Best zero-copy \(String(format: "%.4f", bestMean)) ms \(gatePass ? "<" : ">=") 3.0 ms → \(gateStr)")
    print("Speedup vs mx.load: \(String(format: "%.0f", 6.6 / bestMean))x")
    print("Speedup vs v1 gather: \(String(format: "%.0f", 27.454 / bestMean))x")
}

munmap(mmapPtr, fileSize)
close(fd)
