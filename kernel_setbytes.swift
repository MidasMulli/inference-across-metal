// kernel_setbytes.swift — Test using setBytes for scalar constants
import Metal
import Foundation

let device = MTLCreateSystemDefaultDevice()!
let commandQueue = device.makeCommandQueue()!
let lib = try! device.makeLibrary(URL: URL(fileURLWithPath: "/Users/midas/.mlx-env/lib/python3.11/site-packages/mlx/lib/mlx.metallib"))
let refDir = "/Users/midas/Desktop/cowork/inference-across-metal/reference_data"

func loadBin(_ name: String) -> Data {
    try! Data(contentsOf: URL(fileURLWithPath: "\(refDir)/\(name)"))
}

func bf16ToFloat(_ val: UInt16) -> Float {
    Float(bitPattern: UInt32(val) << 16)
}

let inputBF16 = loadBin("qlinear_input_bf16.bin")
let weightData = loadBin("qlinear_weight_raw.bin")
let scalesBF16 = loadBin("qlinear_scales_bf16.bin")
let biasesBF16 = loadBin("qlinear_biases_bf16.bin")
let outputRefBF16 = loadBin("qlinear_output_bf16.bin")

var M: Int32 = 1, K: Int32 = 4096, N: Int32 = 12288

let kernelName = "affine_qmm_t_nax_bfloat16_t_gs_64_b_4_bm64_bn64_bk64_wm2_wn2_alN_true_batch_0"
let fn = lib.makeFunction(name: kernelName)!
let pso = try! device.makeComputePipelineState(function: fn)

let wBuf = weightData.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: weightData.count, options: .storageModeShared)! }
let sBuf = scalesBF16.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: scalesBF16.count, options: .storageModeShared)! }
let bBuf = biasesBF16.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: biasesBF16.count, options: .storageModeShared)! }
let xBuf = inputBF16.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: inputBF16.count, options: .storageModeShared)! }

let outputSize = Int(M) * Int(N) * 2
let yBuf = device.makeBuffer(length: outputSize, options: .storageModeShared)!

// Batch params (batch_0 ignores them but they must be valid buffers)
var batchNdims: Int32 = 0
var zeroShape: Int32 = 0
var zeroStride: Int64 = 0

let gridW = (Int(N) + 63) / 64  // BN=64
let gridH = (Int(M) + 63) / 64  // BM=64

print("Grid: \(gridW) x \(gridH)")

let cmd = commandQueue.makeCommandBuffer()!
let enc = cmd.makeComputeCommandEncoder()!
enc.setComputePipelineState(pso)

// Data buffers
enc.setBuffer(wBuf, offset: 0, index: 0)
enc.setBuffer(sBuf, offset: 0, index: 1)
enc.setBuffer(bBuf, offset: 0, index: 2)
enc.setBuffer(xBuf, offset: 0, index: 3)
enc.setBuffer(yBuf, offset: 0, index: 4)

// Scalar constants via setBytes
enc.setBytes(&K, length: 4, index: 5)
enc.setBytes(&N, length: 4, index: 6)
enc.setBytes(&M, length: 4, index: 7)
enc.setBytes(&batchNdims, length: 4, index: 8)
enc.setBytes(&zeroShape, length: 4, index: 9)
enc.setBytes(&zeroStride, length: 8, index: 10)
enc.setBytes(&batchNdims, length: 4, index: 11)
enc.setBytes(&zeroShape, length: 4, index: 12)
enc.setBytes(&zeroStride, length: 8, index: 13)
enc.setBytes(&zeroStride, length: 8, index: 14)
enc.setBytes(&zeroStride, length: 8, index: 15)

// MLX uses (32, wn=2, wm=2) threadgroup layout, NOT (128, 1, 1)
enc.dispatchThreadgroups(MTLSize(width: gridW, height: gridH, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 2, depth: 2))
enc.endEncoding()
cmd.commit()
cmd.waitUntilCompleted()

if let error = cmd.error {
    print("GPU ERROR: \(error)")
    exit(1)
}

let resultPtr = yBuf.contents().bindMemory(to: UInt16.self, capacity: Int(M) * Int(N))
let refPtr = outputRefBF16.withUnsafeBytes { $0.bindMemory(to: UInt16.self) }

var exact = 0
let total = Int(M) * Int(N)
for i in 0..<total {
    if resultPtr[i] == refPtr[i] { exact += 1 }
}
print("Bit-exact (setBytes): \(exact)/\(total) (\(String(format: "%.1f", Float(exact) / Float(total) * 100))%)")

print("First 4 got:  ", terminator: "")
for i in 0..<4 { print(String(format: "%.4f ", bf16ToFloat(resultPtr[i])), terminator: "") }
print()
print("First 4 ref:  ", terminator: "")
for i in 0..<4 { print(String(format: "%.4f ", bf16ToFloat(refPtr[i])), terminator: "") }
print()

// Save for comparison
let dumpData = Data(bytes: yBuf.contents(), count: outputSize)
try! dumpData.write(to: URL(fileURLWithPath: "\(refDir)/swift_setbytes_output_bf16.bin"))
