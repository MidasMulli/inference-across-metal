// kernel_minimal.swift — Minimal dispatch matching MLX exactly
// MLX for B<=1 only sets buffers 0-7, skips 8-15
import Metal
import Foundation

let device = MTLCreateSystemDefaultDevice()!
let commandQueue = device.makeCommandQueue()!
let lib = try! device.makeLibrary(URL: URL(fileURLWithPath: "/Users/midas/.mlx-env/lib/python3.11/site-packages/mlx/lib/mlx.metallib"))
let refDir = "/Users/midas/Desktop/cowork/inference-across-metal/reference_data"

func loadBin(_ name: String) -> Data {
    try! Data(contentsOf: URL(fileURLWithPath: "\(refDir)/\(name)"))
}
func bf16ToFloat(_ val: UInt16) -> Float { Float(bitPattern: UInt32(val) << 16) }

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

let gridW = (Int(N) + 63) / 64
let gridH = (Int(M) + 63) / 64

let cmd = commandQueue.makeCommandBuffer()!
let enc = cmd.makeComputeCommandEncoder()!
enc.setComputePipelineState(pso)

// Match MLX exactly: only set buffers 0-7
enc.setBuffer(wBuf, offset: 0, index: 0)       // w
enc.setBuffer(sBuf, offset: 0, index: 1)       // scales
enc.setBuffer(bBuf, offset: 0, index: 2)       // biases
enc.setBuffer(xBuf, offset: 0, index: 3)       // x
enc.setBuffer(yBuf, offset: 0, index: 4)       // y
enc.setBytes(&K, length: 4, index: 5)          // K
enc.setBytes(&N, length: 4, index: 6)          // N
enc.setBytes(&M, length: 4, index: 7)          // M
// Buffers 8-15: NOT SET (MLX skips them for B<=1)

enc.dispatchThreadgroups(MTLSize(width: gridW, height: gridH, depth: 1),
                         threadsPerThreadgroup: MTLSize(width: 32, height: 2, depth: 2))
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
print("Bit-exact (minimal, no buf 8-15): \(exact)/\(total) (\(String(format: "%.1f", Float(exact) / Float(total) * 100))%)")

print("First 4 got: ", terminator: "")
for i in 0..<4 { print(String(format: "%.4f ", bf16ToFloat(resultPtr[i])), terminator: "") }
print()
print("First 4 ref: ", terminator: "")
for i in 0..<4 { print(String(format: "%.4f ", bf16ToFloat(refPtr[i])), terminator: "") }
print()
