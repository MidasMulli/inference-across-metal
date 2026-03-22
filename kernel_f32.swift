// kernel_f32.swift — Dispatch float32 kernel (standard path, not NAX)
import Metal
import Foundation

let device = MTLCreateSystemDefaultDevice()!
let commandQueue = device.makeCommandQueue()!
let lib = try! device.makeLibrary(URL: URL(fileURLWithPath: "/Users/midas/.mlx-env/lib/python3.11/site-packages/mlx/lib/mlx.metallib"))
let refDir = "/Users/midas/Desktop/cowork/inference-across-metal/reference_data"

func loadBin(_ name: String) -> Data {
    try! Data(contentsOf: URL(fileURLWithPath: "\(refDir)/\(name)"))
}

let inputF32 = loadBin("qlinear_input_f32.bin")
let weightData = loadBin("qlinear_weight_raw.bin")
let scalesF32 = loadBin("qlinear_scales_f32.bin")
let biasesF32 = loadBin("qlinear_biases_f32.bin")
let outputRefF32 = loadBin("qlinear_output_f32_input.bin")

var M: Int32 = 1, K: Int32 = 4096, N: Int32 = 12288

// For f32 + not enable_tf32: MLX uses standard qmm_t (not NAX), BM=32, BN=32
let kernelName = "affine_qmm_t_float32_t_gs_64_b_4_alN_true_batch_0"
guard lib.functionNames.contains(kernelName) else { fatalError("Kernel not found: \(kernelName)") }

let fn = lib.makeFunction(name: kernelName)!
let pso = try! device.makeComputePipelineState(function: fn)

let wBuf = weightData.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: weightData.count, options: .storageModeShared)! }
let sBuf = scalesF32.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: scalesF32.count, options: .storageModeShared)! }
let bBuf = biasesF32.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: biasesF32.count, options: .storageModeShared)! }
let xBuf = inputF32.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: inputF32.count, options: .storageModeShared)! }

let outputSize = Int(M) * Int(N) * 4  // float32
let yBuf = device.makeBuffer(length: outputSize, options: .storageModeShared)!

// Standard kernel: BM=32, BN=32, threadgroup (32, 2, 2)
let gridW = (Int(N) + 31) / 32
let gridH = (Int(M) + 31) / 32

let cmd = commandQueue.makeCommandBuffer()!
let enc = cmd.makeComputeCommandEncoder()!
enc.setComputePipelineState(pso)

enc.setBuffer(wBuf, offset: 0, index: 0)
enc.setBuffer(sBuf, offset: 0, index: 1)
enc.setBuffer(bBuf, offset: 0, index: 2)
enc.setBuffer(xBuf, offset: 0, index: 3)
enc.setBuffer(yBuf, offset: 0, index: 4)
enc.setBytes(&K, length: 4, index: 5)
enc.setBytes(&N, length: 4, index: 6)
enc.setBytes(&M, length: 4, index: 7)
// Skip 8-15 (unbatched)

enc.dispatchThreadgroups(MTLSize(width: gridW, height: gridH, depth: 1),
                         threadsPerThreadgroup: MTLSize(width: 32, height: 2, depth: 2))
enc.endEncoding()
cmd.commit()
cmd.waitUntilCompleted()

if let error = cmd.error {
    print("GPU ERROR: \(error)")
    exit(1)
}

let resultPtr = yBuf.contents().bindMemory(to: Float.self, capacity: Int(M) * Int(N))
let refPtr = outputRefF32.withUnsafeBytes { $0.bindMemory(to: Float.self) }

var exact = 0, total = Int(M) * Int(N)
var maxErr: Float = 0
for i in 0..<total {
    if resultPtr[i] == refPtr[i] { exact += 1 }
    maxErr = max(maxErr, abs(resultPtr[i] - refPtr[i]))
}
print("F32 Bit-exact: \(exact)/\(total) (\(String(format: "%.1f", Float(exact) / Float(total) * 100))%)")
print("Max error: \(maxErr)")

print("First 4 got: ", terminator: "")
for i in 0..<4 { print(String(format: "%.6f ", resultPtr[i]), terminator: "") }
print()
print("First 4 ref: ", terminator: "")
for i in 0..<4 { print(String(format: "%.6f ", refPtr[i]), terminator: "") }
print()
