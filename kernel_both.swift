// kernel_both.swift — Compare NAX vs Standard kernel output
import Metal
import Foundation

let device = MTLCreateSystemDefaultDevice()!
let commandQueue = device.makeCommandQueue()!
let lib = try! device.makeLibrary(URL: URL(fileURLWithPath: "/Users/midas/.mlx-env/lib/python3.11/site-packages/mlx/lib/mlx.metallib"))
let refDir = "/Users/midas/Desktop/cowork/inference-across-metal/reference_data"

func loadBin(_ name: String) -> Data {
    try! Data(contentsOf: URL(fileURLWithPath: "\(refDir)/\(name)"))
}

func makeConstant(_ value: Int32) -> MTLBuffer {
    var v = value
    return device.makeBuffer(bytes: &v, length: 4, options: .storageModeShared)!
}

func makeConstantArray<T>(_ values: [T]) -> MTLBuffer {
    values.withUnsafeBufferPointer { ptr in
        device.makeBuffer(bytes: ptr.baseAddress!, length: ptr.count * MemoryLayout<T>.stride, options: .storageModeShared)!
    }
}

func bf16ToFloat(_ val: UInt16) -> Float {
    Float(bitPattern: UInt32(val) << 16)
}

let inputBF16 = loadBin("qlinear_input_bf16.bin")
let weightData = loadBin("qlinear_weight_raw.bin")
let scalesBF16 = loadBin("qlinear_scales_bf16.bin")
let biasesBF16 = loadBin("qlinear_biases_bf16.bin")
let outputRefBF16 = loadBin("qlinear_output_bf16.bin")

let M: Int32 = 1, K: Int32 = 4096, N: Int32 = 12288

// Create shared data buffers
let wBuf = weightData.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: weightData.count, options: .storageModeShared)! }
let sBuf = scalesBF16.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: scalesBF16.count, options: .storageModeShared)! }
let bBuf = biasesBF16.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: biasesBF16.count, options: .storageModeShared)! }
let xBuf = inputBF16.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: inputBF16.count, options: .storageModeShared)! }

let kBuf = makeConstant(K)
let nBuf = makeConstant(N)
let mBuf = makeConstant(M)
let xBatchNdims = makeConstant(Int32(0))
let xShape = makeConstantArray([Int32(0)])  // empty
let xStrides = makeConstantArray([Int64(0)])
let wBatchNdims = makeConstant(Int32(0))
let wShape = makeConstantArray([Int32(0)])
let wStrides = makeConstantArray([Int64(0)])
let sStrides = makeConstantArray([Int64(0)])
let bStrides = makeConstantArray([Int64(0)])

let outputSize = Int(M) * Int(N) * 2

struct KernelConfig {
    let name: String
    let bm: Int
    let bn: Int
}

let configs = [
    KernelConfig(name: "affine_qmm_t_nax_bfloat16_t_gs_64_b_4_bm64_bn64_bk64_wm2_wn2_alN_true_batch_0", bm: 64, bn: 64),
    KernelConfig(name: "affine_qmm_t_nax_bfloat16_t_gs_64_b_4_bm64_bn64_bk64_wm2_wn2_alN_false_batch_0", bm: 64, bn: 64),
    KernelConfig(name: "affine_qmm_t_bfloat16_t_gs_64_b_4_alN_true_batch_0", bm: 32, bn: 32),
    KernelConfig(name: "affine_qmm_t_bfloat16_t_gs_64_b_4_alN_false_batch_0", bm: 32, bn: 32),
]

var outputs: [(String, [UInt16])] = []

for config in configs {
    let fn = lib.makeFunction(name: config.name)!
    let pso = try! device.makeComputePipelineState(function: fn)
    let yBuf = device.makeBuffer(length: outputSize, options: .storageModeShared)!

    let gridW = (Int(N) + config.bn - 1) / config.bn
    let gridH = (Int(M) + config.bm - 1) / config.bm

    let cmd = commandQueue.makeCommandBuffer()!
    let enc = cmd.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pso)
    enc.setBuffer(wBuf, offset: 0, index: 0)
    enc.setBuffer(sBuf, offset: 0, index: 1)
    enc.setBuffer(bBuf, offset: 0, index: 2)
    enc.setBuffer(xBuf, offset: 0, index: 3)
    enc.setBuffer(yBuf, offset: 0, index: 4)
    enc.setBuffer(kBuf, offset: 0, index: 5)
    enc.setBuffer(nBuf, offset: 0, index: 6)
    enc.setBuffer(mBuf, offset: 0, index: 7)
    enc.setBuffer(xBatchNdims, offset: 0, index: 8)
    enc.setBuffer(xShape, offset: 0, index: 9)
    enc.setBuffer(xStrides, offset: 0, index: 10)
    enc.setBuffer(wBatchNdims, offset: 0, index: 11)
    enc.setBuffer(wShape, offset: 0, index: 12)
    enc.setBuffer(wStrides, offset: 0, index: 13)
    enc.setBuffer(sStrides, offset: 0, index: 14)
    enc.setBuffer(bStrides, offset: 0, index: 15)
    enc.dispatchThreadgroups(MTLSize(width: gridW, height: gridH, depth: 1), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()

    let ptr = yBuf.contents().bindMemory(to: UInt16.self, capacity: Int(M) * Int(N))
    var result: [UInt16] = Array(repeating: 0, count: Int(M) * Int(N))
    for i in 0..<result.count { result[i] = ptr[i] }

    outputs.append((config.name, result))

    // Compare vs reference
    let refPtr = outputRefBF16.withUnsafeBytes { $0.bindMemory(to: UInt16.self) }
    var exact = 0
    for i in 0..<result.count {
        if result[i] == refPtr[i] { exact += 1 }
    }
    let shortName = config.name.replacingOccurrences(of: "affine_qmm_t_", with: "").replacingOccurrences(of: "_bfloat16_t_gs_64_b_4_", with: "_")
    print("\(shortName): \(exact)/\(result.count) (\(String(format: "%.1f", Float(exact) / Float(result.count) * 100))%)")
    print("  First 4: ", terminator: "")
    for i in 0..<4 { print(String(format: "%.4f ", bf16ToFloat(result[i])), terminator: "") }
    print()
}

// Compare all pairs
print("\n--- Cross-comparison ---")
for i in 0..<outputs.count {
    for j in (i+1)..<outputs.count {
        var exact = 0
        for k in 0..<outputs[i].1.count {
            if outputs[i].1[k] == outputs[j].1[k] { exact += 1 }
        }
        let pct = Float(exact) / Float(outputs[i].1.count) * 100
        let n1 = String(outputs[i].0.suffix(30))
        let n2 = String(outputs[j].0.suffix(30))
        print("  \(n1) vs \(n2): \(String(format: "%.1f", pct))%")
    }
}
