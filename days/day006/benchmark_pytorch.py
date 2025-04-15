import torch
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Function to run PyTorch benchmark for matrix-vector multiplication
def benchmark_pytorch(size, iterations=10):
    # Create matrix and vector on CPU first
    a_cpu = torch.rand(size, size)
    b_cpu = torch.rand(size)  # Vector with same dimensions as matrix rows
    
    # Measure CPU time
    start_time = time.time()
    for _ in range(iterations):
        c_cpu = torch.matmul(a_cpu, b_cpu)  # Matrix-vector multiplication
    cpu_time = (time.time() - start_time) * 1000 / iterations  # Convert to ms

    # Move to GPU and measure time
    if torch.cuda.is_available():
        a_gpu = a_cpu.cuda()
        b_gpu = b_cpu.cuda()
        
        # Warmup
        c_gpu = torch.matmul(a_gpu, b_gpu)  # Matrix-vector multiplication
        torch.cuda.synchronize()
        
        # Benchmark
        start_time = time.time()
        for _ in range(iterations):
            c_gpu = torch.matmul(a_gpu, b_gpu)  # Matrix-vector multiplication
            torch.cuda.synchronize()
        gpu_time = (time.time() - start_time) * 1000 / iterations  # Convert to ms
    else:
        gpu_time = float('nan')
        print("CUDA not available for PyTorch")
        
    return cpu_time, gpu_time

# Run benchmarks for different matrix sizes
def run_pytorch_benchmarks():
    matrix_sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096]
    results = []
    
    for size in matrix_sizes:
        print(f"Benchmarking PyTorch with matrix size: {size}x{size}, vector size: {size}")
        cpu_time, gpu_time = benchmark_pytorch(size)
        speedup = cpu_time / gpu_time if gpu_time > 0 else float('nan')
        results.append({
            'MatrixSize': size,
            'TotalElements': size * size,
            'PyTorch_CPU_Time_ms': cpu_time,
            'PyTorch_GPU_Time_ms': gpu_time,
            'PyTorch_Speedup': speedup
        })
        print(f"  PyTorch CPU: {cpu_time:.3f} ms")
        print(f"  PyTorch GPU: {gpu_time:.3f} ms")
        print(f"  Speedup: {speedup:.2f}x\n")
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv('build/pytorch_benchmark_matrix_vector_mul_results.csv', index=False)
    print("PyTorch benchmark complete. Results saved to pytorch_benchmark_matrix_vector_mul_results.csv")
    return df

if __name__ == "__main__":
    run_pytorch_benchmarks()