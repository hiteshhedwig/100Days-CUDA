import torch
import time
import pandas as pd

def benchmark_pytorch_matmul(size, iterations=10):
    # Create matrices on CPU first
    a_cpu = torch.rand(size, size)
    b_cpu = torch.rand(size, size)
    
    # Measure CPU time
    start_time = time.time()
    for _ in range(iterations):
        c_cpu = torch.matmul(a_cpu, b_cpu)
    cpu_time = (time.time() - start_time) * 1000 / iterations  # Convert to ms

    # GPU benchmarking
    if torch.cuda.is_available():
        a_gpu = a_cpu.cuda()
        b_gpu = b_cpu.cuda()
        
        # Warmup
        c_gpu = torch.matmul(a_gpu, b_gpu)
        torch.cuda.synchronize()  # Wait for warmup to finish
        
        # Benchmark
        start_time = time.time()
        for _ in range(iterations):
            c_gpu = torch.matmul(a_gpu, b_gpu)
            torch.cuda.synchronize()
        gpu_time = (time.time() - start_time) * 1000 / iterations  # Convert to ms
    else:
        gpu_time = float('nan')
        print("CUDA not available for PyTorch")
    
    return cpu_time, gpu_time

def run_pytorch_matmul_benchmarks():
    matrix_sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096]
    results = []
    
    for size in matrix_sizes:
        print(f"Benchmarking PyTorch matrix multiplication with size: {size}x{size}")
        
        cpu_time, gpu_time = benchmark_pytorch_matmul(size)
        
        # Calculate speedup if GPU is available
        if gpu_time > 0:
            speedup = cpu_time / gpu_time
        else:
            speedup = float('nan')
            
        results.append({
            'MatrixSize': size,
            'TotalElements': size * size,
            'PyTorch_CPU_Time_ms': cpu_time,
            'PyTorch_GPU_Time_ms': gpu_time,
            'PyTorch_Speedup': speedup,
            'FLOPS': 2 * size * size * size  # Number of floating point operations for matrix multiplication
        })
        
        print(f" PyTorch CPU: {cpu_time:.3f} ms")
        print(f" PyTorch GPU: {gpu_time:.3f} ms")
        if gpu_time > 0:
            print(f" Speedup: {speedup:.2f}x")
            print(f" GPU GFLOPS: {(results[-1]['FLOPS'] / gpu_time / 1e6):.2f}")
        print()

    # Save results
    df = pd.DataFrame(results)
    df.to_csv('pytorch_benchmark_matmul_results.csv', index=False)
    print("PyTorch benchmark complete. Results saved to pytorch_benchmark_matmul_results.csv")
    return df

if __name__ == "__main__":
    run_pytorch_matmul_benchmarks()