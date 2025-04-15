import torch
import time
import pandas as pd

# Function to run PyTorch benchmark for matrix transpose
def benchmark_pytorch_transpose_forced(size, iterations=10):
    # Create matrices on CPU first
    a_cpu = torch.rand(size, size)

    # Measure CPU time (forcing contiguous copy)
    start_time = time.time()
    for _ in range(iterations):
        # Perform transpose and force a contiguous copy
        c_cpu = a_cpu.t().contiguous()
    cpu_time = (time.time() - start_time) * 1000 / iterations  # Convert to ms

    # Move to GPU and measure time
    if torch.cuda.is_available():
        a_gpu = a_cpu.cuda()

        # Warmup
        # Perform transpose and force a contiguous copy
        c_gpu = a_gpu.t().contiguous()
        torch.cuda.synchronize() # Wait for warmup to finish

        # Benchmark
        start_time = time.time()
        for _ in range(iterations):
            # Perform transpose and force a contiguous copy
            c_gpu = a_gpu.t().contiguous()
            # Synchronize *after* the operation that forces execution
            torch.cuda.synchronize()
        gpu_time = (time.time() - start_time) * 1000 / iterations  # Convert to ms
    else:
        gpu_time = float('nan')
        print("CUDA not available for PyTorch")

    return cpu_time, gpu_time

# --- Modify your run_pytorch_benchmarks function to call this ---
# --- and save results to a new file ---

# Example of how to adapt the main loop
def run_pytorch_benchmarks_forced():
    matrix_sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096]
    results = []

    for size in matrix_sizes:
        print(f"Benchmarking PyTorch (forced contiguous) with matrix size: {size}x{size}")
        # Call the corrected benchmark function
        cpu_time, gpu_time = benchmark_pytorch_transpose_forced(size)
        # ... rest of the result processing and printing ...
        # Make sure to check for gpu_time > 0 before calculating speedup
        if gpu_time > 0:
             speedup = cpu_time / gpu_time
        else:
             speedup = float('nan')

        results.append({
            'MatrixSize': size,
            'TotalElements': size * size,
            'PyTorch_CPU_Time_ms': cpu_time,
            'PyTorch_GPU_Time_ms': gpu_time,
            'PyTorch_Speedup': speedup
        })
        print(f" PyTorch CPU (forced): {cpu_time:.3f} ms")
        print(f" PyTorch GPU (forced): {gpu_time:.3f} ms")
        if gpu_time > 0:
            print(f" Speedup (forced): {speedup:.2f}x\n")
        else:
            print(" Speedup: NaN (GPU unavailable or zero time)\n")


    # Save results
    df = pd.DataFrame(results)
    df.to_csv(f'build/pytorch_benchmark_matrix_transpose_forced_results.csv', index=False)
    print("PyTorch benchmark complete. Results saved to pytorch_benchmark_matrix_transpose_forced_results.csv")
    return df

if __name__ == "__main__":
    run_pytorch_benchmarks_forced()