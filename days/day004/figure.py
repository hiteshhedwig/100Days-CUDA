import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def combine_results():
    # Load CUDA benchmark results
    cuda_df = pd.read_csv('build/benchmark_results_scalar_mul.csv')
    cuda_df = cuda_df.rename(columns={
        'CPU_Time_ms': 'CUDA_CPU_Time_ms',
        'GPU_Time_ms': 'CUDA_GPU_Time_ms',
        'Speedup': 'CUDA_Speedup'
    })
    
    # Load PyTorch benchmark results
    pytorch_df = pd.read_csv('build/pytorch_benchmark_scalar_mul_results.csv')
    
    # Merge the dataframes
    combined = pd.merge(cuda_df, pytorch_df, on=['MatrixSize', 'TotalElements'])
    return combined

def create_execution_time_plot(df):
    plt.figure(figsize=(12, 8))
    
    # Log scale for better visibility
    plt.yscale('log')
    
    # Plot lines
    plt.plot(df['MatrixSize'], df['CUDA_CPU_Time_ms'], 'o-', label='CUDA-Custom CPU', linewidth=2)
    plt.plot(df['MatrixSize'], df['CUDA_GPU_Time_ms'], 's-', label='CUDA-Custom GPU', linewidth=2)
    plt.plot(df['MatrixSize'], df['PyTorch_CPU_Time_ms'], '^-', label='PyTorch CPU', linewidth=2)
    plt.plot(df['MatrixSize'], df['PyTorch_GPU_Time_ms'], 'D-', label='PyTorch GPU', linewidth=2)
    
    plt.xlabel('Matrix Size (NxN)', fontsize=14)
    plt.ylabel('Execution Time (ms, log scale)', fontsize=14)
    plt.title('Matrix Scalar Multiplication Performance Comparison', fontsize=16)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(fontsize=12)
    
    # Set x-axis to show each matrix size
    plt.xticks(df['MatrixSize'], rotation=45)
    
    plt.tight_layout()
    plt.savefig('execution_time_comparison.png', dpi=300)
    print("Created execution time plot: execution_time_comparison.png")

def create_speedup_plot(df):
    plt.figure(figsize=(12, 8))
    
    # Calculate all speedups relative to CPU performance
    baseline = df['CUDA_CPU_Time_ms']
    
    relative_speedups = pd.DataFrame({
        'MatrixSize': df['MatrixSize'],
        'CUDA-Custom CPU': np.ones(len(df)),  # Baseline (1x)
        'CUDA-Custom GPU': baseline / df['CUDA_GPU_Time_ms'],
        'PyTorch CPU': baseline / df['PyTorch_CPU_Time_ms'],
        'PyTorch GPU': baseline / df['PyTorch_GPU_Time_ms']
    })
    
    # Melt the dataframe for easier plotting
    melted = pd.melt(relative_speedups, id_vars=['MatrixSize'], 
                     var_name='Implementation', value_name='Speedup')
    
    # Create the bar chart
    plt.figure(figsize=(14, 8))
    bar_plot = sns.barplot(x='MatrixSize', y='Speedup', hue='Implementation', data=melted)
    
    plt.xlabel('Matrix Size (NxN)', fontsize=14)
    plt.ylabel('Speedup (relative to CUDA CPU)', fontsize=14)
    plt.title('Relative Performance Comparison', fontsize=16)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Implementation', fontsize=12)
    
    # Add value labels on top of bars
    for p in bar_plot.patches:
        bar_plot.annotate(f'{p.get_height():.1f}x', 
                         (p.get_x() + p.get_width() / 2., p.get_height()), 
                         ha = 'center', va = 'center', 
                         xytext = (0, 10), 
                         textcoords = 'offset points')
    
    plt.tight_layout()
    plt.savefig('speedup_comparison.png', dpi=300)
    print("Created speedup plot: speedup_comparison.png")

def main():
    # Combine the benchmark results
    combined_df = combine_results()
    
    # Create the plots
    create_execution_time_plot(combined_df)
    create_speedup_plot(combined_df)
    
    # Also save the combined data
    # combined_df.to_csv('combined_benchmark_results.csv', index=False)
    # print("Combined results saved to: combined_benchmark_results.csv")

if __name__ == "__main__":
    main()