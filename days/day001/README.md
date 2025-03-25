# Day 1: Vector Addition

## Today's Goals
- Set up CUDA development environment
- Understand basic CUDA kernel launch syntax
- Implement and test vector addition kernel
- Learn about error handling in CUDA
- Measure kernel execution time

## Notes
- The `blockDim.x * blockIdx.x + threadIdx.x` formula gives the global thread ID
- Error checking is important for CUDA operations
- Memory transfers between host and device can be a performance bottleneck

## Performance Results
- Execution time: [Add your measurements here]
- Throughput: [Add your calculations here]

## Questions to Explore
- How does changing the block size affect performance?
- What happens if we use too many threads per block?
- How does the performance compare with CPU implementation?
