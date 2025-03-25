# CUDA Notes

## CUDA Programming Model
- CUDA is an extension to C/C++ for parallel programming on NVIDIA GPUs
- **Thread Hierarchy**:
  - Thread: Single execution unit
  - Block: Group of threads that can cooperate
  - Grid: Group of blocks
- **Memory Hierarchy**:
  - Register: Per-thread
  - Shared Memory: Per-block
  - Global Memory: Accessible by all threads
  - Constant Memory: Read-only for all threads
  - Texture Memory: Specialized for spatial locality

## CUDA Syntax
- `__global__`: Function called from CPU, runs on GPU
- `__device__`: Function called from GPU, runs on GPU
- `__host__`: Function called from CPU, runs on CPU
- `<<<gridDim, blockDim>>>`: Kernel launch configuration

## Best Practices
- Use shared memory for frequently accessed data
- Minimize data transfer between CPU and GPU
- Ensure coalesced memory access
- Avoid warp divergence
- Balance occupancy and resource usage
