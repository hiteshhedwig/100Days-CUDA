add_library(cuda_utils
    src/cuda_utils.cu
    src/timer.cu
    src/data_utils.cu
)

target_include_directories(cuda_utils PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set_target_properties(cuda_utils PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
