add_executable(day002_vector_mul vector_mul.cu)
target_link_libraries(day002_vector_mul cuda_utils)
set_target_properties(day002_vector_mul PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
