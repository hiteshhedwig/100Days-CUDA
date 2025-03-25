add_executable(day001_vector_add vector_add.cu)
target_link_libraries(day001_vector_add cuda_utils)
set_target_properties(day001_vector_add PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
