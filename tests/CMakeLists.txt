add_executable(test_utils test_utils.cu)
target_link_libraries(test_utils cuda_utils)
set_target_properties(test_utils PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
