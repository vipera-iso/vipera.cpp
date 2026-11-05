# kernel-gate: per-quant bytes/second of one mul_mat, so a kernel change can
# be judged by the gate math in tools/vipera/gate.h instead of by feel

set(TARGET kernel-gate)

add_executable(${TARGET} kernel_gate.cpp)
target_link_libraries(${TARGET} PRIVATE llama-common ggml)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# bandwidth-probe: the ceiling that kernel-gate's GB/s is compared against.
# Without it a "65 GB/s" kernel has no way of being called slow or fast.
add_executable(bandwidth-probe bandwidth_probe.cpp)
find_package(Threads REQUIRED)
target_link_libraries(bandwidth-probe PRIVATE Threads::Threads)
target_compile_features(bandwidth-probe PRIVATE cxx_std_17)

# The CUDA half only exists in a CUDA build; the CPU probe stays standalone so
# a CPU-only tree still gets a number to compare the CPU kernels against.
if (GGML_CUDA)
    find_package(CUDAToolkit QUIET)
    if (CUDAToolkit_FOUND)
        enable_language(CUDA)
        add_executable(bandwidth-probe-cuda bandwidth_probe.cu)
        target_link_libraries(bandwidth-probe-cuda PRIVATE CUDA::cudart)
        if (NOT CMAKE_CUDA_ARCHITECTURES)
            set_target_properties(bandwidth-probe-cuda PROPERTIES CUDA_ARCHITECTURES "75")
        endif()
    else()
        message(STATUS "bandwidth-probe-cuda skipped: CUDAToolkit not found")
    endif()
endif()

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
