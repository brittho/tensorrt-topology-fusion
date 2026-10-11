#!/usr/bin/env bash

# 1. Confirm the current directory
echo "_____Current directory_____"

pwd

# 2. JetPack / Jetson Linux version
echo -e "\n_____JetPack / Jetson Linux version_____"

cat /etc/nv_tegra_release
echo -e "\n"
cat /etc/os-release

# 3. CUDA compiler
echo -e "\n_____CUDA compiler_____"

if command -v nvcc >/dev/null 2>&1; then
    nvcc --version
else
    echo "nvcc not found in PATH"
fi

# 4. System Python's TensorRT bindings
echo -e "\n_____System Python's TensorRT bindings_____"

python3 -c 'import tensorrt as trt; print("TensorRT:", trt.__version__)' 2>&1

# 5. TensorRT command-line tool
echo -e "\n_____TensorRT command-line tool_____"

if command -v trtexec >/dev/null 2>&1; then
    command -v trtexec
    trtexec --version
else
    echo "trtexec not found in PATH"
fi

# 6. Nsight Systems
echo -e "\n_____Nsight Systems_____"

if command -v nsys >/dev/null 2>&1; then
    nsys --version
else
    echo "nsys not found in PATH"
fi

# 7. Nsight Compute
echo -e "\n_____Nsight Compute_____"

if command -v ncu >/dev/null 2>&1; then
    ncu --version
else
    echo "ncu not found in PATH"
fi

