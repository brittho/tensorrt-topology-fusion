#!/usr/bin/env bash

echo "_____Search for the missing library_____"
find /usr /opt/nvidia -name 'libnvdla_compiler.so*' 2>/dev/null

echo
echo "_____Check TensorRT library dependencies_____"
ldd /usr/lib/python3/dist-packages/tensorrt/tensorrt.so 2>/dev/null \
  | grep "not found" || true

echo
echo "_____Check NVDLA-related packages_____"
dpkg -l | grep -Ei 'nvdla|nvidia-l4t.*dla' || true
