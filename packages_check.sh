#!/usr/bin/env bash

echo "_____List installed NVIDIA packages_____"
dpkg -l | grep -E 'nvidia-jetpack|nvidia-l4t-cuda|cuda-toolkit|libnvinfer|tensorrt|nvonnxparser' || true

echo
echo "_____Check package availability_____"
apt-cache policy nvidia-jetpack tensorrt libnvinfer-bin python3-libnvinfer cuda-toolkit-12-6

echo
echo "_____Check common executable locations_____"

found=0
for f in \
  /usr/local/cuda/bin/nvcc \
  /usr/local/cuda-12.6/bin/nvcc \
  /usr/src/tensorrt/bin/trtexec \
  /usr/bin/trtexec
do
  if [ -x "$f" ]; then
    echo "FOUND: $f"
    found=1
  fi
done

if [ "$found" -eq 0 ]; then
  echo "No executable found in the checked locations."
fi
