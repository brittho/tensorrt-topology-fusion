# tensorrt-topology-fusion
Investigating how neural-network topology affects TensorRT operator fusion, GPU execution behavior, and inference performance on NVIDIA Jetson.

# Result:
On a Jetson Orin Nano with TensorRT 10.3, exposing the pre-ReLU convolution result as a second graph output changed execution from one fused Conv–ReLU kernel to a convolution kernel followed by a standalone pointwise ReLU kernel. With per-query host–device copies disabled, median GPU compute time increased from 42.24 µs to 54.14 µs (+28.2%) in the measured pair. The Nsight Systems trace confirmed the extra kernel launch and recorded no device-to-device memcpy operations in the exposed plan.
