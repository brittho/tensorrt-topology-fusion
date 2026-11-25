First, use `make_models.py` to generate two ONNX models (`baseline.onnx` and `exposed.onnx`), each containing a graph expressed in ONNX IR. Both graphs compute Conv → ReLU, and the exposed variant also returns the pre-ReLU convolution tensor as a graph output.

Second, use TensorRT 10.3.0 on the Jetson Orin Nano to compile/build each model into an engine and save each serialized engine/execution plan (`baseline.plan` and `exposed.plan`).

Third, inspect and benchmark each plan with `trtexec --loadEngine=… --dumpLayerInfo`, with per-query H2D/D2H copies enabled. TensorRT reports one fused Conv–ReLU engine layer for the baseline and four engine layers for the exposed variant (a convolution, `PWN(relu)`, and two `NoOp` entries).

Fourth, benchmark each plan again with `trtexec --loadEngine=… --noDataTransfers`, which skips per-query H2D/D2H copies while allowing one-time setup transfers. In these two unprofiled runs (one invocation per plan, with per-query H2D/D2H copies disabled), median GPU compute time was 42.2363 µs for the baseline and 54.1382 µs for the exposed variant, **28.18% higher for the exposed variant** using the baseline as the denominator.

Fifth, trace 50 inference iterations per plan using Nsight Systems (`nsys`) with per-query H2D/D2H copies disabled. The baseline launched one fused Conv–ReLU kernel per inference (50 launches total), while the exposed variant launched two kernels per inference (100 launches total), one convolution kernel and one `generatedNativePointwise` kernel. The additional pointwise kernel is consistent with the separate ReLU layer that TensorRT reports as `PWN(relu)`.
