# tensorrt-topology-fusion
Investigating how neural-network topology (graph outputs) affects TensorRT operator fusion, GPU execution behavior, and inference performance on NVIDIA Jetson.

On a Jetson Orin Nano with TensorRT 10.3.0, exposing the pre-ReLU convolution result as a second ONNX graph output changed execution from one fused Conv–ReLU kernel to a convolution kernel followed by a standalone pointwise ReLU kernel. With per-query host–device copies disabled, median GPU compute time rose from **42.24 µs to 54.14 µs**, an increase of **11.90 µs (+28.18%)** in the measured pair. The Nsight Systems trace confirmed the extra kernel launch and recorded no device-to-device memcpy operations in the exposed plan.

## Experiment design

**RQ:** How does exposing intermediate observable (i.e. pre-ReLU convolution tensor as an additional ONNX graph output) affect Conv-ReLU fusion, CUDA kernel launches, and measured inference performance on TensorRT 10.3.0 running on a Jetson Orin Nano with and without per-query H2D/D2H copies?

<img width="75%" alt="image" src="https://github.com/user-attachments/assets/a5779461-13ab-40eb-811a-26818275bd2e" />

Both ONNX graphs express a 3 × 3 convolution with bias followed by ReLU. Both accept a batch-one FP32 tensor of shape `1 × 16 × 64 × 64`. Each output is `1 × 32 × 64 × 64` in linear FP32, or 512 KiB.

| Variant | Graph outputs | Observable intermediate |
| --- | --- | --- |
| Baseline | `relu_out` | The convolution result is internal. |
| Exposed output | `conv_out`, `relu_out` | The caller can read the convolution result before ReLU. |

**Environment.** NVIDIA Jetson Orin Nano (Ampere, compute capability 8.7), TensorRT 10.3.0 (`trtexec` v100300), and Nsight Systems 2024.5.4. Both `trtexec` benchmark conditions used one inference stream, CUDA Graph disabled, a 200 ms warm-up, and a roughly 3-second timing window. The logs show that `trtexec` generated input values during setup. The exact contents were not recorded, and numerical agreement of `relu_out` was not measured.

## TensorRT engine inspection

Source: `results/baseline_inspect.log` and `results/exposed_inspect.log`, `Layer Information` from `trtexec --dumpLayerInfo`.

| Plan | Inspector entries in order | Convolution activation |
| --- | --- | --- |
| Baseline | `conv + relu` (`CaskConvolution`) | `RELU` |
| Exposed output | `conv` → input reformatting `CopyNode` (`NoOp`) → `PWN(relu)` (`PointWiseV2`) → output reformatting `CopyNode` (`NoOp`) | `NONE` |

Both plans report the same convolution tactic ID, `0x94119b4c514b211a`. The tactic name contains `relu` in both plans. The inspection activation field identifies whether the convolution applies ReLU.

## Unprofiled inference benchmarks

Each column represents one independent `trtexec` invocation. Thousands of queries within an invocation describe its latency distribution. Run-to-run variability requires repeated invocations. All deltas use the original logged values before rounding. For comparisons between engines, Δ = exposed − baseline and Δ% = Δ / baseline × 100. All timings are in microseconds per query. Throughput is measured in queries per second (qps).

### Per-query transfers enabled

Source: `results/baseline_inspect.log` and `results/exposed_inspect.log`, `Performance summary`. The timed portions contained 48,387 baseline queries and 37,145 exposed-output queries. `trtexec` reports `Latency` as the sum of host-to-device (H2D), GPU compute, and device-to-host (D2H) time for each query.

| Metric | Baseline | Exposed output | Δ | Δ% |
| --- | ---: | ---: | ---: | ---: |
| Reported latency, mean (µs) | 92.50 | 127.19 | +34.69 | +37.50% |
| Reported latency, median (µs) | 92.29 | 127.20 | +34.91 | +37.83% |
| Reported latency, p99 (µs) | 98.71 | 141.63 | +42.92 | +43.48% |
| H2D latency, mean (µs) | 21.58 | 23.10 | +1.53 | +7.08% |
| GPU compute time, mean (µs) | 43.61 | 53.10 | +9.49 | +21.76% |
| GPU compute time, median (µs) | 43.46 | 52.49 | +9.03 | +20.79% |
| GPU compute time, p99 (µs) | 46.14 | 59.05 | +12.91 | +27.98% |
| D2H latency, mean (µs) | 27.32 | 50.99 | +23.67 | +86.64% |
| Throughput (qps) | 16,127.9 | 12,380.6 | −3,747.3 | −23.23% |

Mean D2H time contributed **23.67 µs of the 34.69 µs increase in mean latency**, about 68%. This is consistent with copying a second 512 KiB output back to the host. The mean component changes add up to the change in mean latency. Percentiles are computed per metric, so adding component percentiles does not reproduce the corresponding total-latency percentile.

### Per-query transfers disabled

Source: `results/baseline_no_transfers.log` and `results/exposed_no_transfers.log`, `Performance summary` from `trtexec --noDataTransfers`. The timed portions contained 65,806 baseline queries and 51,680 exposed-output queries. Both logs report zero timed H2D and D2H latency. The exposed engine still produces both device-side outputs.

| Metric | Baseline | Exposed output | Δ | Δ% |
| --- | ---: | ---: | ---: | ---: |
| GPU compute time, mean (µs) | 42.21 | 54.15 | +11.94 | +28.29% |
| GPU compute time, median (µs) | 42.24 | 54.14 | +11.90 | +28.18% |
| GPU compute time, p99 (µs) | 43.15 | 55.42 | +12.27 | +28.43% |
| Throughput (qps) | 21,934.6 | 17,225.9 | −4,708.7 | −21.47% |

The remaining timing gap measures the difference between the two engines while producing their respective device outputs. It includes the consequences of materializing the extra output and running the standalone ReLU.

### Effect of disabling copies within each engine

Here Δ = transfers disabled − transfers enabled. All percentages use the transfer-enabled value for the same engine as the denominator.

| Metric | Baseline enabled → disabled | Baseline Δ | Exposed enabled → disabled | Exposed Δ |
| --- | ---: | ---: | ---: | ---: |
| Reported latency, median (µs) | 92.29 → 42.24 | −50.05 (−54.23%) | 127.20 → 54.14 | −73.06 (−57.44%) |
| GPU compute time, median (µs) | 43.46 → 42.24 | −1.22 (−2.81%) | 52.49 → 54.14 | +1.65 (+3.14%) |
| Throughput (qps) | 16,127.9 → 21,934.6 | +5,806.7 (+36.00%) | 12,380.6 → 17,225.9 | +4,845.3 (+39.14%) |

These comparisons are between separate invocations. The small shifts in GPU compute time across transfer settings have no established cause in this experiment.

## CUDA execution trace

Source: Nsight Systems `cuda_gpu_kern_sum` for both `*_no_transfers_trace.nsys-rep` captures. Each capture used `--noDataTransfers --warmUp=0 --duration=0 --iterations=50`.

| CUDA kernel | Baseline launches | Exposed launches | Baseline mean (µs) | Exposed mean (µs) |
| --- | ---: | ---: | ---: | ---: |
| `trt_ampere_scudnn_winograd_128x128_ldg1_ldg4_relu_tile148t_nt_v1` | 50 | 50 | 106.406 | 106.417 |
| `generatedNativePointwise` | 0 | 50 | — | 26.242 |
| Sum of kernel durations per inference | 1 per inference | 2 per inference | 106.406 | 132.659 |

The convolution means differed by **0.012 µs**. The additional pointwise kernel accounts for almost all of the **26.253 µs (+24.67%)** increase in summed kernel duration in the traces. Here the sum is total GPU kernel time across the capture divided by 50. It excludes gaps between kernels. These short, no-warm-up profiling runs measure a different quantity under different conditions from the warmed-up benchmark. Use the unprofiled **11.90 µs** median gap for the inference timing claim and the trace to identify the extra launch.

The exposed capture's `cuda_gpu_mem_time_sum` reported one H2D memcpy (12.800 µs), one CUDA memset (2.144 µs), zero D2H memcpys, and zero device-to-device memcpys across the 50 inferences. The singleton operations are consistent with setup, although the aggregate report does not show their placement. The two `NoOp` entries corresponded to no additional observed CUDA kernel launches or device-to-device memcpys. GPU loads and stores inside kernels are outside the scope of this memcpy report.

## Reproduce the measurements

Generate or obtain the two ONNX inputs described above. The build commands below assume `baseline.onnx` and `exposed.onnx` are at the repo root. Adjust their paths to match your generator. Run on the target Jetson with TensorRT and Nsight Systems installed.

```bash
set -e
mkdir -p results
trtexec=/usr/src/tensorrt/bin/trtexec

for plan in baseline exposed; do
  "$trtexec" --onnx="${plan}.onnx" \
    --saveEngine="results/${plan}.plan" \
    --profilingVerbosity=detailed --skipInference \
    > "results/${plan}_build.log" 2>&1

  "$trtexec" --loadEngine="results/${plan}.plan" --dumpLayerInfo \
    > "results/${plan}_inspect.log" 2>&1

  "$trtexec" --loadEngine="results/${plan}.plan" --noDataTransfers \
    > "results/${plan}_no_transfers.log" 2>&1
done
```

```bash
set -euo pipefail
trtexec=/usr/src/tensorrt/bin/trtexec

for plan in baseline exposed; do
  trace="results/${plan}_no_transfers_trace"
  nsys profile --trace=cuda,nvtx --sample=none --output="$trace" \
    "$trtexec" --loadEngine="results/${plan}.plan" \
    --noDataTransfers --warmUp=0 --duration=0 --iterations=50

  nsys stats --report cuda_gpu_kern_sum "${trace}.nsys-rep" \
    | tee "results/${plan}_kernel_summary.txt"
done

nsys stats --report cuda_gpu_mem_time_sum \
  results/exposed_no_transfers_trace.nsys-rep
```

## Scope of the result

This is a case study of one convolution shape, one graph-output change, and one TensorRT and GPU configuration. It shows that exposing the pre-ReLU tensor changed fusion and added one recurring kernel launch. The relative timing describes the measured engine pair. A repeatability study would alternate execution order over multiple invocations, record power and clock conditions, and compare `relu_out` using the same saved input tensor.

## References

- [NVIDIA TensorRT 10.x performance benchmarking with `trtexec`](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/performance/benchmarking.html)
- [NVIDIA TensorRT 10.x profiling with Nsight Systems](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/performance/measurement-techniques.html)
