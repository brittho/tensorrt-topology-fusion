
import os
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

OUT = "models"
os.makedirs(OUT, exist_ok=True)

# Fixed seed: both models use exactly the same weights.
rng = np.random.default_rng(20261010)

input_info = helper.make_tensor_value_info(
    "input", TensorProto.FLOAT, [1, 16, 64, 64]
)
conv_info = helper.make_tensor_value_info(
    "conv_out", TensorProto.FLOAT, [1, 32, 64, 64]
)
relu_info = helper.make_tensor_value_info(
    "relu_out", TensorProto.FLOAT, [1, 32, 64, 64]
)

weights = (rng.standard_normal((32, 16, 3, 3)) * 0.05).astype(
    np.float32
)
bias = (rng.standard_normal(32) * 0.01).astype(np.float32)

initializers = [
    numpy_helper.from_array(weights, name="W"),
    numpy_helper.from_array(bias, name="B"),
]

nodes = [
    helper.make_node(
        "Conv",
        ["input", "W", "B"],
        ["conv_out"],
        name="conv",
        pads=[1, 1, 1, 1],
        strides=[1, 1],
    ),
    helper.make_node(
        "Relu",
        ["conv_out"],
        ["relu_out"],
        name="relu",
    ),
]

def save_model(path, expose_conv):
    outputs = [relu_info]
    if expose_conv:
        outputs = [conv_info, relu_info]

    graph = helper.make_graph(
        nodes,
        "conv_relu_test",
        [input_info],
        outputs,
        initializer=initializers,
    )

    model = helper.make_model(
        graph,
        producer_name="tensorrt-topology-fusion",
        opset_imports=[helper.make_opsetid("", 13)],
    )
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, path)
    print(f"Saved {path}")

save_model("models/baseline.onnx", expose_conv=False)
save_model("models/exposed.onnx", expose_conv=True)

