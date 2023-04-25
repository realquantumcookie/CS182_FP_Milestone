import torch
import numpy as np

TRANSFORMER_LAYER_FILE = "grader/transformer_layer_test.npz"
TRANSFORMER_LAYER_SUB_FILE = "grader/transformer_layer_test_sub.npz"

def grade_transformer_layer(transformer_layer_class):
    data = np.load(TRANSFORMER_LAYER_FILE)
    seed = data["seed"]
    transformer_layer_in = torch.tensor(data["input"])
    transformer_layer_out = data["output"]

    torch.manual_seed(seed)
    transformer_layer = transformer_layer_class(1024)
    out = transformer_layer(transformer_layer_in)
    out = out.detach().numpy()
    assert np.allclose(out, transformer_layer_out, rtol=1e-3)

def generate_transformer_layer_sub(transformer_layer_class):
    data = np.load(TRANSFORMER_LAYER_SUB_FILE)
    seed = data["seed"]
    transformer_layer_in = torch.tensor(data["input"])

    torch.manual_seed(seed)
    transformer_layer = transformer_layer_class(1024)
    out = transformer_layer(transformer_layer_in)
    out = out.detach().numpy()
    return out