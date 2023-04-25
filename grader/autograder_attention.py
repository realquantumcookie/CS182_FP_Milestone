import torch
import numpy as np

ATTENTION_FILE = "grader/attention_test.npz"
ATTENTION_SUB_FILE = "grader/attention_test_sub.npz"

def grade_attention(attention_class):
    data = np.load(ATTENTION_FILE)
    seed = data["seed"]
    attention_in = torch.tensor(data["input"])
    attention_out = data["output"]

    torch.manual_seed(seed)
    attention_layer = attention_class(1024)
    attention = attention_layer(attention_in)
    out = attention.detach().numpy()
    assert np.allclose(out, attention_out, rtol=1e-3)

def generate_attention_sub(attention_class):
    data = np.load(ATTENTION_SUB_FILE)
    seed = data["seed"]
    attention_in = torch.tensor(data["input"])

    torch.manual_seed(seed)
    attention_layer = attention_class(1024)
    attention = attention_layer(attention_in)
    out = attention.detach().numpy()
    return out