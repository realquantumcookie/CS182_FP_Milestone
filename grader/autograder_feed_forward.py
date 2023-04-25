import torch
import numpy as np

FEED_FORWARD_FILE = "grader/feed_forward_test.npz"
FEED_FORWARD_SUB_FILE = "grader/feed_forward_test_sub.npz"

def grade_feed_forward(feed_forward_class):
    data = np.load(FEED_FORWARD_FILE)
    seed = data["seed"]
    feed_forward_in = torch.tensor(data["input"])
    feed_forward_out = data["output"]

    torch.manual_seed(seed)
    feed_forward_layer = feed_forward_class(1024)
    out = feed_forward_layer(feed_forward_in)
    out = out.detach().numpy()
    assert np.allclose(out, feed_forward_out, rtol=1e-3)

def generate_feed_forward_sub(feed_forward_class):
    data = np.load(FEED_FORWARD_SUB_FILE)
    seed = data["seed"]
    feed_forward_in = torch.tensor(data["input"])

    torch.manual_seed(seed)
    feed_forward_layer = feed_forward_class(1024)
    out = feed_forward_layer(feed_forward_in)
    out = out.detach().numpy()
    return out