import numpy as np
import torch
from torch import nn
import math
import copy
from autograder_attention import get_filename

BATCH_SIZE = 4
SEQ_LEN = 1_024
D_MODEL = 1_024 # 256 (embedding dim) * 4 (n_heads) = 1,024
SEED_SIZE = 1_000

def gelu(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class FeedForward(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, 4 * d_model)
        self.gelu = gelu
        self.fc2 = nn.Linear(4 * d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for feed-forward block

        Parameters
        ----------
        x: Input tensor of tokens with size (batch_size, seq_len, d_model)

        Returns
        -------
        Output tensor after forward pass with size (batch_size, seq_len, d_model)
        """
        # YOUR CODE HERE
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
    
FEED_FORWARD_FILE = "feed_forward_test"
FEED_FORWARD_SUB_FILE = "feed_forward_test_sub"
    
def generate_feed_forward_test_data(grader_internal: bool):
    seed = np.random.randint(SEED_SIZE)
    feed_forward_in = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL).float()
    torch.manual_seed(seed)

    feed_forward_layer = FeedForward(D_MODEL)
    feed_forward_out = feed_forward_layer(copy.deepcopy(feed_forward_in))

    feed_forward_in = feed_forward_in.detach().numpy()
    feed_forward_out = feed_forward_out.detach().numpy()
    if grader_internal:
        np.savez_compressed(
            get_filename(FEED_FORWARD_SUB_FILE, False),
            seed=seed,
            input=feed_forward_in,
        )
        np.savez_compressed(
            get_filename(FEED_FORWARD_SUB_FILE, True),
            output=feed_forward_out
        )
    else:
        np.savez_compressed(
            get_filename(FEED_FORWARD_FILE, False), 
            seed=seed,
            input=feed_forward_in,
            output=feed_forward_out,
        )


# generate_feed_forward_test_data(False)
# generate_feed_forward_test_data(True)
