import numpy as np
import torch
from torch import nn
import copy
from autograder_attention import get_filename, MultiHeadAttention
from autograder_feed_forward import FeedForward

BATCH_SIZE = 4
SEQ_LEN = 1_024
D_MODEL = 1_024 # 256 (embedding dim) * 4 (n_heads) = 1,024
SEED_SIZE = 1_000

class TransformerLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attention_layer = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, dropout)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for transformer layer

        Parameters
        ----------
        x: Input tensor of tokens with size (batch_size, seq_len, d_model)

        Returns
        -------
        Output tensor after forward pass with size (batch_size, seq_len, d_model)
        """
        # YOUR CODE HERE
        attention = self.attention_layer(x)
        x = x + self.norm1(attention)
        out = self.feed_forward(x)
        out = x + self.norm2(out)
        return out
    
TRANSFORMER_LAYER_FILE = "transformer_layer_test"
TRANSFORMER_LAYER_SUB_FILE = "transformer_layer_test_sub"
    
def generate_transformer_layer_test_data(grader_internal: bool):
    seed = np.random.randint(SEED_SIZE)
    transformer_layer_in = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL).float()

    torch.manual_seed(seed)
    transformer_layer = TransformerLayer(D_MODEL)
    transformer_layer_out = transformer_layer(copy.deepcopy(transformer_layer_in))

    transformer_layer_in = transformer_layer_in.detach().numpy()
    transformer_layer_out = transformer_layer_out.detach().numpy()

    if grader_internal:
        np.savez_compressed(
            get_filename(TRANSFORMER_LAYER_SUB_FILE, False),
            seed=seed,
            input=transformer_layer_in,
        )
        np.savez_compressed(
            get_filename(TRANSFORMER_LAYER_SUB_FILE, True),
            output=transformer_layer_out
        )
    else:
        np.savez_compressed(
            get_filename(TRANSFORMER_LAYER_FILE, False), 
            seed=seed,
            input=transformer_layer_in,
            output=transformer_layer_out,
        )


# generate_transformer_layer_test_data(False)
# generate_transformer_layer_test_data(True)
