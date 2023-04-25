import torch
from torch import nn
import numpy as np
import math
import einops
import copy
from autograder_tokenizer import (
    DictionaryRef,
    generate_test_data_tokenizer
)

CORPUS_OF_TEXT_SIZE = 1_024
DICTIONARY_LENGTH = 306 # 256 (base dictionary size) + 50 (pairs to expand) = 306
# EMBEDDING_DIM = 256
# N_HEADS = 4
D_MODEL = 1_024 # 256 (embedding dim) * 4 (n_heads) = 1,024
SEED_SIZE = 1_000


def get_filename(filename: str, grader_internal: bool) -> str:
    if grader_internal:
        return "grader_internal/" + filename
    else:
        return "grader/" + filename


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout = 0.1, max_len: int = 1024):
        super().__init__()
        pe = self.calculate_pe(d_model, max_len)
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)

    def calculate_pe(self, d_model : int, max_len : int):
        """
        Calculate positional encoding for transformer

        Parameters
        ----------
        d_model : int
            The dimension of each embedding token
        
        max_len : int
            The maximum length of the sequence

        Returns
        -------
        pe : Tensor
            The positional encoding tensor of shape ``[1, max_len, d_model]``
        """
        # YOUR CODE HERE
        # raise NotImplementedError()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)).unsqueeze(0)
        pe = torch.zeros(1, max_len, d_model)
        combined_term = position * div_term
        pe[0, :, 0::2] = torch.sin(combined_term)
        pe[0, :, 1::2] = torch.cos(combined_term)
        return pe

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:,:x.shape[1],:]
        return self.dropout(x)

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, max_len: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model, dropout, max_len)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            tokens: Tensor, shape ``[batch_size, seq_len]``
        """
        return self.pe(self.embedding(tokens) * math.sqrt(self.d_model))
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def compute_masked_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """
        Calculate masked attention for transformer

        Parameters
        ----------
        Q: Input tensor of queries with size (batch_size, n_heads, tgt_len, d_head)
        K: Input tensor of keys with size (batch_size, n_heads, seq_len, d_head)
        V: Input tensor of values with size (batch_size, n_heads, seq_len, d_head)

        Returns
        -------
        Output attention tensor with size (batch_size, n_heads, tgt_len, d_head)
        """
        # YOUR CODE HERE
        _, _, seq_len, d_head = Q.shape
        weights = Q @ K.transpose(-2, -1) / (d_head ** 0.5)
        mask = torch.tril(torch.ones(seq_len, seq_len))
        masked_weights = weights.masked_fill(mask == 0, float('-inf'))
        return nn.functional.softmax(masked_weights, -1) @ V
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for multi-headed attention

        Parameters
        ----------
        x: Input tensor of tokens with size (batch_size, seq_len, d_model)

        Returns
        -------
        Output tensor after forward pass with size (batch_size, tgt_len, d_model)
        """
        # YOUR CODE HERE
        Q = self.query_proj(x)
        K = self.key_proj(x)
        V = self.value_proj(x)
        Q = einops.rearrange(Q, 'a b (c d) -> a c b d', c=self.n_heads)
        K = einops.rearrange(K, 'a b (c d) -> a c b d', c=self.n_heads)
        V = einops.rearrange(V, 'a b (c d) -> a c b d', c=self.n_heads)
        attention = self.compute_masked_attention(Q, K, V)
        attention = einops.rearrange(attention, 'a c b d -> a b (c d)', c=self.n_heads)
        out = self.output_proj(attention)
        return self.dropout(out)

# GRADER_FILENAME = "../grader/"
ATTENTION_FILENAME = "attention_test"
ATTENTION_SUB_FILENAME = "attention_test_sub"

def generate_attention_test_data(grader_internal: bool):
    _, _, tokenized_text = generate_test_data_tokenizer(DictionaryRef())
    tokens_tensor = torch.tensor(tokenized_text, dtype=torch.long).reshape(1, -1)
    embedding_layer = TokenEmbedding(DICTIONARY_LENGTH, D_MODEL)
    tokens_embed = embedding_layer(tokens_tensor)

    seed = np.random.randint(SEED_SIZE)
    torch.manual_seed(seed)
    attention_layer = MultiHeadAttention(D_MODEL)
    attention = attention_layer(copy.deepcopy(tokens_embed))
    attention_in = tokens_embed.detach().numpy()
    attention_out = attention.detach().numpy()
    # Embedded tokens are the input, Attention is the expected output
    if grader_internal:
        np.savez_compressed(
            get_filename(ATTENTION_SUB_FILENAME, False),
            seed=seed,
            input=attention_in,
        )
        np.savez_compressed(
            get_filename(ATTENTION_SUB_FILENAME, True),
            output=attention_out
        )
    else:
        np.savez_compressed(
            get_filename(ATTENTION_FILENAME, False), 
            seed=seed,
            input=attention_in,
            output=attention_out,
        )


# generate_attention_test_data(False)
# generate_attention_test_data(True)
