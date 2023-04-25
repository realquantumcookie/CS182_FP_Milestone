import torch
import math
import numpy as np

def calculate_pe(d_model : int, max_len : int):
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
    div_term = (torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))).unsqueeze(0)
    pe = torch.zeros(1, max_len, d_model)
    combined_term = position * div_term
    pe[0, :, 0::2] = torch.sin(combined_term)
    pe[0, :, 1::2] = torch.cos(combined_term)
    return pe

def generate_and_save_pe(d_model, max_len, filename):
    pe : torch.Tensor = calculate_pe(d_model, max_len)
    mat = pe.cpu().numpy()
    np.savez_compressed(filename, pe=mat)