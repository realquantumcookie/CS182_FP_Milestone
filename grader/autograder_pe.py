import numpy as np
import os

PE_TEST_ANS = np.load(
    os.path.dirname(__file__) + "/pe_test_d6_m128.npz"
)['pe']

def grade_pe(PE):
    pe = PE.calculate_pe(6,128)
    pe = pe.cpu().numpy()
    assert np.allclose(pe, PE_TEST_ANS, rtol=1e-3)

def generate_pe_sub(PE):
    pe = PE.calculate_pe(12,256)
    pe = pe.cpu().numpy()
    return pe