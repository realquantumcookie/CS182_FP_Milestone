from .autograder_tokenizer import *
from .autograder_pe import *
from .autograder_attention import *
from .autograder_feed_forward import *
from .autograder_transformer_layer import *

import typing
class AutograderSubmitter:
    def __init__(self):
        self.submission_data : typing.Dict[str, np.ndarray]  = {

        }
    
    def generate_submission_file(self, filename):
        np.savez_compressed(filename, **self.submission_data)