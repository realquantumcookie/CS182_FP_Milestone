import typing
import numpy as np
import os

class Autograder:
    def __init__(self):
        self._tokenizer_comb_ans = np.load(
            os.path.dirname(__file__) + "/tokenizer_comb_submission_ans.npz"
        )["possible_combinations"]
        self._tokenizer_ans = np.load(
            os.path.dirname(__file__) + "/tokenizer_sub.npz"
        )["tokenized_text"]
        self._pe_ans = np.load(
            os.path.dirname(__file__) + "/pe_sub_d12_m256.npz"
        )["pe"]
        self._attention_ans = np.load(
            os.path.dirname(__file__) + "/attention_test_sub.npz"
        )["output"]
        self._feed_forward_ans = np.load(
            os.path.dirname(__file__) + "/feed_forward_test_sub.npz"
        )["output"]
        self._transformer_layer_ans = np.load(
            os.path.dirname(__file__) + "/transformer_layer_test_sub.npz"
        )["output"]
    
    def grade(self, submission_dat : typing.Dict[str, np.ndarray]) -> np.ndarray:
        grader_result = []
        grader_result.append(self.catch_error_helper(self.grade_tokenizer_possible_combination, submission_dat))
        grader_result.append(self.catch_error_helper(self.grade_tokenizer, submission_dat))
        grader_result.append(self.catch_error_helper(self.grade_pe, submission_dat))
        grader_result.append(self.catch_error_helper(self.grade_attention, submission_dat))
        grader_result.append(self.catch_error_helper(self.grade_feed_forward, submission_dat))
        grader_result.append(self.catch_error_helper(self.grade_transformer_layer, submission_dat))
        return grader_result
    
    def catch_error_helper(self, question, submission_dat):
        try:
            return 1 if question(submission_dat) else 0
        except:
            return 0

    # Part 1: Tokenizer
    def grade_tokenizer_possible_combination(self, submission_dat : typing.Dict[str, np.ndarray]) -> bool:
        comb_submitted = submission_dat["tokenizer_comb"]
        comb_ans = self._tokenizer_comb_ans
        found_combination = False
        for possible_comb in comb_ans:
            if np.all(comb_submitted == possible_comb):
                found_combination = True
                break
        if not found_combination:
            return False
        return True
    
    def grade_tokenizer(self, submission_dat : typing.Dict[str, np.ndarray]) -> bool:
        submitted_tokenized = submission_dat["tokenizer"]
        ans_tokenized = self._tokenizer_ans
        if len(submitted_tokenized) != len(ans_tokenized):
            return False
        return np.all(submitted_tokenized == ans_tokenized)
    
    def grade_pe(self, submission_dat : typing.Dict[str, np.ndarray]) -> bool:
        submitted_pe = submission_dat["PE"]
        ans_pe = self._pe_ans
        return np.allclose(submitted_pe, ans_pe, rtol=1e-3)
    
    # Part 2: Transformer Layer
    def grade_attention(self, submission_dat : typing.Dict[str, np.ndarray]) -> bool:
        submitted_attention = submission_dat["Attention"]
        ans_attention = self._attention_ans
        return np.allclose(submitted_attention, ans_attention, rtol=1e-3)
    
    def grade_feed_forward(self, submission_dat : typing.Dict[str, np.ndarray]) -> bool:
        submitted_feed_forward = submission_dat["FeedForward"]
        ans_feed_forward = self._feed_forward_ans
        return np.allclose(submitted_feed_forward, ans_feed_forward, rtol=1e-3)
    
    def grade_transformer_layer(self, submission_dat : typing.Dict[str, np.ndarray]) -> bool:
        submitted_transformer_layer = submission_dat["TransformerLayer"]
        ans_transformer_layer = self._transformer_layer_ans
        return np.allclose(submitted_transformer_layer, ans_transformer_layer, rtol=1e-3)
