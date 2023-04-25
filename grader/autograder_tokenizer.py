import random
import copy
import numpy as np
import os

def grade_dictionary_class_expand_dictionary(dictionary, n = 50):
    for i in range(n):
        grade_dictionary_class_expand_dictionary_single(dictionary)

def grade_dictionary_class_expand_dictionary_single(dictionary):
    dict_cpy = copy.deepcopy(dictionary)
    random_words = (random.randint(0, len(dict_cpy) - 1), random.randint(0, len(dict_cpy) - 1))
    while random_words[0] == random_words[1]:
        random_words = (random.randint(0, len(dict_cpy) - 1), random.randint(0, len(dict_cpy) - 1))
    
    dict_cpy.expand_dictionary(random_words)
    assert len(dict_cpy) == len(dictionary) + 1, "Dictionary size is not correct"
    assert dict_cpy.dictionary_array == dictionary.dictionary_array + [dictionary.dictionary_array[random_words[0]] + dictionary.dictionary_array[random_words[1]]], "Dictionary array is not correct"
    assert dict_cpy.combinations_to_index[random_words] == len(dict_cpy) - 1, "Combinations to index is not correct"

DICTIONARY_CLASS_COMBINATION_TO_EXPAND_DATA = np.load(
    os.path.dirname(__file__) + "/dictionary_class_combination_to_expand_data.npz"
)

def grade_dictionary_class_find_combination_to_expand(dictionary):
    comb = dictionary.find_combination_to_expand(DICTIONARY_CLASS_COMBINATION_TO_EXPAND_DATA["corpus_of_text"].tolist())
    for possible_comb in DICTIONARY_CLASS_COMBINATION_TO_EXPAND_DATA["possible_combinations"]:
        if tuple(possible_comb) == comb:
            return
    assert False, "Combination is not correct"
    
DICTIONARY_CLASS_COMBINATION_TO_EXPAND_DATA_SUB = np.load(
    os.path.dirname(__file__) + "/tokenizer_comb_submission_question.npz"
)
def generate_dictionary_class_find_combination_to_expand_dat(dictionary) -> np.ndarray:
    comb = dictionary.find_combination_to_expand(DICTIONARY_CLASS_COMBINATION_TO_EXPAND_DATA_SUB["corpus_of_text"].tolist())
    return np.asarray(comb)

TOKENIZER_TEST_TEXT_FILE = os.path.dirname(__file__) + "/tokenizer_test.txt"
TOKENIZER_TEST_EXPAND_PAIRS = np.load(
    os.path.dirname(__file__) + "/tokenizer_test_pairs.npz"
)["expand_pairs"].tolist()
TOKENIZER_TEST_ANS = np.load(
    os.path.dirname(__file__) + "/tokenizer_test.npz"
)["tokenized_text"].tolist()

def grade_tokenizer(tokenizer, dictionary):
    dictionary = copy.deepcopy(dictionary)
    with open(TOKENIZER_TEST_TEXT_FILE, "r") as f:
        text = f.read()
    
    for pair in TOKENIZER_TEST_EXPAND_PAIRS:
        pair = tuple(pair)
        dictionary.expand_dictionary(pair)
    
    assert TOKENIZER_TEST_ANS == tokenizer(text, dictionary), "Tokenizer is not correct"


TOKENIZER_SUB_TEXT_FILE = os.path.dirname(__file__) + "/tokenizer_sub.txt"
TOKENIZER_SUB_EXPAND_PAIRS = np.load(
    os.path.dirname(__file__) + "/tokenizer_sub_pairs.npz"
)["expand_pairs"].tolist()

def generate_tokenizer_submission(tokenizer, dictionary):
    dictionary = copy.deepcopy(dictionary)
    
    with open(TOKENIZER_SUB_TEXT_FILE, "r") as f:
        text = f.read()

    for pair in TOKENIZER_SUB_EXPAND_PAIRS:
        pair = tuple(pair)
        dictionary.expand_dictionary(pair)

    tokenized_text = tokenizer(text, dictionary)
    return np.asarray(tokenized_text)