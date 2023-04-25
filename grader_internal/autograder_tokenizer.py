import typing
import random
import numpy as np
import string
import copy

CORPUS_OF_TEXT_SIZE = 1_024

def generate_and_save_test_dictionary_class_find_combination_to_expand(
    dictionary_length,
    filename
):
    corpus_of_text, possible_combinations = generate_test_data_dictionary_class_find_combination_to_expand(
        dictionary_length
    )
    np.savez_compressed(
        filename,
        corpus_of_text=np.asarray(corpus_of_text),
        possible_combinations=np.asarray(possible_combinations)
    )

def generate_and_save_submission_dictionary_class_find_combination_to_expand(
    dictionary_length,
    filename_question,
    filename_answer
):
    corpus_of_text, possible_combinations = generate_test_data_dictionary_class_find_combination_to_expand(
        dictionary_length
    )
    np.savez_compressed(
        filename_question,
        corpus_of_text=np.asarray(corpus_of_text)
    )
    np.savez_compressed(
        filename_answer,
        possible_combinations=np.asarray(possible_combinations)
    )

def generate_test_data_dictionary_class_find_combination_to_expand(
    dictionary_length
) -> typing.Tuple[typing.List[int], typing.List[typing.Tuple[int,int]]]:
    corpus_of_text = []
    for _ in range(CORPUS_OF_TEXT_SIZE):
        corpus_of_text.append(random.randint(0, dictionary_length - 1))
    
    count_dict = {}
    for i in range(len(corpus_of_text) - 1):
        if (corpus_of_text[i], corpus_of_text[i+1]) in count_dict:
            count_dict[(corpus_of_text[i], corpus_of_text[i+1])] += 1
        else:
            count_dict[(corpus_of_text[i], corpus_of_text[i+1])] = 1
    count_arr = []
    for k, v in count_dict.items():
        count_arr.append((v, k))
    count_arr.sort()
    count_arr.reverse()
    
    possible_combinations = []
    possible_combinations_count = count_arr[0][0]
    for i in range(len(count_arr)):
        if count_arr[i][0] == possible_combinations_count:
            possible_combinations.append(count_arr[i][1])
        else:
            break
    return corpus_of_text, possible_combinations

def get_random_string(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase + string.ascii_uppercase + string.digits + " "
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str

def generate_test_data_tokenizer(
    dictionary,
    length : int = 1_024,
    num_newvocab : int = 50
):
    dict = copy.deepcopy(dictionary)
    random_text = get_random_string(length)
    expand_pairs = []
    for _ in range(num_newvocab):
        pair = dict.find_combination_to_expand(
            tokenize_ref(
                random_text,
                dict
            )
        )
        dict.expand_dictionary(
            pair
        )
        expand_pairs.append(pair)
    tokenized_text = tokenize_ref(random_text, dict)

    return random_text, expand_pairs, tokenized_text

def generate_and_save_test_data_tokenizer(
    dictionary,
    filename_text,
    filename_expand_pairs,
    filename_tokenized_text,
    length : int = 1_024,
    num_newvocab : int = 50
):
    random_text, expand_pairs, tokenized_text = generate_test_data_tokenizer(dictionary, length, num_newvocab)
    with open(filename_text, 'w') as f:
        f.write(random_text)

    np.savez_compressed(
        filename_expand_pairs,
        expand_pairs=np.asarray(expand_pairs)
    )
    
    np.savez_compressed(
        filename_tokenized_text,
        tokenized_text=np.asarray(tokenized_text)
    )
    
def grade_tokenizer(
    tokenizer,
    dictionary,
    filename_text,
    filename_expand_pairs,
    filename_tokenized_text
):
    with open(filename_text, "r") as f:
        text = f.read()
    
    expand_pairs = np.load(filename_expand_pairs)["expand_pairs"].tolist()
    tokenized_text = np.load(filename_tokenized_text)["tokenized_text"].tolist()
    
    for pair in expand_pairs:
        pair = tuple(pair)
        dictionary.expand_dictionary(pair)
    
    print(dictionary.__dict__)
    
    assert tokenized_text == tokenizer(text, dictionary), "Tokenizer is not correct"

class DictionaryRef:
    def __init__(self, base_dictionary : typing.List[bytes] = [i.to_bytes(1,'big') for i in range(256)]) -> None:
        
        # dictionary holds all volcabulary items and the index of each item in this array will be the input idx to the model
        self.dictionary_array : typing.List[bytes] = base_dictionary.copy()

        # This is a dictionary that maps a combination of two vocab items to a later vocab item
        self.combinations_to_index : typing.Dict[typing.Tuple[int, int], int] = {}
    
    def __len__(self) -> int:
        return len(self.dictionary_array)
    
    def __getitem__(self, key: int) -> str:
        return self.dictionary_array[key]
    
    def __contains__(self, key: str) -> bool:
        return key in self.dictionary_array
    
    def expand_dictionary(self, combination_vocab : typing.Tuple[int, int]) -> None:
        """
        This function should expand the dictionary with one more vocabulary item, 
        the item should be the concatenation of the two vocab items in combination_vocab
        You need to modify both the dictionary_array and combinations_to_index

        Parameters
        ----------
        combination_vocab : typing.Tuple[int, int]
            The combination of two vocab items to expand the dictionary with
        """
        # YOUR CODE HERE
        # raise NotImplementedError()
        self.dictionary_array.append(self.dictionary_array[combination_vocab[0]] + self.dictionary_array[combination_vocab[1]])
        self.combinations_to_index[combination_vocab] = len(self.dictionary_array) - 1
    

    def find_combination_to_expand(self, corpus_of_text: typing.List[int]) -> typing.Tuple[int, int]:
        """
        This function should find the combination of two vocab items that occurs the most in the corpus of text and return it
        
        Parameters
        ----------
        corpus_of_text : typing.List[int]
            The corpus of text represented by a list of integers (with each integer representing a vocab in the dictionary) to expand the dictionary with
        """
        # YOUR CODE HERE
        # raise NotImplementedError()
        count_dict = {}
        for i in range(len(corpus_of_text) - 1):
            if (corpus_of_text[i], corpus_of_text[i+1]) in count_dict:
                count_dict[(corpus_of_text[i], corpus_of_text[i+1])] += 1
            else:
                count_dict[(corpus_of_text[i], corpus_of_text[i+1])] = 1
        return max(count_dict, key=count_dict.get)

def tokenize_ref(text : str, dictionary : DictionaryRef) -> typing.List[int]:
    """
    This function should tokenize the text using the dictionary and return the tokenized text as a list of integers

    Parameters
    ----------
    text : str
        The text to tokenize
    
    dictionary : Dictionary
        The dictionary to use for tokenization
    """

    text_bytestream = bytes(text, "utf-8") # convert text to bytestream
    tokenized_text : typing.List[int] = [] # initialize tokenized text
    for i in range(len(text_bytestream)):
        tokenized_text.append(
            dictionary.dictionary_array.index(text_bytestream[i:i+1])
        )
    
    num_tokenized_last_pass = len(tokenized_text)
    # We will sweep through the tokenized text and replace any combination of two vocab items with the later vocab item
    while num_tokenized_last_pass > 0:
        # YOUR CODE HERE
        # raise NotImplementedError()
        num_tokenized_last_pass = 0
        new_tokenized_text = []
        for i in range(len(tokenized_text) - 1):
            if (tokenized_text[i], tokenized_text[i+1]) in dictionary.combinations_to_index:
                new_tokenized_text.append(dictionary.combinations_to_index[(tokenized_text[i], tokenized_text[i+1])])
                num_tokenized_last_pass += 1
            else:
                new_tokenized_text.append(tokenized_text[i])
                if i == len(tokenized_text) - 2:
                    new_tokenized_text.append(tokenized_text[i+1])
        
        tokenized_text = new_tokenized_text
    
    return tokenized_text