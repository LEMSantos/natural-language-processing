import string
from typing import List, Union
from collections import Counter

import numpy as np


class CipherModel:

    def __init__(self, data: List[str]):
        letters_qty = len(list(string.ascii_lowercase))

        self.__raw_data = data
        self.__A = np.ones((letters_qty, letters_qty))
        self.__pi = np.ones(letters_qty)

        self.__compute_pi()
        self.__compute_matrix_A()

    def __compute_matrix_A(self):
        for sentence in self.__raw_data:
            for word in sentence.split():
                for char, next_char in zip(word[:-1], word[1:]):
                    self.__A[
                        self.__get_char_idx(char),
                        self.__get_char_idx(next_char)
                    ] += 1

        self.__A /= self.__A.sum(axis=1, keepdims=True)
        self.__log_A = np.log(self.__A)

    def __compute_pi(self):
        first_words_letters = [
            word[0]
            for sentence in self.__raw_data
            for word in sentence.split()
        ]

        for char, count in Counter(first_words_letters).most_common():
            self.__pi[self.__get_char_idx(char)] = count

        self.__pi /= self.__pi.sum()
        self.__log_pi = np.log(self.__pi)

    def __get_char_idx(self, char: str) -> int:
        return ord(char) - 97

    def __get_word_proba(self, word: str) -> float:
        proba = self.__log_pi[self.__get_char_idx(word[0])]

        for char, next_char in zip(word[:-1], word[1:]):
            proba += self.__log_A[
                self.__get_char_idx(char),
                self.__get_char_idx(next_char)
            ]

        return proba

    def get_sequence_proba(self, sequence: Union[str, List[str]]) -> float:
        if isinstance(sequence, str):
            sequence = sequence.split()

        return sum([self.__get_word_proba(word) for word in sequence])
