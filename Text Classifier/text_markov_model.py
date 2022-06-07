from typing import List, Dict
from collections import Counter

import numpy as np


class TextMarkovModel:

    def __init__(
        self,
        data: List[str],
        words_map: Dict[str, int],
        log_prior: float
    ):
        V = len(words_map)

        self.__raw_data = data
        self.__log_prior = log_prior
        self.__A = np.ones((V, V))
        self.__pi = np.ones(V)

        self.__compute_pi()
        self.__compute_matrix_A()

    def __compute_matrix_A(self):
        for _input in self.__raw_data:
            for last_idx, idx in zip(_input[:-1], _input[1:]):
                self.__A[last_idx, idx] += 1

        self.__A /= self.__A.sum(axis=1, keepdims=True)
        self.__log_A = np.log(self.__A)

    def __compute_pi(self):
        first_column = [_input[0] for _input in self.__raw_data]

        for word_idx, count in Counter(first_column).most_common():
            self.__pi[word_idx] = count

        self.__pi /= self.__pi.sum()
        self.__log_pi = np.log(self.__pi)

    def get_proba(self, _input: List[int]):
        proba = self.__log_pi[_input[0]]

        for last_idx, idx in zip(_input[:-1], _input[1:]):
            proba += self.__log_A[last_idx, idx]

        return proba + self.__log_prior
