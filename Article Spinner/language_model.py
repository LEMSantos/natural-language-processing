from collections import Counter, defaultdict
from typing import Any, List

import numpy as np


class NormalizedCounter(Counter):

    def normalized(self, key: Any):
        total = sum(self.values())
        return self[key] / total if total else 0.0


class LanguageModel:

    def __init__(self, data: List[str]):
        self.__raw_data = data

        self.__pi = NormalizedCounter()
        self.__A0 = defaultdict(NormalizedCounter)
        self.__A1 = defaultdict(NormalizedCounter)

        self.__compute_pi()
        self.__compute_A0()
        self.__compute_A1()

    def __get_n_wise(self, elements: List[Any], n=2):
        for idx in range(0, len(elements) - n + 1):
            yield tuple(elements[idx:idx + n])

    def __compute_pi(self):
        first_column = [line[0] for line in self.__raw_data if line]
        self.__pi.update(first_column)

    def __compute_A0(self):
        for line in self.__raw_data:
            for key, word in self.__get_n_wise(elements=line, n=2):
                self.__A0[key].update([word])

    def __compute_A1(self):
        for line in self.__raw_data:
            for p_key, word, n_key in self.__get_n_wise(elements=line, n=3):
                self.__A1[(p_key, n_key)].update([word])

    def __sample_word(self, counter: NormalizedCounter):
        p0 = np.random.random()
        cumulative = 0

        for word, _ in counter.most_common():
            cumulative += counter.normalized(word)

            if p0 <= cumulative:
                return word

    def spin_article(self, article: List[List[str]], probability: float=0.2):
        new_article = []

        for sentence in article:
            new_sentence = sentence.copy()

            for idx in range(len(new_sentence) - 2):
                p = np.random.random()

                if p < probability:
                    key = (new_sentence[idx], new_sentence[idx + 2])

                    if key in self.__A1:
                        new_word = self.__sample_word(self.__A1[key])
                        new_sentence[idx + 1] = new_word

            new_article.append(new_sentence)

        return new_article
