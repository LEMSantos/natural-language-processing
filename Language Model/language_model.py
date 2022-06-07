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
        first_column = [line[0] for line in self.__raw_data]
        self.__pi.update(first_column)

    def __compute_A0(self):
        for line in self.__raw_data:
            for key, word in self.__get_n_wise(elements=line, n=2):
                self.__A0[key].update([word])

    def __compute_A1(self):
        for line in self.__raw_data:
            for *key, word in self.__get_n_wise(elements=line, n=3):
                self.__A1[tuple(key)].update([word])

    def __sample_word(self, counter: NormalizedCounter):
        p0 = np.random.random()
        cumulative = 0

        for word, _ in counter.most_common():
            cumulative += counter.normalized(word)

            if p0 <= cumulative:
                return word

    def generate(self, num_lines=5, max_words=100):
        sentences = []

        for _ in range(num_lines):
            words_count = 0
            first_word = self.__sample_word(self.__pi)
            second_word = self.__sample_word(self.__A0[first_word])

            sentence = [first_word, second_word]

            while True:
                next_word = self.__sample_word(
                    self.__A1[(first_word, second_word)]
                )

                if 'END' in [first_word, second_word, next_word]:
                    break

                sentence.append(next_word)
                words_count += 1

                first_word, second_word = second_word, next_word

            sentences.append(
                ' '.join([
                    word or '' for word in sentence
                ]).replace('END', '')
            )

        return sentences
