from typing import List, Dict, Any
from collections import Counter

import numpy as np

from text_markov_model import TextMarkovModel


class TextClassifier:

    def __init__(self):
        self.__models = {}
        self.__words_map = {}

    def __get_words_map(self, data: List[str]) -> Dict[str, int]:
        words_set = set(['<unk>'])

        for line in data:
            words_set.update(line.split())

        return {
            word: index
            for index, word in enumerate(words_set)
        }

    def __get_encoded_data(self, data: List[str]) -> List[List[int]]:
        unknown_idx = self.__words_map['<unk>']

        return [
            list(map(
                lambda word: self.__words_map.get(word, unknown_idx),
                line.split()
            ))
            for line in data
        ]

    def __get_models_prediction(self, _input: List[int]):
        return max(
            self.__models.items(),
            key=lambda item: item[1].get_proba(_input),
        )[0]

    def fit(self, data: List[str], labels: List[Any]) -> None:
        self.__words_map = self.__get_words_map(data)

        _classes = set(labels)
        _class_counter = Counter(labels)

        print(_class_counter.most_common())

        for _class in _classes:
            self.__models[_class] = TextMarkovModel(
                self.__get_encoded_data([
                    _input
                    for _input, label in zip(data, labels)
                    if label == _class
                ]),
                words_map=self.__words_map,
                log_prior=np.log(_class_counter[_class] / len(data)),
            )

    def predict(self, inputs: List[str]) -> List[Any]:
        encoded_data = self.__get_encoded_data(inputs)

        return [
            self.__get_models_prediction(_input)
            for _input in encoded_data
        ]


    def score(self, data: List[str], labels: List[int]):
        predicted = self.predict(data)
        return np.mean(predicted == labels)
