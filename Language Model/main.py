import string
from typing import List

import numpy as np

from language_model import LanguageModel


def get_text_data(filename: str) -> List[str]:
    with open(f'data/{filename}', 'r') as __file:
        lines = [line.rstrip().lower() + ' END' for line in __file if line.rstrip()]

    return lines


def remove_punctuation(line: str) -> str:
    return line.translate(str.maketrans('', '', string.punctuation))


def tokenize(data: List[str]) -> List[List[str]]:
    return [line.split() for line in data]


robert_data = get_text_data('robert_frost.txt')
robert_data = list(map(remove_punctuation, robert_data))
robert_data = tokenize(robert_data)

model = LanguageModel(robert_data)

for phrase in model.generate():
    print(phrase)
