import string
from typing import List

from language_model import LanguageModel


def get_text_data(filename: str) -> List[str]:
    with open(f'data/{filename}', 'r') as __file:
        lines = [line.rstrip().lower() for line in __file if line.rstrip()]

    return lines


def remove_punctuation(line: str) -> str:
    return line.translate(str.maketrans('', '', string.punctuation))


def tokenize(data: List[str]) -> List[List[str]]:
    return [line.split() for line in data]


print('Reading harry potter book data...')

harry_potter = get_text_data("Harry Potter - The Philosopher's Stone.txt")
harry_potter = list(map(remove_punctuation, harry_potter))
harry_potter = tokenize(harry_potter)

print('Reading moby dick book data...')

moby_dick = get_text_data("moby_dick.txt")
moby_dick = list(map(remove_punctuation, moby_dick))
moby_dick = tokenize(moby_dick)

text_data = harry_potter + moby_dick

print('Creating Language Model...')

model = LanguageModel(text_data)

print('Reading article data to spin...')

article = get_text_data('article.txt')
article = list(map(remove_punctuation, article))
article = tokenize(article)

print('Running article spinner...')

new_article = model.spin_article(article)

print('Saving new article...')

with open('new_article.txt', 'w') as file:
    for splited_line in new_article:
        file.write(' '.join(splited_line))
        file.write('\n')
