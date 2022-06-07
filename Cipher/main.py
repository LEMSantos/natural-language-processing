import re
import sys
import string

import numpy as np

from genetic_algorithm import GeneticAlgorithm
from cipher_model import CipherModel
from utils import (
    get_cipher_map,
    get_text_data,
    print_cipher,
    remove_non_alpha_chars,
    encode_message,
    decode_message,
    print_divider,
)

sys.stdout = open('output.txt', 'w')

true_cipher = list(string.ascii_lowercase)
np.random.shuffle(true_cipher)

cipher_map = get_cipher_map(true_cipher)

moby_dick_book = get_text_data('moby_dick.txt')
moby_dick_book = remove_non_alpha_chars(moby_dick_book)

original_message = '''I then lounged down the street and found,
as I expected, that there was a mews in a lane which runs down
by one wall of the garden. I lent the ostlers a hand in rubbing
down their horses, and received in exchange twopence, a glass of
half-and-half, two fills of shag tobacco, and as much information
as I could desire about Miss Adler, to say nothing of half a dozen
other people in the neighbourhood in whom I was not in the least
interested, but whose biographies I was compelled to listen to.
'''

print_divider('Original Message')
print()
print(original_message)

encoded_message = encode_message(original_message, cipher_map)

print('\n')
print_divider('Original Message after encode with true cipher')
print_cipher(true_cipher)

print('Message:\n')
print(encoded_message)

model = CipherModel(moby_dick_book)
ag = GeneticAlgorithm(
    fitness_func=model.get_sequence_proba,
    population_size=30,
    keep_best=5,
    evolve_factor=3,
    evolve_probability=0.6,
)

print_divider('Start genetic algorithm')

predicted_cipher = ag.search(
    re.sub('\n', ' ', encoded_message), 20000
)
predicted_cipher_map = get_cipher_map(predicted_cipher)
decoded_message = decode_message(encoded_message, predicted_cipher_map)

print('\n')
print_divider('Original Message decoded by predicted cipher')
print_cipher(predicted_cipher)

print('Message:\n')
print(decoded_message)

print_divider('ciphers difference')

print('Predicted -> True\n')

for i, j in zip(predicted_cipher, true_cipher):
    if i != j:
        print('\t', i, '->', j)
