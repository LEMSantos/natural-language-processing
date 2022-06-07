import re
import string
from typing import List, Union

regex = re.compile('[^a-zA-Z\n]')


def get_text_data(filename: str):
    with open(f'data/{filename}', 'r') as __file:
        lines = [line.rstrip().lower() for line in __file if line.rstrip()]

    return lines


def remove_non_alpha_chars(
    data: Union[str, List[str]]
) -> Union[str, List[str]]:
    global regex

    if isinstance(data, str):
        return regex.sub(' ', data)

    return [regex.sub(' ', sequence) for sequence in data]


def get_cipher_map(cipher: List[str]):
    letters = list(string.ascii_lowercase)

    return {
        letter: encode
        for letter, encode in zip(letters, cipher)
    }


def encode_message(message: str, encode_map: List[str]) -> str:
    global regex

    new_message = message.lower()
    new_message = regex.sub(' ', new_message)
    new_message = re.sub('[ ]+', ' ', new_message)

    encoded_message = []

    for char in new_message:
        encoded_char = char

        if char in encode_map:
            encoded_char = encode_map[char]

        encoded_message.append(encoded_char)

    return ''.join(encoded_message)


def decode_message(encoded_message: str, encode_map: List[str]) -> str:
    decode_map = {value: key for key, value in encode_map.items()}

    decoded_message = []

    for char in encoded_message:
        encoded_char = char

        if char in decode_map:
            encoded_char = decode_map[char]

        decoded_message.append(encoded_char)

    return ''.join(decoded_message)


def print_cipher(cipher: List[str]) -> None:
    print('\nCipher:\n')
    print(*list(string.ascii_lowercase), sep=' ')
    print('-' * (len(cipher) * 2 - 1))
    print(*cipher, sep=' ')
    print()


def print_divider(title: str, divider_length=78) -> None:
    print('->', title.upper())
    print('-' * divider_length)
