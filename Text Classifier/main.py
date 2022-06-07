import string

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score

from text_classifier import TextClassifier


def get_text_data(filename: str):
    with open(f'data/{filename}', 'r') as __file:
        lines = [line.rstrip().lower() for line in __file if line.rstrip()]

    return lines


def remove_punctuation(line: str):
    return line.translate(str.maketrans('', '', string.punctuation))


edgar_data = get_text_data('edgar_allan_poe.txt')
robert_data = get_text_data('robert_frost.txt')

edgar_data = list(map(remove_punctuation, edgar_data))
robert_data = list(map(remove_punctuation, robert_data))

input_texts = np.array([*edgar_data, *robert_data])
labels = np.array([0] * len(edgar_data) + [1] * len(robert_data))

X_train, X_test, Y_train, Y_test = train_test_split(
    input_texts, labels, random_state=42, train_size=0.8
)

clf = TextClassifier()
clf.fit(X_train, Y_train)

print(f"Train acc: {clf.score(X_train, Y_train)}")
print(f"Test acc: {clf.score(X_test, Y_test)}\n")

train_predicted = clf.predict(X_train)
test_predicted = clf.predict(X_test)

print('Confusion Matrix Train: \n', confusion_matrix(Y_train, train_predicted))
print('\nConfusion Matrix Test: \n', confusion_matrix(Y_test, test_predicted))

print('\nTrain F1 Score: ', f1_score(Y_train, train_predicted))
print('Test F1 Score: ', f1_score(Y_test, test_predicted))
