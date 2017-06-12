import warnings
from asl_data import SinglesData
import numpy as np


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key is a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    indices = []
    for idx, (X, lengths) in test_set.get_all_Xlengths().items():
        indices.append(idx)
        probs = dict()
        best_guess = None
        best_score = float('-inf')
        for word, model in models.items():
            score = float('-inf')
            try:
                score = model.score(X, lengths)
                if score > best_score:
                    best_guess = word
                    best_score = score
            except:
                print("Scoring failure on item {} with word {} ".format(idx, word))
            probs[word] = score
        probabilities.append(probs)
        guesses.append(best_guess)
    sorted_probabilities = []
    sorted_guesses = []
    for idx in np.argsort(indices):
        sorted_probabilities.append(probabilities[idx])
        sorted_guesses.append(guesses[idx])
    return sorted_probabilities, sorted_guesses
