"""
@author Wildo Monges
This class train a classifier using naive bayes algorithm
"""

from naive_bayes_classifier.model import Model
import re
import math

def train(model, files, c, K):
    """
    Train a model
    :param model: model
    :param files: list of files
    :param c: class
    :param K: Laplace Smoothing factor
    :return: model
    """
    if model is None:
        model = Model()
    model.set_total_files_by_class(c, len(files))
    # for each file in the directory (c is the class)
    for file in files:
        f = open(file)
        buffer = f.read()
        buffer = re.sub(r'([^\s\w]|_)+', ' ', buffer)
        for word in buffer.lower().split():
            model.add_word_by_class(word, c)
        model.calculate_probability(K)
    return model


def classify(model, file, K):
    """
    Classify using the model
    :param model:
    :param file:
    :param K:
    :return: class
    """
    if model is None:
        return None
    classes = model.classes.keys()
    f = open(file)
    buffer = f.read()
    buffer = re.sub(r'([^\s\w]|_)+', ' ', buffer)
    probabilities = []
    for c in classes:
        prob_by_class = model.get_probability_by_class(c)
        probability = math.log((1 - prob_by_class) / prob_by_class)
        for word in buffer.lower().split():
            total_word_by_class = model.get_total_word_by_class(word, c)
            total_all_words = model.get_total_words()
            total_words_by_class = model.get_total_words_by_class(c)
            laplace = model.laplace_smoothing(total_word_by_class, total_all_words, total_words_by_class, K)
            pb = math.log((1 - laplace) / laplace)
            probability = probability * pb
        probabilities.append((c, probability))
    return model.calculate_total_probability(probabilities)[0][0]
