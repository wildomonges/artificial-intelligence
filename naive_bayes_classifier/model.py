"""
@author Wildo Monges
Store all information in the model, probabilities, amount of words, etc
"""


class Model:
    def __init__(self):
        self.total_files_by_class = {}
        self.total_words_by_class = {}
        self.classes = {}
        self.words = {}
        self.probabilities = {}

    def set_total_files_by_class(self, class_label, total):
        """Return amount of files by class"""
        self.total_files_by_class[class_label] = total

    def add_word_by_class(self, word, c):
        """Add word by class"""
        if c not in self.classes:
            self.classes[c] = {}
            self.total_words_by_class[c] = 0

        if word not in self.words:
            self.words[word] = 1
        else:
            self.words[word] += 1

        if word in self.classes[c]:
            self.classes[c][word] += 1
        else:
            self.classes[c][word] = 1
        self.total_words_by_class[c] += 1

    @staticmethod
    def laplace_smoothing(total_word_by_class, total_words, total_words_by_class, k):
        """Laplace Smoothing: https://en.wikipedia.org/wiki/Additive_smoothing"""
        k += 1
        d = k * 1.0 + total_word_by_class
        dv = total_words_by_class + (k * total_words)
        if dv == 0:
            return 0
        else:
            return d / dv

    def calculate_probability(self, k):
        """Calculate probabilities using Laplace Smoothing"""
        total_classes = len(self.classes)
        for c in self.classes:
            total_words = len(self.words)
            total_words_by_class = self.total_words_by_class[c]
            prob = self.laplace_smoothing(total_classes, total_words, total_words_by_class, k)
            self.probabilities[c] = prob

    @staticmethod
    def calculate_total_probability(probabilities):
        """Return total probabilities"""
        total_probabilities = []
        for p1 in probabilities:
            dv = 0
            for p2 in probabilities:
                dv += p2[1]
            aux_probs = p1[1] / dv
            total_probabilities.append((p1[0], aux_probs))
        total_probabilities = sorted(total_probabilities, key=lambda x: x[1])
        return total_probabilities

    def get_probability_by_class(self, c):
        return self.probabilities[c]

    def classes(self):
        return self.classes()

    def get_total_word_by_class(self, word, c):
        if word in self.classes[c]:
            return self.classes[c][word]
        else:
            return 0

    def get_total_words_by_class(self, c):
        if c in self.total_words_by_class:
            return self.total_words_by_class[c]
        else:
            return 0

    def get_total_words(self):
        return len(self.words)
