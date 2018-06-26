"""
@author Wildo Monges
This program runs the classifier using naive bayes algorithm to classify
a group of files by topics
Note:
    Python 3.5
    Execute from console: python classifier.py
Result:
    K =  0  Corrects:  242  Random:  86
    K =  1  Corrects:  242  Random:  104
    K =  2  Corrects:  242  Random:  100
    K =  3  Corrects:  242  Random:  87
    K =  4  Corrects:  242  Random:  115
    K =  5  Corrects:  242  Random:  102
    K =  6  Corrects:  242  Random:  104
    K =  7  Corrects:  242  Random:  103
    K =  8  Corrects:  242  Random:  90
    K =  9  Corrects:  242  Random:  103
Conclusion:
    The classifier works better than a random classifier
"""
from os import listdir
from os.path import isfile, join
import naive_bayes_classifier.naive_bayes as naive_bayes
import random

LIMIT_K = 10

dirs = ('comp.os.ms-windows.misc', 'rec.sport.baseball', 'talk.politics.misc')

# Store the classifier model
model = None
k = 0
max_corrects = 0
for K in range(LIMIT_K):

    # Train phase
    # For each class (d is the directory that represent a class)
    for d in dirs:
        # Get all names from the files
        path = join('data', 'train', d)
        files = [join(path, f) for f in listdir(path) if isfile(join(path, f))]

        # train a model
        model = naive_bayes.train(model, files, d, K)

    # Cross validation phase
    corrects = 0
    random_corrects = 0
    for d in dirs:
        # Get file names
        path = join('data', 'validation', d)
        files = [join(path, f) for f in listdir(path) if isfile(join(path, f))]

        # For each file
        for fn in files:
            # Test classifier
            class_label = naive_bayes.classify(model, fn, K)
            if class_label == d:
                corrects += 1

            # Random test to compare
            random_class = dirs[random.randint(0, len(dirs) - 1)]
            if random_class == d:
                random_corrects += 1

    print('K = ', K, ' Corrects: ', corrects, ' Random: ', random_corrects)

    if corrects > max_corrects:
        k = K

