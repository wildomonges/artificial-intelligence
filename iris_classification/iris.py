"""
@author Wildo Monges
This code is in the book named "Building Machine Learning
Systems with Python - Willi Richert and Luis Pedro Coelho"
I write the code for learn and apply machine learning purpose
"""

from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
import numpy as np

# We load the data with load_iris from sklearn
data = load_iris()
features = data['data']
features_names = data['feature_names']
target = data['target']

for t, marker, c in zip(range(3), ">ox", "rgb"):
    # we plot each class
    plt.scatter(features[target == t, 0],
                features[target == t, 1],
                marker=marker,
                c=c)
