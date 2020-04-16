import pandas as pd
import numpy as np
from sklearn import svm
import data

import matplotlib.pyplot as plt
import seaborn as sns; sns.set(font_scale=1.2)

#Inputs for model
inputs, sample_sizes = data.get_feature_vec([6, 7, 8, 11, 13, 15, 16])
#if type is true put 0 else 1
type_label = data.get_labels(sample_sizes)

#Fit SVM Model
model = svm.SVC(kernel='rbf', C=1, gamma=2**-5)
model.fit(inputs, type_label)

def classify(num):
    if num == 0:
        print('Top')
    elif num == 1:
        print('Jungle')
    elif num == 2:
        print('Mid')
    elif num == 3:
        print('ADC')
    elif num == 4:
        print('Support')

classify(model.predict([[0.5, 3.21, 6.14, 1.03, 216, 4.1, 12.9]]))