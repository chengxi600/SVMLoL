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

#testing_data should be an array of testing data with label at end
def get_acc(testing_data):
    count = 0
    total = 0
    for features in testing_data:
        params = features[:-1]
        correct = features[-1]
        total = total + 1
        predicted = model.predict([params])
        if predicted == correct:
            count = count + 1      
        print('Predicted: ' + str(predicted) + '\nActual: ' + str(correct) + '\nAccuracy: ' + str(count/total) + '\n')

    return count/total
        
sup_2020_playoffs_url = "https://lol.gamepedia.com/Special:RunQuery/TournamentStatistics?TS%5Bpreload%5D=TournamentByChampionRole&TS%5Brole%5D=Support&TS%5Btournament%5D=LCS/2020%20Season/Spring%20Playoffs&pfRunQueryFormName=TournamentStatistics"
top_2020_playoffs_url = "https://lol.gamepedia.com/Special:RunQuery/TournamentStatistics?TS%5Bpreload%5D=TournamentByChampionRole&TS%5Brole%5D=Top&TS%5Btournament%5D=NA%20Academy%20League/2020%20Season/Spring%20Playoffs&pfRunQueryFormName=TournamentStatistics"
mid_2020_playoffs_url = "https://lol.gamepedia.com/Special:RunQuery/TournamentStatistics?TS%5Bpreload%5D=TournamentByChampionRole&TS%5Brole%5D=Mid&TS%5Btournament%5D=NA%20Academy%20League/2020%20Season/Spring%20Playoffs&pfRunQueryFormName=TournamentStatistics"
adc_2020_playoffs_url = "https://lol.gamepedia.com/Special:RunQuery/TournamentStatistics?TS%5Bpreload%5D=TournamentByChampionRole&TS%5Brole%5D=AD%20Carry&TS%5Btournament%5D=NA%20Academy%20League/2020%20Season/Spring%20Playoffs&pfRunQueryFormName=TournamentStatistics"
jg_2020_playoffs_url = "https://lol.gamepedia.com/Special:RunQuery/TournamentStatistics?TS%5Bpreload%5D=TournamentByChampionRole&TS%5Brole%5D=Jungle&TS%5Btournament%5D=NA%20Academy%20League/2020%20Season/Spring%20Playoffs&pfRunQueryFormName=TournamentStatistics"
inputs = data.parse_data(jg_2020_playoffs_url, [6, 7, 8, 11, 13, 15, 16])

for params in inputs:
    params.append(1)

get_acc(inputs)