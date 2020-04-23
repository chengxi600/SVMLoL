import pandas as pd
import numpy as np
from sklearn import svm
import data

import matplotlib.pyplot as plt

#Clarifications:
#A feature is a parameter taken from a Champion e.g. Kill, Gold Share, CS
#A sample is an array of features taken from a Champion
#Training samples is an array of samples taken from LCS Spring Split 2019 used to train the model
#Testing samples is an array of samples taken from LCS Spring Playoffs 2020 used to test the model's accuracy

#indexes of features 
params = [7, 8, 11, 16]
#training samples and class size array
inputs, sample_sizes = data.get_feature_vec(params)
#scaling training samples
inputs = data.scaled_feature_vec(inputs)
#array of labels corresponding to training data
#0: top
#1: jg
#2: mid
#3: adc
#4: sup
type_label = data.get_labels(sample_sizes)

#Fit SVM Model
model = svm.LinearSVC(max_iter=10000)
#model = svm.SVC(gamma='scale', C=1, kernel='rbf')
model.fit(inputs, type_label)

#takes in an int corresponding to position
#prints position
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

#takes in array of testing data
#returns accuracy of model 
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
        #print('Predicted: ' + str(predicted) + '\nActual: ' + str(correct) + '\nAccuracy: ' + str(count/total) + '\n')

    return count/total

#takes in training samples and int corresponding to position
#returns training samples where the correct position is appended at the end of each sample
def get_testing(inputs, x):
    for params in inputs:
        params.append(x)
    return inputs

#takes in testing data for each position
#prints the accuracy for each position
def print_acc(top, jg, mid, adc, sup):
    print('Top: ' + str(get_acc(get_testing(top, 0))))
    print('Jg: ' + str(get_acc(get_testing(jg, 1))))
    print('Mid: ' + str(get_acc(get_testing(mid, 2))))
    print('Adc: ' + str(get_acc(get_testing(adc, 3))))
    print('Sup: ' + str(get_acc(get_testing(sup, 4))))

#takes in two indexes of a parameter (feature) 'a' and 'b', an array of samples, and array of class sizes
#return a scatter plot of the two features 
def plot_param(a, b, data_set, sample_sizes):
    fig, ax = plt.subplots()
    ax.scatter(data.get_single_feature(a, 0, data_set, sample_sizes), data.get_single_feature(b, 0, data_set, sample_sizes), color='b')
    ax.scatter(data.get_single_feature(a, 1, data_set, sample_sizes), data.get_single_feature(b, 1, data_set, sample_sizes), color='g')
    ax.scatter(data.get_single_feature(a, 2, data_set, sample_sizes), data.get_single_feature(b, 2, data_set, sample_sizes), color='r')
    ax.scatter(data.get_single_feature(a, 3, data_set, sample_sizes), data.get_single_feature(b, 3, data_set, sample_sizes), color='c')
    ax.scatter(data.get_single_feature(a, 4, data_set, sample_sizes), data.get_single_feature(b, 4, data_set, sample_sizes), color='y')
    ax.legend(['top', 'jg', 'mid', 'adc', 'sup'])
    ax.grid(True)
    plt.xlabel('Param ' + str(a))
    plt.ylabel('Param ' + str(b))
    plt.show()

#Scatterplot of Assists vs Gold Share
plot_param(1, 3, inputs, sample_sizes)
plt.clf()
#Scatterplot of Deaths vs Gold Share
plot_param(0, 3, inputs, sample_sizes)

#url strings of testing data for each position 
sup_2020_playoffs_url = "https://lol.gamepedia.com/Special:RunQuery/TournamentStatistics?TS%5Bpreload%5D=TournamentByChampionRole&TS%5Brole%5D=Support&TS%5Btournament%5D=LCS/2020%20Season/Spring%20Playoffs&pfRunQueryFormName=TournamentStatistics"
top_2020_playoffs_url = "https://lol.gamepedia.com/Special:RunQuery/TournamentStatistics?TS%5Bpreload%5D=TournamentByChampionRole&TS%5Brole%5D=Top&TS%5Btournament%5D=NA%20Academy%20League/2020%20Season/Spring%20Playoffs&pfRunQueryFormName=TournamentStatistics"
mid_2020_playoffs_url = "https://lol.gamepedia.com/Special:RunQuery/TournamentStatistics?TS%5Bpreload%5D=TournamentByChampionRole&TS%5Brole%5D=Mid&TS%5Btournament%5D=NA%20Academy%20League/2020%20Season/Spring%20Playoffs&pfRunQueryFormName=TournamentStatistics"
adc_2020_playoffs_url = "https://lol.gamepedia.com/Special:RunQuery/TournamentStatistics?TS%5Bpreload%5D=TournamentByChampionRole&TS%5Brole%5D=AD%20Carry&TS%5Btournament%5D=NA%20Academy%20League/2020%20Season/Spring%20Playoffs&pfRunQueryFormName=TournamentStatistics"
jg_2020_playoffs_url = "https://lol.gamepedia.com/Special:RunQuery/TournamentStatistics?TS%5Bpreload%5D=TournamentByChampionRole&TS%5Brole%5D=Jungle&TS%5Btournament%5D=NA%20Academy%20League/2020%20Season/Spring%20Playoffs&pfRunQueryFormName=TournamentStatistics"

#testing samples of each position
inputs_top = data.parse_data(top_2020_playoffs_url, params)
inputs_jg = data.parse_data(jg_2020_playoffs_url, params)
inputs_mid = data.parse_data(mid_2020_playoffs_url, params)
inputs_adc = data.parse_data(adc_2020_playoffs_url, params)
inputs_sup = data.parse_data(sup_2020_playoffs_url, params)

#prints accuracy of model 
print_acc(inputs_top, inputs_jg, inputs_mid, inputs_adc, inputs_sup)