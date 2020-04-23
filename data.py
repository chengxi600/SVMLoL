from bs4 import BeautifulSoup
import pandas as pd
import urllib
import re
import numpy as np
from decimal import Decimal
from sklearn import preprocessing

#links for data to be scraped from
url_top = "https://lol.gamepedia.com/Special:RunQuery/TournamentStatistics?TS%5Bpreload%5D=TournamentByChampionRole&TS%5Brole%5D=Top&TS%5Bspl%5D=Yes&TS%5Btournament%5D=LCS/2020%20Season/Spring%20Season&pfRunQueryFormName=TournamentStatistics"
url_jg = "https://lol.gamepedia.com/Special:RunQuery/TournamentStatistics?TS%5Bpreload%5D=TournamentByChampionRole&TS%5Brole%5D=Jungle&TS%5Bspl%5D=Yes&TS%5Btournament%5D=LCS/2020%20Season/Spring%20Season&pfRunQueryFormName=TournamentStatistics"
url_mid = "https://lol.gamepedia.com/Special:RunQuery/TournamentStatistics?TS%5Bpreload%5D=TournamentByChampionRole&TS%5Brole%5D=Mid&TS%5Bspl%5D=Yes&TS%5Btournament%5D=LCS/2020%20Season/Spring%20Season&pfRunQueryFormName=TournamentStatistics"
url_ad = "https://lol.gamepedia.com/Special:RunQuery/TournamentStatistics?TS%5Bpreload%5D=TournamentByChampionRole&TS%5Brole%5D=AD%20Carry&TS%5Bspl%5D=Yes&TS%5Btournament%5D=LCS/2020%20Season/Spring%20Season&pfRunQueryFormName=TournamentStatistics"
url_sup = "https://lol.gamepedia.com/Special:RunQuery/TournamentStatistics?TS%5Bpreload%5D=TournamentByChampionRole&TS%5Brole%5D=Support&TS%5Bspl%5D=Yes&TS%5Btournament%5D=LCS/2020%20Season/Spring%20Season&pfRunQueryFormName=TournamentStatistics"
urls = [url_top, url_jg, url_mid, url_ad, url_sup]

#takes in string url and array of indexes of the parameters (features) you want from each sample on Gamepedia
#returns an array of samples for the specified url
def parse_data(url, params):
    #connect to website
    page = urllib.request.urlopen(url)
    soup = BeautifulSoup(page, 'html.parser')

    samples = []
    
    #loop through rows in table on Gamepedia
    for row in soup.find('tbody').find_all('tr'):
        #get list of all the cells on each row, each row being a sample
        td = row.find_all('td')
        #temporary array to store the row's features
        sample_features = []
        #keep track of current parameter (feature) index
        count = 0
        for cell in td:
            if count not in params:
                count = count + 1
            else:
                #if parameter is wanted, remove % and append cell text as a float to array of the row's features
                x = float(cell.text.strip('%'))
                sample_features.append(x)
                count = count + 1
        #reset count and append the sample into array of samples
        count = 0
        samples.append(sample_features)
        sample_features = []
    #slice first three as they are empty rows
    return samples[3:]

#takes in array of sizes of each class
#returns array of same size as the total samples array, where each element indicates the sample's class
def get_labels(sample_sizes):
    labels = np.array([])
    #iterate through each class and appends an array of a single number that represents the class to labels array
    for i in range(len(sample_sizes)):
        labels = np.append(labels, np.full((1, sample_sizes[i]), i))
    return labels

#takes in array of parameters (features) you want from each sample
#returns the sample array of all five positions and array of each class' size
def get_feature_vec(params):
    features = []
    sample_sizes = []
    #iterate through url array and append each position's samples array together
    for url in urls:
        for row in parse_data(url, params):
            features.append(row)
        sample_sizes.append(len(parse_data(url, params)))
    return features, sample_sizes

#takes in array of samples
#returns an array where each feature is scaled to a distribution 
def scaled_feature_vec(features):
    #mm_scaler = preprocessing.MinMaxScaler()
    #scaled = mm_scaler.fit_transform(features)
    return preprocessing.normalize(features)

#takes in the index of the parameter (feature) you want (from processed param array), the int 
#corresponding to a position, samples array, and array of class sizes
#returns an array of features of given position
def get_single_feature(param, position, inputs, sample_sizes):
    sample_sizes = np.cumsum(sample_sizes)

    arr = []
    if position == 0:
        for sample in inputs[:sample_sizes[position]]:
            arr.append(sample[param])
    elif position == 5:
        for sample in inputs[sample_sizes[position-1]:]:
            arr.append(sample[param])
    else:
        for sample in inputs[sample_sizes[position-1]:sample_sizes[position]]:
            arr.append(sample[param])
    return arr