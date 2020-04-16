from bs4 import BeautifulSoup
import pandas as pd
import urllib
import re
import numpy as np
from decimal import Decimal

url_top = "https://lol.gamepedia.com/Special:RunQuery/TournamentStatistics?TS%5Bpreload%5D=TournamentByChampionRole&TS%5Brole%5D=Top&TS%5Bspl%5D=Yes&TS%5Btournament%5D=LCS/2020%20Season/Spring%20Season&pfRunQueryFormName=TournamentStatistics"
url_jg = "https://lol.gamepedia.com/Special:RunQuery/TournamentStatistics?TS%5Bpreload%5D=TournamentByChampionRole&TS%5Brole%5D=Jungle&TS%5Bspl%5D=Yes&TS%5Btournament%5D=LCS/2020%20Season/Spring%20Season&pfRunQueryFormName=TournamentStatistics"
url_mid = "https://lol.gamepedia.com/Special:RunQuery/TournamentStatistics?TS%5Bpreload%5D=TournamentByChampionRole&TS%5Brole%5D=Mid&TS%5Bspl%5D=Yes&TS%5Btournament%5D=LCS/2020%20Season/Spring%20Season&pfRunQueryFormName=TournamentStatistics"
url_ad = "https://lol.gamepedia.com/Special:RunQuery/TournamentStatistics?TS%5Bpreload%5D=TournamentByChampionRole&TS%5Brole%5D=AD%20Carry&TS%5Bspl%5D=Yes&TS%5Btournament%5D=LCS/2020%20Season/Spring%20Season&pfRunQueryFormName=TournamentStatistics"
url_sup = "https://lol.gamepedia.com/Special:RunQuery/TournamentStatistics?TS%5Bpreload%5D=TournamentByChampionRole&TS%5Brole%5D=Support&TS%5Bspl%5D=Yes&TS%5Btournament%5D=LCS/2020%20Season/Spring%20Season&pfRunQueryFormName=TournamentStatistics"
urls = [url_top, url_jg, url_mid, url_ad, url_sup]

#takes in url and params of data wanted
def parse_data(url, params):
    #connect to website
    page = urllib.request.urlopen(url)
    soup = BeautifulSoup(page, 'html.parser')

    res = []
    for row in soup.find('tbody').find_all('tr'):
        td = row.find_all('td')
        temp = []
        count = 0
        for cell in td:
            if count not in params:
                count = count + 1
            else:
                x = float(cell.text.strip('%'))
                temp.append(x)
                count = count + 1
        count = 0
        res.append(temp)
        temp = []
    return res[3:]
    print('parsed data')

def get_labels(sample_sizes):
    labels = np.array([])
    for i in range(len(sample_sizes)):
        labels = np.append(labels, np.full((1, sample_sizes[i]), i))
    return labels
    print('parsed labels')

def get_feature_vec(params):
    features = []
    sample_sizes = []
    for url in urls:
        for row in parse_data(url, params):
            features.append(row)
        sample_sizes.append(len(parse_data(url, params)))
    return features, sample_sizes

