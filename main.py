import numpy as np
import pandas as pd
import json
import pprint as pp
from sklearn import metrics, datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score, r2_score
from models import linearRegression
from util import vis

PATH = "data/data/mpd.slice.0-999.json"

class exploreData():
    def __init__(self, path):
        self.data = self.readData(path)
        self.displayData()
    
    def readData(self, path):
        with open(path) as f:
            data = json.load(f)
            playlists = data["playlists"]
        return pd.DataFrame(playlists)
    
    def displayData(self):
        data = self.data
        vis.displayPlaylistLengthDistribution(data)
        vis.displayPopularArtists(data)

if __name__ == "__main__":
    x = exploreData(PATH)

