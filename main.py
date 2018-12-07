import json, argparse, os

import pprint as pp
import numpy as np
import pandas as pd

from sklearn import metrics, datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score, r2_score

from models import linearRegression
from util import vis, dataIn

PATH = "data/data/mpd.slice.0-999.json"

class exploreData():
    def __init__(self, idx, numFiles):
        self.data = self.readData(idx=idx,
            numFiles=numFiles)
        # self.displayData()
    
    def readData(self, idx, numFiles):
        # don't have to write every time
        if True:
            # extract number from file
            def sortFile(f):
                f = f.split('.')[2].split('-')[0]
                return int(f)
            files = os.listdir("data/data")
            files.sort(key=sortFile)
            dirPath = "data/data/"
            dataIn.createCSVs(idx=idx, 
                numFiles=numFiles,
                path=dirPath,
                files=files)
        self.playlists = pd.read_csv("data/CSV/playlists.csv")
        self.songs = pd.read_csv("data/CSV/tracks.csv")

        # Keep data for Current vis model
        #TODO change this later
        with open(f"data/data/{files[idx]}") as f:
            data = json.load(f)
            playlists = data["playlists"]
        return pd.DataFrame(playlists)
    
    def displayData(self):
        data = self.data
        vis.displayPlaylistLengthDistribution(data)
        vis.displayPopularArtists(data)
        vis.displayMostCommonKeyWord(data)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-f')
    parser.add_argument('-n')
    args = parser.parse_args()
    if args.f:
        idx = int(args.f)
    else: 
        idx = 0
    if args.n:
        numFiles = int(args.n)
    else: 
        numFiles = 1
    x = exploreData(idx, numFiles)
