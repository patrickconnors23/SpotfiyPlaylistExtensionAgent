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

from models import linearRegression, KMeansCluster
from util import vis, dataIn

class exploreData():
    def __init__(self, idx, numFiles, parseFiles):
        self.readData(idx=idx,
            numFiles=numFiles,
            shouldProcess=parseFiles)
        self.KMC = KMeansCluster.KMeansCluster(self.playlists)
        print(self.KMC.predict([[4,5,5]]))
    
    def readData(self, idx, numFiles, shouldProcess):
        # don't have to write every time
        if shouldProcess:
            # extract number from file
            def sortFile(f):
                f = f.split('.')[2].split('-')[0]
                return int(f)
            files = os.listdir("data/data")
            files.sort(key=sortFile)

            dataIn.createDFs(idx=idx, 
                numFiles=numFiles,
                path="data/data/",
                files=files)

        # Read data
        print("Reading data")
        self.playlists = pd.read_pickle("lib/playlists.pkl")
        self.songs = pd.read_pickle("lib/tracks.pkl")
        self.playlistSparse = pd.read_pickle("lib/playlistSparse.pkl")
        print(f"Working with {len(self.playlists)} playlists " + \
            f"and {len(self.songs)} songs")
    
    #TODO change this later
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
    parser.add_argument('--parseData')
    args = parser.parse_args()
    if args.f:
        idx = int(args.f)
    else: 
        idx = 0
    if args.n:
        numFiles = int(args.n)
    else: 
        numFiles = 1
    if args.parseData:
        parse = True
    else:
        parse = False
    x = exploreData(idx, numFiles, parse)
