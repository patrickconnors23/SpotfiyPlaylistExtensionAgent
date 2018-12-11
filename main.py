import json, argparse, os, random

import pprint as pp
import numpy as np
import pandas as pd
from sklearn import metrics, datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score, r2_score

from models.NNeighClassifier import NNeighClassifier
from models.BaseClassifier import BaseClassifier
from util import vis, dataIn
from util.helpers import playlistToSparseMatrixEntry
#from test.test import TestTracks

class SpotifyExplorer:
    def __init__(self, idx, numFiles, parseFiles, classifier="NNC"):
        self.readData(idx=idx,
            numFiles=numFiles,
            shouldProcess=parseFiles)

        if classifier == "NNC": 
            self.currentClassifier = "NNC"
            self.classifier = self.buildNNC()
        else: 
            self.currentClassifier = "Base"
            self.classifier = self.buildBaseClassifier()

    def buildNNC(self): 
        self.NNC = NNeighClassifier(
            sparsePlaylists=self.playlistSparse,
            songs=self.songs,
            playlists=self.playlists,
            reTrain=True) 
        return self.NNC

    def buildBaseClassifier(self):
        self.baseClassifier = BaseClassifier(
            songs=self.songs,
            playlists=self.playlists)  
        return self.baseClassifier

    def switchClassifier(self, classifier=None): 
        if classifier == None: 
            classifier = "NNC" if classifier == "Base" else "Base"
        if classifier == "Base": 
            self.currentClassifier = "Base"
            self.classifier = self.baseClassifier
        else: 
            self.currentClassifier = "NNC"
            self.classifier = self.NNC

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
    
    def getRandomPlaylist(self): 
        return self.playlists.iloc[random.randint(0,len(self.playlists) - 1)]

    def predictNeighbour(self, playlist):
        return self.classifier.predict(playlist)
        
    #TODO change this later
    def displayData(self):
        pass
        #data = self.data
        #vis.displayPlaylistLengthDistribution(data)
        #vis.displayPopularArtists(data)
        #vis.displayMostCommonKeyWord(data)

    def test(self): 
        playlist = self.getRandomPlaylist()
        print("Selected playlist contains ", len(playlist), "songs")
        print("Obscuring half of them for testing...")
        self.predictNeighbour(playlist)





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

    """
    Builds explorer

    idx:      ????
    numFiles: Number of files to load (each with 1000 playlists)
    parse:    Boolean to load in data
    """

    spotify_explorer = SpotifyExplorer(idx, numFiles, parse)

    #Create our classifiers
    spotify_explorer.buildNNC()
    spotify_explorer.buildBaseClassifier()

    #Run tests
    spotify_explorer.test()
