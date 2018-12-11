import json, argparse, os, random

import pprint as pp
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import metrics, datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score, r2_score

from models.NNeighClassifier import NNeighClassifier
from models.BaseClassifier import BaseClassifier
from util import vis, dataIn
from util.helpers import playlistToSparseMatrixEntry, getPlaylistTracks
#from test.test import TestTracks

class SpotifyExplorer:
    def __init__(self, numFiles, classifier="NNC"):
        self.readData(numFiles)

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

    def readData(self, numFilesToProcess):
        # don't have to write every time
        if numFilesToProcess > 0:
            # extract number from file
            def sortFile(f):
                f = f.split('.')[2].split('-')[0]
                return int(f)
            files = os.listdir("data/data")
            files.sort(key=sortFile)

            dataIn.createDFs(idx=0, 
                numFiles=numFilesToProcess,
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

    def predictNeighbour(self, playlist, numPredictions, songs):
        return self.classifier.predict(playlist, numPredictions, songs)
        
    #TODO change this later
    def displayData(self):
        pass
        #data = self.data
        #vis.displayPlaylistLengthDistribution(data)
        #vis.displayPopularArtists(data)
        #vis.displayMostCommonKeyWord(data)

    def obscurePlaylist(self, playlist, obscurity): 
        k = len(playlist['tracks']) * obscurity // 100
        indices = random.sample(range(len(playlist['tracks'])), k)
        obscured = [playlist['tracks'][i] for i in indices]
        tracks = [i for i in playlist['tracks'] + obscured if i not in playlist['tracks'] or i not in obscured]
        return tracks, obscured

    """
    Obscures a percentage of songs
    Iterates and sees how many reccomendations match the missing songs
    """
    def test(self, iterations, percent=50): 
        print("Selecting", iterations, "Playlists...")
        print("Obscuring", percent, "% of values ")

        accuracies = []
        for i in tqdm(range(iterations)):
            playlist = self.getRandomPlaylist()

            keptTracks, obscured = self.obscurePlaylist(playlist, percent)
            playlistSub = playlist.copy()
            playlistSub['tracks'] = keptTracks

            predictions = self.predictNeighbour(playlistSub, len(obscured), self.songs)

            obscuredTracks = [self.songs.loc[x]['track_name'] for x in obscured]
            
            overlap = [value for value in predictions if value in obscuredTracks]

            accuracy = len(overlap)/len(obscuredTracks)
            accuracies.append(accuracy)

        print("Using model",self.currentClassifier, ", we have an accuracy that averages", sum(accuracies)/len(accuracies), "across", iterations, "iterations")




if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--parseData')
    args = parser.parse_args()
    if args.parseData:
        numToParse = int(args.parseData)
    else:
        numToParse = 0

    """
    Builds explorer
    numFiles: Number of files to load (each with 1000 playlists)
    parse:    Boolean to load in data
    """

    spotify_explorer = SpotifyExplorer(
        numToParse, 
        classifier="Base")

    #Create our classifiers
    spotify_explorer.buildNNC()
    spotify_explorer.buildBaseClassifier()

    #Run tests on Base
    spotify_explorer.test(30)

    spotify_explorer.switchClassifier(classifier="NNC")

    #Run teset on our model
    spotify_explorer.test(30)
