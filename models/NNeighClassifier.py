import os, pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, r2_score
import matplotlib
import matplotlib.pyplot as plt

from util.helpers import playlistToSparseMatrixEntry, getPlaylistTracks

class NNeighClassifier():
    def __init__(self, playlists, sparsePlaylists, songs, reTrain=False, name="NNClassifier.pkl"):
        self.pathName = name
        self.playlistData = sparsePlaylists
        self.playlists = playlists 
        self.songs = songs
        self.initModel(reTrain)
    
    def initModel(self, reTrain):
        libContents = os.listdir("lib")
        if self.pathName not in libContents or reTrain:
            self.model = NearestNeighbors(
                n_neighbors=5,
                metric="cosine")
            self.trainModel(self.playlistData)
        else:
            self.model = pickle.load(open(f"lib/{self.pathName}", "rb"))
    
    def trainModel(self, data):
        print(f"Training Nearest Neighbors classifier")
        self.model.fit(data)
        self.saveModel()
    
    def getNeighbors(self, X):
        return self.model.kneighbors(X=X, return_distance=False)[0]
    
    def getPlaylistsFromNeighbors(self, playlists):
        return [self.playlists.loc[x] for x in playlists]
    
    def getPredictionsFromTracks(self, tracks):
        songs = defaultdict(int)
        for i, playlist in enumerate(tracks): 
            for song in playlist:
                track_name = song['track_name']
                songs[track_name] += (1/(i+1))
        scores = heapq.nlargest(7, songs, key=songs.get) 
        return scores

    
    def predict(self, X):
        predictions = []
        sparseX = playlistToSparseMatrixEntry(X, self.songs)
        neighbors = self.getNeighbors(sparseX) # PlaylistIDs
        playlists = self.getPlaylistsFromNeighbors(neighbors)
        tracks = [getPlaylistTracks(x, self.songs) for x in playlists]
        predictions = self.getPredictionsFromTracks(tracks)
        return predictions
    
    def saveModel(self):
        pickle.dump(self.model, open(f"lib/{self.pathName}", "wb"))
    
    def tagPlaylistClusters(self, data):
        data["clusterNum"] = data.apply(lambda row: self.model.predict(row), axis=1)
        return data