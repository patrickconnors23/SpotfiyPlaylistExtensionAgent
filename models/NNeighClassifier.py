import os, pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, r2_score
import matplotlib
import matplotlib.pyplot as plt

class NNeighClassifier():
    def __init__(self, playlists, reTrain=False, name="NNClassifier.pkl"):
        self.pathName = name
        self.playlistData = playlists
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
    
    def predict(self, X):
        predictions = []
        neighbors = kneighbors(X=X,
            return_distance=False)         
        # TODO list of indices of playlist locations in sparse matrix, should be same as indices in reg matrix
        # Step 1 -> convert playlist indicies to playlist and get track ids
        # Step 2 -> get most common track IDs that aren't in current playlist (X)
        # Step 3 (Maybe)-> Convert trackIDs of relevant tracks to song names
        return predictions
    
    def saveModel(self):
        pickle.dump(self.model, open(f"lib/{self.pathName}", "wb"))
    
    def tagPlaylistClusters(self, data):
        data["clusterNum"] = data.apply(lambda row: self.model.predict(row), axis=1)
        return data