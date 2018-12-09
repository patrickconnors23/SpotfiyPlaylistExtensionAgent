import os, pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, r2_score
import matplotlib
import matplotlib.pyplot as plt

class KMeansCluster():
    def __init__(self, playlists, reTrain=False, name="KMeansCluster.pkl"):
        self.pathName = name
        self.playlistData = playlists
        self.initModel(reTrain)
    
    def initModel(self, reTrain):
        libContents = os.listdir("lib")
        if self.pathName not in libContents or reTrain:
            self.model = KMeans(n_clusters=5)
            self.trainModel(self.playlistData)
        else:
            self.model = pickle.load(open(f"lib/{self.pathName}", "rb"))
    
    def trainModel(self, data):
        dummy = [
            [1,2,3],
            [1,6,3],
            [1,3,3],
            [1,6,3],
            [1,1,3],
            [7,6,3],
            [9,2,3],
            [1,9,3],
            [1,2,9],
            [1,2,70],
        ]
        print(f"Training KMeans clusterer")
        self.model.fit(dummy)
        self.saveModel()
    
    def predict(self, X):
        return self.model.predict(X)
    
    def saveModel(self):
        pickle.dump(self.model, open(f"lib/{self.pathName}", "wb"))
    
    def tagPlaylistClusters(self, data):
        data["clusterNum"] = data.apply(lambda row: self.model.predict(row), axis=1)
        return data