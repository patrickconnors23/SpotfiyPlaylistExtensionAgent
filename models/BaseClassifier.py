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

class BaseClassifier:
    def __init__(self, playlists, songs, reTrain=False, name="BaseClassifier.pkl"):
        pass