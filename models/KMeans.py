import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, r2_score
import matplotlib
import matplotlib.pyplot as plt

class KMeansCluster():
    def __init__(self):
        self.model = KMeans()
        self.path = None
    
    def predict(self, X):
        pass
    
    def saveModel(self):
        pass