import numpy as np
import pandas as pd
from sklearn import metrics, datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score, r2_score
import matplotlib
import matplotlib.pyplot as plt


class exploreData():
    def __init__(self):
        self.data = pd.DataFrame()
    
    def readData(self, data):
        return

if __name__ == "__main__":
    x = exploreData()
    print(x)

