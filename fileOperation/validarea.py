import pandas as pd
import math
import numpy as np
import tkinter as tk

dataH1 = pd.DataFrame
dataD1 = pd.DataFrame

dataH1 = dataH1.fillna(0)
dataD1 = dataD1.fillna(0)
edges = list(dataH1.shape[1])
for i in range(0,dataH1.shape[1]):
    max = 0.0
    for j in range(0,dataH1.shape[0]):
        if dataH1.iloc[j,i]>max:
            max = dataH1.iloc[j,i]
            edges[i] = dataD1.iloc[j,i]

toAdd = 1/300*math.pi*(abs(math.pow(dataD1.iloc[i,j],2) - math.pow(dataD1.iloc[i,j-1],2)) + abs(math.pow(dataH1.iloc[i,j]) - math.pow(dataH1.iloc[i,j-1])))