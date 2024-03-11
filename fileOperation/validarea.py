import pandas as pd
import math
import numpy as np
import tkinter as tk

dataH1 = pd.DataFrame
dataD1 = pd.DataFrame

dataH1 = dataH1.fillna(0)
dataD1 = dataD1.fillna(0)
edges = np.zeros(dataH1.shape[1])
for i in range(0,dataH1.shape[1]):
    max = 0.0
    for j in range(0,dataH1.shape[0]):
        if dataH1.iloc[j,i]>max:
            max = dataH1.iloc[j,i]
            edges[i] = j


def cal_curve(r, h):
    if r >= h:
        return math.pow(r, 2) * math.atan2((r-h) , r)
    else:
        return math.pow(r, 2) * math.atan2((h-r) , h)


toAdd = 1/300*math.pi*(abs(cal_curve(dataD1.iloc[i, j], dataH1.iloc[i, j]) - cal_curve(dataD1.iloc[i, j-1], dataH1.iloc[i, j-1])))