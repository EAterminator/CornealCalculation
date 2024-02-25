# coding=utf-8
import pandas as pd
import math
import numpy as np
import tkinter as tk
from tkinter import filedialog
import os

dataHB = pd.DataFrame()
dataHF = pd.DataFrame()
for i in range(24):
    temp = pd.read_table('240129515R.V5B', sep=';', skiprows=19+i*601, nrows=600, names=['r', 'h'], header=None)
    dataHB['h'+str(i+1)] = temp['h']

for i in range(24):
    temp = pd.read_table('240129515R.V5F', sep=';', skiprows=20+i*692, nrows=691, names=['r', 'h'], header=None)
    dataHF['h'+str(i+1)] = temp['h']
print(dataHF)


