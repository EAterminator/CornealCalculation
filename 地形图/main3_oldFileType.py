# coding=utf-8
import pandas
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.interpolate import griddata
import tkinter as tk
from tkinter import filedialog
import os
import sys


def open_folder():
    file_list_axl = []
    file_list_dst = []
    # 使用 filedialog.askdirectory() 方法让用户选择一个文件夹
    folder = filedialog.askdirectory()
    # 如果用户选择了一个文件夹
    if folder:
        # 使用 os.listdir() 方法获取文件夹中的所有文件名
        files = os.listdir(folder)
        # 遍历文件名
        for file in files:
            if file.endswith('axl'):
                # 将文件路径添加到列表中
                file_list_axl.append(os.path.join(folder, file))
            if file.endswith('dst'):
                file_list_dst.append(os.path.join(folder, file))
    return file_list_axl, file_list_dst


file_list_axl, file_list_dst = open_folder()
toWrite = {'No': [], 'Name': [], 'result': []}
df = pandas.DataFrame(toWrite)
df.to_csv('output_oldType.csv', mode='w', index=False, header=True)
for index in range(0,len(file_list_axl),1):
    partsRow = file_list_axl[index].split('\\')
    # partsRow = partsRow[len(partsRow)-1].split('.')
    # partsRow = partsRow[0]
    partsRow = partsRow[len(partsRow)-1]
    parts = partsRow.split('_') # 记录文件名，后续使用
    dataH1 = pd.read_table(file_list_axl[index], delim_whitespace=True, header=None,
                           names=['r1','r2','r3','r4','r5','r6','r7','r8','r9','r10','r11','r12','r13','r14','r15','r16','r17','r18','r19','r20','r21','r22','r23','r24','r25','r26','r27','r28','r29','r30','r31','r32'])
    dataD1 = pd.read_table(file_list_dst[index], delim_whitespace=True, header=None,
                           names=['r1','r2','r3','r4','r5','r6','r7','r8','r9','r10','r11','r12','r13','r14','r15','r16','r17','r18','r19','r20','r21','r22','r23','r24','r25','r26','r27','r28','r29','r30','r31','r32'])
    # 对 dataH1 进行线性插值
    dataH1.interpolate(method='linear', axis=1, inplace=True)  # axis=1 表示沿着列进行插值

    # 对 dataH1 中非NaN的值进行转换
    dataH1 = dataH1.applymap(lambda x: 337.5 / x if x > 0.0 else x)
    # 对 dataD1 进行线性插值
    dataD1.interpolate(method='linear', axis=1, inplace=True)  # axis=1 表示沿着列进行插值

    # 寻找有效面积的边缘
    edges = np.zeros(dataH1.shape[0], dtype=int)
    for i in range(0, dataH1.shape[0]):
        max = -100
        for j in range(0, dataH1.shape[1]):
            if dataH1.iloc[i, j] > max:
                max = dataH1.iloc[i, j]
                edges[i] = j
                
    # 有效面积外归零
    for i in range(0, dataH1.shape[0]):
        for j in range(edges[i]+1, dataH1.shape[1]):
            dataH1.iloc[i, j] = 0

    # 使用矩阵乘法计算toadd和result
    def cal_curve(r, h):
        for i in range(len(r)):
            if h[i] == 0:
                r[i] = 0
            elif r[i] >= h[i]:
                r[i] = np.power(r[i], 2) * math.atan2((r[i] - h[i]), r[i])
            else:
                r[i] = np.power(r[i], 2) * math.atan2((h[i] - r[i]), h[i])
        return r


    result = 0
    toAdd_0 = 1 / 300 * np.pi * cal_curve(dataD1.iloc[0, :edges[0]], dataH1.iloc[0, :edges[0]])

    for i in range(1, 300):
        # 计算每个 toAdd
        toAdd = abs(1 / 300 * np.pi * cal_curve(dataD1.iloc[i, :edges[i]], dataH1.iloc[i, :edges[i]])
                - (1 / 300 * np.pi * cal_curve(dataD1.iloc[i - 1, :edges[i - 1]], dataH1.iloc[i - 1, :edges[i - 1]])))
        result += np.sum(toAdd)

    # 单位：平方方毫米*屈光度（mm2*D）
    toWrite = {'No': [parts[0]], 'Name': [parts[1] + parts[2] + '_' + parts[3] + '_' + parts[4]],
               'result': [abs(result)]}
    df = pandas.DataFrame(toWrite)
    df.to_csv('output_oldType.csv', mode='a', index=False, header=False)