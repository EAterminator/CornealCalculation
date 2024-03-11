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


def open_folder():
    file_list_hgt = []
    file_list_dst = []
    # 使用 filedialog.askdirectory() 方法让用户选择一个文件夹
    folder = filedialog.askdirectory()
    # 如果用户选择了一个文件夹
    if folder:
        # 使用 os.listdir() 方法获取文件夹中的所有文件名
        files = os.listdir(folder)
        # 遍历文件名
        for file in files:
            if file.endswith('hgt'):
                # 将文件路径添加到列表中
                file_list_hgt.append(os.path.join(folder, file))
            if file.endswith('dst'):
                file_list_dst.append(os.path.join(folder, file))
    return file_list_hgt, file_list_dst


file_list_hgt, file_list_dst = open_folder()
toWrite = {'No': [], 'Name': [], 'result': []}
df = pandas.DataFrame(toWrite)
df.to_csv('output_oldType.csv', mode='w', index=False, header=True)
for index in range(0,len(file_list_hgt),1):
    partsRow = file_list_hgt[index].split('\\')
    partsRow = partsRow[len(partsRow)-1].split('.')
    partsRow = partsRow[0]
    parts = partsRow.split('_')# 记录文件名，后续使用
    dataH1 = pd.read_table(file_list_hgt[index], delim_whitespace=True, header=None,
                           names=['r1','r2','r3','r4','r5','r6','r7','r8','r9','r10','r11','r12','r13','r14','r15','r16','r17','r18','r19','r20','r21','r22','r23','r24','r25','r26','r27','r28','r29','r30','r31','r32'])
    dataD1 = pd.read_table(file_list_dst[index], delim_whitespace=True, header=None,
                           names=['r1','r2','r3','r4','r5','r6','r7','r8','r9','r10','r11','r12','r13','r14','r15','r16','r17','r18','r19','r20','r21','r22','r23','r24','r25','r26','r27','r28','r29','r30','r31','r32'])

    # 对 dataH1 进行线性插值
    dataH1.interpolate(method='linear', axis=1, inplace=True)  # axis=1 表示沿着列进行插值
    # 对 dataD1 进行线性插值
    dataD1.interpolate(method='linear', axis=1, inplace=True)  # axis=1 表示沿着列进行插值

    # 寻找有效面积的边缘
    edges = np.zeros(dataH1.shape[1])
    for i in range(0, dataH1.shape[1]):
        max = 0.0
        for j in range(0, dataH1.shape[0]):
            if dataH1.iloc[j, i] > max:
                max = dataH1.iloc[j, i]
                edges[i] = j

    # 有效面积外归零
    # 使用矩阵乘法计算toadd和result