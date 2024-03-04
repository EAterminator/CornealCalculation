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
toWrite = {'No': [], 'Name': [], 'result2mm': [], 'result2.4mm': [], 'result3mm': []}
df = pandas.DataFrame(toWrite)
df.to_csv('output.csv', mode='w', index=False, header=True)
for index in range(0,len(file_list_axl),1):
    partsRow = file_list_axl[index].split('\\')
    partsRow = partsRow[len(partsRow)-1].split('.')
    partsRow = partsRow[0]
    parts = partsRow.split('_')# 记录文件名，后续使用
    dataH1 = pd.read_table(file_list_axl[index], delim_whitespace=True, header=None,
                           names=['r1','r2','r3','r4','r5','r6','r7','r8','r9','r10','r11','r12','r13','r14','r15','r16','r17','r18','r19','r20','r21','r22','r23','r24','r25','r26','r27','r28','r29','r30','r31','r32'])
    dataD1 = pd.read_table(file_list_dst[index], delim_whitespace=True, header=None,
                           names=['r1','r2','r3','r4','r5','r6','r7','r8','r9','r10','r11','r12','r13','r14','r15','r16','r17','r18','r19','r20','r21','r22','r23','r24','r25','r26','r27','r28','r29','r30','r31','r32'])
    # dataH1 = dataH1.fillna(0)
    # dataD1 = dataD1.fillna(0)

    # # 计算屈光度并解决中间异常值
    # for i in range(300):
    #     for j in range(32):
    #         if 0 < j < 31 and dataH1.iloc[i, j] <= 0.0 < dataH1.iloc[i, j - 1] and dataH1.iloc[i, j + 1] > 0.0:
    #             dataH1.iloc[i, j] = 337.5 / ((dataH1.iloc[i, j - 1] + dataH1.iloc[i, j + 1]) * 0.5)
    #         elif dataH1.iloc[i, j] > 0.0:
    #             dataH1.iloc[i,j] = 337.5 / dataH1.iloc[i,j]
    #         if 0 < j < 31 and dataD1.iloc[i, j] <= 0.0 < dataD1.iloc[i, j - 1] and dataD1.iloc[i, j + 1] > 0.0:
    #             dataD1.iloc[i, j] = (dataD1.iloc[i, j - 1] + dataD1.iloc[i, j + 1]) * 0.5
    # 对 dataH1 进行线性插值
    dataH1.interpolate(method='linear', axis=1, inplace=True)  # axis=1 表示沿着列进行插值

    # 对 dataH1 中非NaN的值进行转换
    dataH1 = dataH1.applymap(lambda x: 337.5 / x if x > 0.0 else x)

    # 对 dataD1 进行线性插值
    dataD1.interpolate(method='linear', axis=1, inplace=True)  # axis=1 表示沿着列进行插值

    centerH = 0.0
    # 寻找符合要求面积点集，及拟合中心点曲率
    area2mm = pd.DataFrame(
        columns=['r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7', 'r8', 'r9', 'r10', 'r11', 'r12', 'r13', 'r14', 'r15', 'r16',
                 'r17', 'r18', 'r19', 'r20', 'r21', 'r22', 'r23', 'r24', 'r25', 'r26', 'r27', 'r28', 'r29', 'r30',
                 'r31', 'r32'], index=range(0, 300, 1))
    area2_4mm = pd.DataFrame(
        columns=['r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7', 'r8', 'r9', 'r10', 'r11', 'r12', 'r13', 'r14', 'r15', 'r16',
                 'r17', 'r18', 'r19', 'r20', 'r21', 'r22', 'r23', 'r24', 'r25', 'r26', 'r27', 'r28', 'r29', 'r30',
                 'r31', 'r32'], index=range(0, 300, 1))
    area3mm = pd.DataFrame(
        columns=['r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7', 'r8', 'r9', 'r10', 'r11', 'r12', 'r13', 'r14', 'r15', 'r16',
                 'r17', 'r18', 'r19', 'r20', 'r21', 'r22', 'r23', 'r24', 'r25', 'r26', 'r27', 'r28', 'r29', 'r30',
                 'r31', 'r32'], index=range(0, 300, 1))
    for i in range(300):
        centerH += dataH1.iloc[i, 0]/dataD1.iloc[i,0]
        for j in range(32):
            temp = dataD1.iloc[i,j]
            tempDis = math.pow(temp * math.cos(math.radians(1.2 * i)), 2) + math.pow(temp * math.sin(math.radians(1.2 * i)), 2)
            area2mm.iloc[i, j] = tempDis <= 4.0
            area2_4mm.iloc[i, j] = tempDis <= 5.76
            area3mm.iloc[i, j] = tempDis <= 9.0
    centerH = centerH / np.sum(1/dataD1.iloc[:,0])
    print(centerH)
    # print(dataH1)
    # print(dataD1)
    # 计算曲率总变化值
    result2mm = 0.0
    result2_4mm = 0.0
    result3mm = 0.0
    # for i in range(300):
    #     toAdd = 1/300*math.pi*(math.pow(dataD1.iloc[i,0],2))*dataH1.iloc[i,0]
    #     if area2mm.iloc[i, 0]:
    #         result2mm += toAdd
    #     if area2_4mm.iloc[i, 0]:
    #         result2_4mm += toAdd
    #     if area3mm.iloc[i, 0]:
    #         result3mm += toAdd
    #     for j in range(32):
    #         if dataD1.iloc[i, j] < dataD1.iloc[i, 0]:
    #             break
    #         toAdd = (1/300)*math.pi*(abs(math.pow(dataD1.iloc[i,j],2)-math.pow(dataD1.iloc[i,j-1],2)))*(dataH1.iloc[i, j] + dataH1.iloc[i, j-1])*0.5
    #         if area2mm.iloc[i, j]:
    #             result2mm += toAdd
    #         if area2_4mm.iloc[i, j]:
    #             result2_4mm += toAdd
    #         if area3mm.iloc[i, j]:
    #             result3mm += toAdd
    centerH_tensor = np.ones((300,32))
    centerH_tensor = centerH_tensor*centerH
    # 使用 numpy 的广播功能进行向量化计算
    # 首先计算每个元素的 toAdd 值
    toAdd_0 = (1/300) * np.pi * (np.power(dataD1.iloc[:,0], 2)) * (dataH1.iloc[:,0] - centerH_tensor[:,0])
    result2mm = np.sum(toAdd_0 * area2mm.iloc[:, 0])
    result2_4mm = np.sum(toAdd_0 * area2_4mm.iloc[:, 0])
    result3mm = np.sum(toAdd_0 * area3mm.iloc[:, 0])

    # 对于 j > 0 的情况
    for j in range(1, 32):
        # 计算每个 toAdd
        toAdd_j = (1/300) * np.pi * (np.power(dataD1.iloc[:,j], 2) - np.power(dataD1.iloc[:,j-1], 2)) * ((dataH1.iloc[:, j] + dataH1.iloc[:, j-1]) * 0.5 - centerH_tensor[:,j])
        # 根据条件进行累加
        mask = dataD1.iloc[:, j] >= dataD1.iloc[:, 0]
        result2mm += np.sum(toAdd_j * area2mm.iloc[:, j] * mask)
        result2_4mm += np.sum(toAdd_j * area2_4mm.iloc[:, j] * mask)
        result3mm += np.sum(toAdd_j * area3mm.iloc[:, j] * mask)
    # print(result2mm,result2_4mm,result3mm)
    # print(math.pi*4*centerH,math.pi*2.4*2.4*centerH,math.pi*9*centerH)


    # 单位：平方方毫米*屈光度（mm2*D）
    toWrite = {'No': [parts[0]], 'Name': [parts[1]+parts[2]+'_'+parts[3]+'_'+parts[4]], 'result2mm': [result2mm], 'result2.4mm': [result2_4mm], 'result3mm': [result3mm]}
    df = pandas.DataFrame(toWrite)
    df.to_csv('output.csv', mode='a', index=False, header=False)
