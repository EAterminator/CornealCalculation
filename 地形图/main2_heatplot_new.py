# coding=utf-8
import pandas
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
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
df.to_csv('output.csv', mode='a', index=False, header=True)
for index in range(0,len(file_list_axl),1):
    parts = file_list_axl[index].split('_')# 记录文件名，后续使用
    dataH1 = pd.read_table(file_list_axl[index], delim_whitespace=True, header=None,
                           names=['r1','r2','r3','r4','r5','r6','r7','r8','r9','r10','r11','r12','r13','r14','r15','r16','r17','r18','r19','r20','r21','r22','r23','r24','r25','r26','r27','r28','r29','r30','r31','r32'])
    dataD1 = pd.read_table(file_list_dst[index], delim_whitespace=True, header=None,
                           names=['r1','r2','r3','r4','r5','r6','r7','r8','r9','r10','r11','r12','r13','r14','r15','r16','r17','r18','r19','r20','r21','r22','r23','r24','r25','r26','r27','r28','r29','r30','r31','r32'])
    dataH1 = dataH1.fillna(0)
    dataD1 = dataD1.fillna(0)

    # 计算屈光度并解决中间异常值
    for i in range(300):
        for j in range(32):
            if 0 < j < 31 and dataH1.iloc[i, j] <= 0.0 < dataH1.iloc[i, j - 1] and dataH1.iloc[i, j + 1] > 0.0:
                dataH1.iloc[i, j] = 337.5 / ((dataH1.iloc[i, j - 1] + dataH1.iloc[i, j + 1]) * 0.5)
                dataD1.iloc[i, j] = (dataD1.iloc[i, j - 1] + dataD1.iloc[i, j + 1]) * 0.5
            elif dataH1.iloc[i, j] > 0.0:
                dataH1.iloc[i,j] = 337.5 / dataH1.iloc[i,j]

    # 找每个轴最大2个采样点
    dataChosen = [list()] * 300
    for i in range(300):
        dataChosen[i].append(0)
        dataChosen[i].append(0)
        for j in range(32):
            temp = dataD1.iloc[i,j]
            if temp > dataChosen[i][0]:
                dataChosen[i][1] = dataChosen[i][0]
                dataChosen[i][0] = temp
            elif temp > dataChosen[i][1]:
                dataChosen[i][1] = temp


    # 采样点极坐标转化为x，y
    dataChosenX = [list()] * 3
    dataChosenR = list()
    for i in range(300):
        dataChosenX[0].append(dataChosen[i][0] * math.cos(math.radians(11.25 * i)))
        dataChosenX[0].append(dataChosen[i][1] * math.cos(math.radians(11.25 * i)))
        dataChosenX[1].append(dataChosen[i][0] * math.sin(math.radians(11.25 * i)))
        dataChosenX[1].append(dataChosen[i][1] * math.sin(math.radians(11.25 * i)))
        dataChosenX[2].append(1)
        dataChosenX[2].append(1)
        dataChosenR.append(dataChosen[i][0])
        dataChosenR.append(dataChosen[i][1])
    dataChosenX = np.array(dataChosenX)
    dataChosenR = np.array(dataChosenR)
    center = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(dataChosenX), dataChosenX)), np.transpose(dataChosenX)), dataChosenR)
    print(center[0])
    print(center[1])

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
    minDis = list()
    minDisH = list()
    for i in range(32):
        minDis.append(2147483647)
        minDisH.append(0)
    for i in range(300):
        for j in range(32):
            temp = dataD1.iloc[i,j]
            tempDis = math.pow(temp * math.cos(math.radians(11.25 * i)) - center[0], 2) + math.pow(temp * math.sin(math.radians(11.25 * i)) - center[1], 2)
            if tempDis<minDis[j]:
                minDis[j] = tempDis
                minDisH[j] = dataH1.iloc[i, j]
            area2mm.iloc[i, j] = tempDis < 4.0
            area2_4mm.iloc[i, j] = tempDis < 5.76
            area3mm.iloc[i, j] = tempDis < 9.0
    Point1Dis = 2147483647
    Point2Dis = 2147483647
    Point1Index = -1
    Point2Index = -1
    for i in range(32):
        if minDis[i] < Point1Dis:
            mostMinDis = minDis[i]
            Point1Index = i
    if Point1Index == 0:
        Point2Index = 1
    else:
        if minDis[Point1Index-1] > minDis[Point1Index+1]:
            Point2Index = Point1Index - 1
        else :
            Point2Index = Point1Index - 1
    centerHPercent = minDisH[Point2Index] / minDisH[Point1Index] + minDisH[Point2Index]
    centerH = minDisH[Point2Index] * centerHPercent + minDisH[Point1Index] * (1 - centerHPercent)

    # 计算曲率总变化值
    result2mm = 0.0
    result2_4mm = 0.0
    result3mm = 0.0
    for i in range(300):
        toAdd = 1/300*math.pi*(abs(math.pow(dataD1.iloc[i,0],2)))*(dataH1.iloc[i,0]-centerH)
        if area2mm.iloc[i, 0]:
            result2mm += toAdd
        if area2_4mm.iloc[i, 0]:
            result2_4mm += toAdd
        if area3mm.iloc[i, 0]:
            result3mm += toAdd
        for j in range(32):
            if dataD1.iloc[i, j] < dataD1.iloc[i, 0]:
                break
            toAdd = 1/300*math.pi*(abs(math.pow(dataD1.iloc[i,j],2)-math.pow(dataD1.iloc[i,j-1],2)))*(centerH-(dataH1.iloc[i, j] + dataH1.iloc[i, j-1])*0.5)
            if area2mm.iloc[i, j]:
                result2mm += toAdd
            if area2_4mm.iloc[i, j]:
                result2_4mm += toAdd
            if area3mm.iloc[i, j]:
                result3mm += toAdd

    # 单位：平方方毫米*屈光度（mm2*D）
    toWrite = {'No': [parts[0]], 'Name': [parts[1]+parts[2]+'_'+parts[3]], 'result2mm': [result2mm], 'result2.4mm': [result2_4mm], 'result3mm': [result3mm]}
    df = pandas.DataFrame(toWrite)
    df.to_csv('output.csv', mode='a', index=False, header=False)
