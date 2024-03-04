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
df.to_csv('output_half_nihe.csv', mode='w', index=False, header=True)
for index in range(0,len(file_list_axl),1):
    partsRow = file_list_axl[index].split('\\')
    partsRow = partsRow[len(partsRow)-1].split('.')
    partsRow = partsRow[0]
    parts = partsRow.split('_')# 记录文件名，后续使用
    dataH1 = pd.read_table(file_list_axl[index], delim_whitespace=True, header=None,
                           names=['r1','r2','r3','r4','r5','r6','r7','r8','r9','r10','r11','r12','r13','r14','r15','r16','r17','r18','r19','r20','r21','r22','r23','r24','r25','r26','r27','r28','r29','r30','r31','r32'])
    dataD1 = pd.read_table(file_list_dst[index], delim_whitespace=True, header=None,
                           names=['r1','r2','r3','r4','r5','r6','r7','r8','r9','r10','r11','r12','r13','r14','r15','r16','r17','r18','r19','r20','r21','r22','r23','r24','r25','r26','r27','r28','r29','r30','r31','r32'])
    dataH1 = dataH1.fillna(0)
    dataD1 = dataD1.fillna(0)

    # # 计算屈光度并解决中间异常值
    # for i in range(300):
    #     for j in range(32):
    #         if 0 < j < 31 and dataH1.iloc[i, j] <= 0.0 < dataH1.iloc[i, j - 1] and dataH1.iloc[i, j + 1] > 0.0:
    #             dataH1.iloc[i, j] = 337.5 / ((dataH1.iloc[i, j - 1] + dataH1.iloc[i, j + 1]) * 0.5)
    #         elif dataH1.iloc[i, j] > 0.0:
    #             dataH1.iloc[i,j] = 337.5 / dataH1.iloc[i,j]
    #         if 0 < j < 31 and dataD1.iloc[i, j] <= 0.0 < dataD1.iloc[i, j - 1] and dataD1.iloc[i, j + 1] > 0.0:
    #             dataD1.iloc[i, j] = (dataD1.iloc[i, j - 1] + dataD1.iloc[i, j + 1]) * 0.5
    dataH1.interpolate(method='linear', axis=1, inplace=True)  # axis=1 表示沿着列进行插值

    # 对 dataH1 中非NaN的值进行转换
    dataH1 = dataH1.applymap(lambda x: 337.5 / x if x > 0.0 else x)

    # 对 dataD1 进行线性插值
    dataD1.interpolate(method='linear', axis=1, inplace=True)  # axis=1 表示沿着列进行插值
    # 找每个轴最大2个采样点
    dataChosen = np.zeros((2, 300))
    dataChosenH = np.zeros((2, 300))
    for j in range(32):
        for i in range(300):
            temp = dataH1.iloc[i, j]
            if temp > dataChosenH[0][i]:
                dataChosen[1][i] = dataChosen[0][i]
                dataChosen[0][i] = dataD1.iloc[i, j]
                dataChosenH[1][i] = dataChosenH[0][i]
                dataChosenH[0][i] = temp
            elif temp > dataChosenH[1][i]:
                dataChosen[1][i] = dataD1.iloc[i, j]
                dataChosenH[1][i] = temp

    def residuals(c, x, y):
        """计算每个数据点到圆的距离与半径之差的平方"""
        ri = np.sqrt((x - c[0]) ** 2 + (y - c[1]) ** 2)
        return ri - c[2]


    center = np.array([0, 0, 1])
    # 采样点极坐标转化为x，y
    dataChosenX = list()
    dataChosenY = list()
    for i in range(300):
        dataChosenX.append(dataChosen[0][i] * math.cos(math.radians(1.2 * i)))
        dataChosenX.append(dataChosen[1][i] * math.cos(math.radians(1.2 * i)))
        dataChosenY.append(dataChosen[0][i] * math.sin(math.radians(1.2 * i)))
        dataChosenY.append(dataChosen[1][i] * math.sin(math.radians(1.2 * i)))
    dataChosenX = np.array(dataChosenX)
    dataChosenY = np.array(dataChosenY)
    res = least_squares(residuals, center, args=(dataChosenX, dataChosenY))
    center = res.x

    # 绘制数据点和拟合的圆
    circle = plt.Circle((center[0], center[1]), center[2], color='blue', fill=False)
    fig, ax = plt.subplots()
    ax.add_artist(circle)
    plt.scatter(dataChosenX, dataChosenY, color='red')
    plt.axis('equal')
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    # plt.show()
    plt.savefig('savefigs/'+partsRow+'.png', bbox_inches = 'tight')
    plt.close()

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
    minDis = np.ones(32) * 2147483647
    minDisH = np.zeros(32)
    for i in range(300):
        for j in range(32):
            temp = dataD1.iloc[i,j]
            tempDis = math.pow(temp * math.cos(math.radians(1.2 * i)) - center[0], 2) + math.pow(temp * math.sin(math.radians(1.2 * i)) - center[1], 2)
            if tempDis < minDis[j]:
                minDis[j] = tempDis
                if dataH1.iloc[i, j] == 0:
                    minDisH[j] = minDisH[j-1]
                else:
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
    elif Point1Index == 31:
        Point2Index = 30
    else:
        if minDis[Point1Index-1] > minDis[Point1Index+1]:
            Point2Index = Point1Index + 1
        else :
            Point2Index = Point1Index - 1
    centerHPercent = minDisH[Point2Index] / minDisH[Point1Index] + minDisH[Point2Index]
    centerH = minDisH[Point2Index] * centerHPercent + minDisH[Point1Index] * (1 - centerHPercent)

    # 计算曲率总变化值
    result2mm = 0.0
    result2_4mm = 0.0
    result3mm = 0.0
    # for i in range(300):
    #     toAdd = 1/300*math.pi*(abs(math.pow(dataD1.iloc[i,0],2)))*(dataH1.iloc[i,0]-centerH)
    #     if area2mm.iloc[i, 0]:
    #         result2mm += toAdd
    #     if area2_4mm.iloc[i, 0]:
    #         result2_4mm += toAdd
    #     if area3mm.iloc[i, 0]:
    #         result3mm += toAdd
    #     for j in range(32):
    #         if dataD1.iloc[i, j] < dataD1.iloc[i, 0]:
    #             break
    #         toAdd = 1/300*math.pi*(abs(math.pow(dataD1.iloc[i,j],2)-math.pow(dataD1.iloc[i,j-1],2)))*(centerH-(dataH1.iloc[i, j] + dataH1.iloc[i, j-1])*0.5)
    #         if area2mm.iloc[i, j]:
    #             result2mm += toAdd
    #         if area2_4mm.iloc[i, j]:
    #             result2_4mm += toAdd
    #         if area3mm.iloc[i, j]:
    #             result3mm += toAdd
    # 使用 numpy 的广播功能进行向量化计算
    # 首先计算每个元素的 toAdd 值
    toAdd_0 = (1/300) * np.pi * (np.power(dataD1.iloc[:,0], 2)) * dataH1.iloc[:,0]
    result2mm = np.sum(toAdd_0 * area2mm.iloc[:, 0])
    result2_4mm = np.sum(toAdd_0 * area2_4mm.iloc[:, 0])
    result3mm = np.sum(toAdd_0 * area3mm.iloc[:, 0])

    # 对于 j > 0 的情况
    for j in range(1, 32):
        # 计算每个 toAdd
        toAdd_j = (1/300) * np.pi * (np.power(dataD1.iloc[:,j], 2) - np.power(dataD1.iloc[:,j-1], 2)) * (dataH1.iloc[:, j] + dataH1.iloc[:, j-1]) * 0.5
        # 根据条件进行累加
        mask = dataD1.iloc[:, j] >= dataD1.iloc[:, 0]
        result2mm += np.sum(toAdd_j * area2mm.iloc[:, j] * mask)
        result2_4mm += np.sum(toAdd_j * area2_4mm.iloc[:, j] * mask)
        result3mm += np.sum(toAdd_j * area3mm.iloc[:, j] * mask)
    # print(result2mm,result2_4mm,result3mm)
    # print(math.pi*4*centerH,math.pi*2.4*2.4*centerH,math.pi*9*centerH)
    result2mm = result2mm - math.pi*4*centerH
    result2_4mm = result2_4mm - math.pi*2.4*2.4*centerH
    result3mm = result3mm - math.pi*9*centerH

    # 单位：平方方毫米*屈光度（mm2*D）
    toWrite = {'No': [parts[0]], 'Name': [parts[1]+parts[2]+'_'+parts[3]+'_'+parts[4]], 'result2mm': [result2mm], 'result2.4mm': [result2_4mm], 'result3mm': [result3mm]}
    df = pandas.DataFrame(toWrite)
    df.to_csv('output_half_nihe.csv', mode='a', index=False, header=False)
