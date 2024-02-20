# coding=utf-8
import pandas
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
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
                # 将文件添加到列表中
                file_list_axl.append(file)
            if file.endswith('dst'):
                file_list_dst.append(file)
    return file_list_axl,file_list_dst

file_list_axl,file_list_dst = open_folder()
file_list_axl_left = []
file_list_axl_right = []
file_list_dst_left = []
file_list_dst_right = []
for i in range(len(file_list_axl)):
    parts = file_list_axl[i].split('_')
    if parts[3] == 'Left':
        file_list_axl_left.append(file_list_axl[i])
    if parts[3] == "Right":
        file_list_axl_right.append(file_list_axl[i])
for i in range(len(file_list_dst)):
    parts = file_list_dst[i].split('_')
    if parts[3] == 'Left':
        file_list_dst_left.append(file_list_dst[i])
    if parts[3] == "Right":
        file_list_dst_right.append(file_list_dst[i])
toWrite = {'No': [], 'Name': [], 'Dif8mm': [], 'Dif6mm': [], 'Dif4o8mm': []}
df = pandas.DataFrame(toWrite)
df.to_csv('output_left.csv', mode='a', index=False, header=True)
df.to_csv('output_right.csv', mode='a', index=False, header=True)
for index in range(0,len(file_list_axl_left),2):
    parts = file_list_axl_left[index].split('_')# 记录文件名，后续使用
    dataH1 = pd.read_table(file_list_axl_left[index], delim_whitespace=True, header=None,
                           names=['r1','r2','r3','r4','r5','r6','r7','r8','r9','r10','r11','r12','r13','r14','r15','r16','r17','r18','r19','r20','r21','r22','r23','r24','r25','r26','r27','r28','r29','r30','r31','r32'])
    dataD1 = pd.read_table(file_list_dst_left[index], delim_whitespace=True, header=None,
                           names=['r1','r2','r3','r4','r5','r6','r7','r8','r9','r10','r11','r12','r13','r14','r15','r16','r17','r18','r19','r20','r21','r22','r23','r24','r25','r26','r27','r28','r29','r30','r31','r32'])
    dataH1 = dataH1.fillna(0)
    dataD1 = dataD1.fillna(0)

    dataH2 = pd.read_table(file_list_axl_left[index+1], delim_whitespace=True, header=None,
                           names=['r1','r2','r3','r4','r5','r6','r7','r8','r9','r10','r11','r12','r13','r14','r15','r16','r17','r18','r19','r20','r21','r22','r23','r24','r25','r26','r27','r28','r29','r30','r31','r32'])
    dataD2 = pd.read_table(file_list_dst_left[index+1], delim_whitespace=True, header=None,
                           names=['r1','r2','r3','r4','r5','r6','r7','r8','r9','r10','r11','r12','r13','r14','r15','r16','r17','r18','r19','r20','r21','r22','r23','r24','r25','r26','r27','r28','r29','r30','r31','r32'])
    dataH2 = dataH2.fillna(0)
    dataD2 = dataD2.fillna(0)

    # 计算屈光度
    for i in range(300):
        for j in range(32):
            if (dataH1.iloc[i,j]>0.0):
                dataH1.iloc[i,j] = 337.5 / dataH1.iloc[i,j]
            if (dataH2.iloc[i, j] > 0.0):
                dataH2.iloc[i, j] = 337.5 / dataH2.iloc[i, j]


    # 计算体积
    Vol8mm1 = 0.0
    Vol8mm2 = 0.0
    Vol6mm1 = 0.0
    Vol6mm2 = 0.0
    Vol4o8mm1 = 0.0
    Vol4o8mm2 = 0.0
    for i in range(300):
        toAdd = 1/300*math.pi*(math.pow(dataD1.iloc[i,0],2))*dataH1.iloc[i,0]*0.5
        Vol8mm1 += toAdd
        Vol6mm1 += toAdd
        Vol4o8mm1 += toAdd
        toAdd = 1/300*math.pi*(math.pow(dataD2.iloc[i,0],2))*dataH2.iloc[i,0]*0.5
        Vol8mm2 += toAdd
        Vol6mm2 += toAdd
        Vol4o8mm2 += toAdd
        for j in range(32):
            if(dataD1.iloc[i,j]<dataD1.iloc[i,0] or dataD1.iloc[i,j]>4.0):
                break
            toAdd = 1/300*math.pi*(math.pow(dataD1.iloc[i,j],2)-math.pow(dataD1.iloc[i,j-1],2))*(dataH1.iloc[i,j]+dataH1.iloc[i,j-1])*0.5
            Vol8mm1 += toAdd
            if (dataD1.iloc[i, j] <= 3.0):
                Vol6mm1 += toAdd
            if (dataD1.iloc[i, j] <= 2.4):
                Vol4o8mm1 += toAdd
        for j in range(32):
            if (dataD2.iloc[i, j] < dataD2.iloc[i, 0] or dataD2.iloc[i,j]>4.0):
                break
            toAdd = 1 / 300 * math.pi * (math.pow(dataD2.iloc[i, j], 2) - math.pow(dataD2.iloc[i, j - 1], 2)) * (dataH2.iloc[i, j] + dataH2.iloc[i, j - 1]) * 0.5
            Vol8mm2 += toAdd
            if (dataD2.iloc[i, j] <= 3.0):
                Vol6mm2 += toAdd
            if (dataD2.iloc[i, j] <= 2.4):
                Vol4o8mm2 += toAdd

    print(Vol8mm1)
    print(Vol8mm2)
    print(Vol8mm2-Vol8mm1) # 单位：平方方毫米*屈光度（mm2*D）
    print(Vol6mm1)
    print(Vol6mm2)
    print(Vol6mm2 - Vol6mm1)  # 单位：平方方毫米*屈光度（mm2*D）
    print(Vol4o8mm1)
    print(Vol4o8mm2)
    print(Vol4o8mm2 - Vol4o8mm1)  # 单位：平方方毫米*屈光度（mm2*D）
    toWrite = {'No': [parts[0]], 'Name': [parts[1]+parts[2]+'_'+parts[3]], 'Dif8mm': [Vol8mm2-Vol8mm1], 'Dif6mm': [Vol6mm2-Vol6mm1], 'Dif4o8mm': [Vol4o8mm2-Vol4o8mm1]}
    df = pandas.DataFrame(toWrite)
    df.to_csv('output_left.csv', mode='a', index=False, header=False)
    # 绘图
    x1 = np.zeros((300,32))
    y1 = np.zeros((300,32))
    x2 = np.zeros((300,32))
    y2 = np.zeros((300,32))
    for i in range(300):
        for j in range(32):
            x1[i,j] = dataD1.iloc[i,j]*math.cos(2*math.pi/300*i)
            y1[i,j] = dataD1.iloc[i,j]*math.sin(2*math.pi/300*i)
            x2[i,j] = dataD2.iloc[i,j]*math.cos(2*math.pi/300*i)
            y2[i,j] = dataD2.iloc[i,j]*math.sin(2*math.pi/300*i)
    x3 = (x1+x2)/2
    y3 = (y1+y2)/2
    # x_1 = np.linspace(np.min(x1),np.max(x1),1000)
    # y_1 = np.linspace(np.min(y1),np.max(y1),1000)
    # x_2 = np.linspace(np.min(x2),np.max(x2),1000)
    # y_2 = np.linspace(np.min(y2),np.max(y2),1000)
    # x_3 = np.divide(np.add(x_1,x_2),2)
    # y_3 = np.divide(np.add(y_1,y_2),2)
    # X1, Y1 = np.meshgrid(x_1,y_1)
    # X2, Y2 = np.meshgrid(x_2,y_2)
    # X3, Y3 = np.meshgrid(x_3,y_3)
    x_ = np.linspace(-6,6,300)
    y_ = np.linspace(-6,6,300)
    X1, Y1 = np.meshgrid(x_,y_)
    X2, Y2 = np.meshgrid(x_,y_)
    X3, Y3 = np.meshgrid(x_,y_)
    #拟合曲面
    interpolate_H1 = griddata((x1.flatten(),y1.flatten()),dataH1.values.flatten(),(X1,Y1),method="cubic")
    interpolate_H2 = griddata((x2.flatten(),y2.flatten()),dataH2.values.flatten(),(X2,Y2),method="cubic")
    # interpolate_H3 = griddata((X1.flatten(),Y1.flatten()),np.subtract(interpolate_H2.flatten(),interpolate_H1.flatten()),(X3,Y3),method="cubic")

    interpolate_H3 = np.subtract(interpolate_H2,interpolate_H1)
    for i in range(300):
        for j in range(300):
            if abs(interpolate_H3[i,j]) > 5:
                interpolate_H3[i,j] = 0

    # print(interpolate_H3.shape)
    # print(interpolate_H1)
    # print(interpolate_H3)
    fig = plt.figure()
    ax1 = fig.add_subplot(131)
    surf1 = ax1.imshow(interpolate_H1,cmap = "rainbow",vmin=35, vmax=45,extent=[min(X1.flatten()), max(X1.flatten()), min(Y1.flatten()), max(Y1.flatten())])
    fig.colorbar(surf1)
    # ax1.set_xlim([-5, 5])
    # ax1.set_ylim([-5, 5])
    # ax1.set_zlim([-2.5, 0])
    # ax1.set_xlabel('mm')
    # ax1.set_ylabel('mm')
    # ax1.set_zlabel('mm')
    ax2 = fig.add_subplot(132)
    surf2 = ax2.imshow(interpolate_H2,cmap="rainbow",vmin=35, vmax=45,extent=[min(X2.flatten()), max(X2.flatten()), min(Y2.flatten()), max(Y2.flatten())])
    fig.colorbar(surf2)
    # ax2.set_xlim([-5, 5])
    # ax2.set_ylim([-5, 5])
    # ax2.set_zlim([-2.5, 0])
    # ax2.set_xlabel('mm')
    # ax2.set_ylabel('mm')
    # ax2.set_zlabel('mm')

    ax3 = fig.add_subplot(133)
    avg = np.nanmean(interpolate_H3.flatten())
    sigma = np.nanvar(interpolate_H3.flatten())
    surf3 = ax3.imshow(interpolate_H3, cmap='rainbow',vmin=avg-2*sigma,vmax=avg+2*sigma, extent=[min(X3.flatten()), max(X3.flatten()), min(Y3.flatten()), max(Y3.flatten())], origin='lower')
    fig.colorbar(surf3)
    # parts我放前面去了，内容不变的
    name = '_'.join(parts[:4])
    plt.savefig('C:\\Users\\25428\\Desktop\\临床角膜地形图立体重构计算系统\\地形图\\pictures\\'+name+'.pdf')
    plt.close()
for index in range(0,len(file_list_axl_right),2):
    parts = file_list_axl_right[index].split('_')# 记录文件名，后续使用
    dataH1 = pd.read_table(file_list_axl_right[index], delim_whitespace=True, header=None,
                           names=['r1','r2','r3','r4','r5','r6','r7','r8','r9','r10','r11','r12','r13','r14','r15','r16','r17','r18','r19','r20','r21','r22','r23','r24','r25','r26','r27','r28','r29','r30','r31','r32'])
    dataD1 = pd.read_table(file_list_dst_right[index], delim_whitespace=True, header=None,
                           names=['r1','r2','r3','r4','r5','r6','r7','r8','r9','r10','r11','r12','r13','r14','r15','r16','r17','r18','r19','r20','r21','r22','r23','r24','r25','r26','r27','r28','r29','r30','r31','r32'])
    dataH1 = dataH1.fillna(0)
    dataD1 = dataD1.fillna(0)

    dataH2 = pd.read_table(file_list_axl_right[index+1], delim_whitespace=True, header=None,
                           names=['r1','r2','r3','r4','r5','r6','r7','r8','r9','r10','r11','r12','r13','r14','r15','r16','r17','r18','r19','r20','r21','r22','r23','r24','r25','r26','r27','r28','r29','r30','r31','r32'])
    dataD2 = pd.read_table(file_list_dst_right[index+1], delim_whitespace=True, header=None,
                           names=['r1','r2','r3','r4','r5','r6','r7','r8','r9','r10','r11','r12','r13','r14','r15','r16','r17','r18','r19','r20','r21','r22','r23','r24','r25','r26','r27','r28','r29','r30','r31','r32'])
    dataH2 = dataH2.fillna(0)
    dataD2 = dataD2.fillna(0)

    # 计算屈光度
    for i in range(300):
        for j in range(32):
            if (dataH1.iloc[i,j]>0.0):
                dataH1.iloc[i,j] = 337.5 / dataH1.iloc[i,j]
            if (dataH2.iloc[i, j] > 0.0):
                dataH2.iloc[i, j] = 337.5 / dataH2.iloc[i, j]


    # 计算体积
    Vol8mm1 = 0.0
    Vol8mm2 = 0.0
    Vol6mm1 = 0.0
    Vol6mm2 = 0.0
    Vol4o8mm1 = 0.0
    Vol4o8mm2 = 0.0
    for i in range(300):
        toAdd = 1/300*math.pi*(math.pow(dataD1.iloc[i,0],2))*dataH1.iloc[i,0]*0.5
        Vol8mm1 += toAdd
        Vol6mm1 += toAdd
        Vol4o8mm1 += toAdd
        toAdd = 1/300*math.pi*(math.pow(dataD2.iloc[i,0],2))*dataH2.iloc[i,0]*0.5
        Vol8mm2 += toAdd
        Vol6mm2 += toAdd
        Vol4o8mm2 += toAdd
        for j in range(32):
            if(dataD1.iloc[i,j]<dataD1.iloc[i,0] or dataD1.iloc[i,j]>4.0):
                break
            toAdd = 1/300*math.pi*(math.pow(dataD1.iloc[i,j],2)-math.pow(dataD1.iloc[i,j-1],2))*(dataH1.iloc[i,j]+dataH1.iloc[i,j-1])*0.5
            Vol8mm1 += toAdd
            if (dataD1.iloc[i, j] <= 3.0):
                Vol6mm1 += toAdd
            if (dataD1.iloc[i, j] <= 2.4):
                Vol4o8mm1 += toAdd
        for j in range(32):
            if (dataD2.iloc[i, j] < dataD2.iloc[i, 0] or dataD2.iloc[i,j]>4.0):
                break
            toAdd = 1 / 300 * math.pi * (math.pow(dataD2.iloc[i, j], 2) - math.pow(dataD2.iloc[i, j - 1], 2)) * (dataH2.iloc[i, j] + dataH2.iloc[i, j - 1]) * 0.5
            Vol8mm2 += toAdd
            if (dataD2.iloc[i, j] <= 3.0):
                Vol6mm2 += toAdd
            if (dataD2.iloc[i, j] <= 2.4):
                Vol4o8mm2 += toAdd

    print(Vol8mm1)
    print(Vol8mm2)
    print(Vol8mm2-Vol8mm1) # 单位：平方方毫米*屈光度（mm2*D）
    print(Vol6mm1)
    print(Vol6mm2)
    print(Vol6mm2 - Vol6mm1)  # 单位：平方方毫米*屈光度（mm2*D）
    print(Vol4o8mm1)
    print(Vol4o8mm2)
    print(Vol4o8mm2 - Vol4o8mm1)  # 单位：平方方毫米*屈光度（mm2*D）
    toWrite = {'No': [parts[0]], 'Name': [parts[1]+parts[2]+'_'+parts[3]], 'Dif8mm': [Vol8mm2-Vol8mm1], 'Dif6mm': [Vol6mm2-Vol6mm1], 'Dif4o8mm': [Vol4o8mm2-Vol4o8mm1]}
    df = pandas.DataFrame(toWrite)
    df.to_csv('output_right.csv', mode='a',index=False,header=False)
    # 绘图
    x1 = np.zeros((300,32))
    y1 = np.zeros((300,32))
    x2 = np.zeros((300,32))
    y2 = np.zeros((300,32))
    for i in range(300):
        for j in range(32):
            x1[i,j] = dataD1.iloc[i,j]*math.cos(2*math.pi/300*i)
            y1[i,j] = dataD1.iloc[i,j]*math.sin(2*math.pi/300*i)
            x2[i,j] = dataD2.iloc[i,j]*math.cos(2*math.pi/300*i)
            y2[i,j] = dataD2.iloc[i,j]*math.sin(2*math.pi/300*i)
    x3 = (x1+x2)/2
    y3 = (y1+y2)/2
    # x_1 = np.linspace(np.min(x1),np.max(x1),1000)
    # y_1 = np.linspace(np.min(y1),np.max(y1),1000)
    # x_2 = np.linspace(np.min(x2),np.max(x2),1000)
    # y_2 = np.linspace(np.min(y2),np.max(y2),1000)
    # x_3 = np.divide(np.add(x_1,x_2),2)
    # y_3 = np.divide(np.add(y_1,y_2),2)
    # X1, Y1 = np.meshgrid(x_1,y_1)
    # X2, Y2 = np.meshgrid(x_2,y_2)
    # X3, Y3 = np.meshgrid(x_3,y_3)
    x_ = np.linspace(-6,6,300)
    y_ = np.linspace(-6,6,300)
    X1, Y1 = np.meshgrid(x_,y_)
    X2, Y2 = np.meshgrid(x_,y_)
    X3, Y3 = np.meshgrid(x_,y_)
    print(X1.shape)
    #拟合曲面
    interpolate_H1 = griddata((x1.flatten(),y1.flatten()),dataH1.values.flatten(),(X1,Y1),method="cubic")
    interpolate_H2 = griddata((x2.flatten(),y2.flatten()),dataH2.values.flatten(),(X2,Y2),method="cubic")
    # interpolate_H3 = griddata((X1.flatten(),Y1.flatten()),np.subtract(interpolate_H2.flatten(),interpolate_H1.flatten()),(X3,Y3),method="cubic")

    interpolate_H3 = np.subtract(interpolate_H2,interpolate_H1)
    for i in range(300):
        for j in range(300):
            if abs(interpolate_H3[i,j]) > 5:
                interpolate_H3[i,j] = 0

    # print(interpolate_H3.shape)
    # print(interpolate_H1)
    # print(interpolate_H3)
    fig = plt.figure()
    ax1 = fig.add_subplot(131)
    surf1 = ax1.imshow(interpolate_H1,cmap = "rainbow",vmin=35, vmax=45,extent=[min(X1.flatten()), max(X1.flatten()), min(Y1.flatten()), max(Y1.flatten())])
    fig.colorbar(surf1)
    # ax1.set_xlim([-5, 5])
    # ax1.set_ylim([-5, 5])
    # ax1.set_zlim([-2.5, 0])
    # ax1.set_xlabel('mm')
    # ax1.set_ylabel('mm')
    # ax1.set_zlabel('mm')
    ax2 = fig.add_subplot(132)
    surf2 = ax2.imshow(interpolate_H2,cmap="rainbow",vmin=35, vmax=45,extent=[min(X2.flatten()), max(X2.flatten()), min(Y2.flatten()), max(Y2.flatten())])
    fig.colorbar(surf2)
    # ax2.set_xlim([-5, 5])
    # ax2.set_ylim([-5, 5])
    # ax2.set_zlim([-2.5, 0])
    # ax2.set_xlabel('mm')
    # ax2.set_ylabel('mm')
    # ax2.set_zlabel('mm')

    ax3 = fig.add_subplot(133)
    avg = np.nanmean(interpolate_H3.flatten())
    sigma = np.nanvar(interpolate_H3.flatten())
    surf3 = ax3.imshow(interpolate_H3, cmap='rainbow',vmin=avg-2*sigma,vmax=avg+2*sigma, extent=[min(X3.flatten()), max(X3.flatten()), min(Y3.flatten()), max(Y3.flatten())], origin='lower')
    fig.colorbar(surf3)
    # parts我放前面去了，内容不变的
    name = '_'.join(parts[:4])
    plt.savefig('C:\\Users\\25428\\Desktop\\临床角膜地形图立体重构计算系统\\地形图\\pictures\\'+name+'.pdf')
    plt.close()