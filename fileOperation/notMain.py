# coding=utf-8
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import tkinter as tk
from tkinter import filedialog
import os

dataH1 = pd.read_table('02_常_会莹2-5校准版4.2mm_Left_2022-08-17_08-59-11.hgt', delim_whitespace=True, header=None,names=['r1','r2','r3','r4','r5','r6','r7','r8','r9','r10','r11','r12','r13','r14','r15','r16','r17','r18','r19','r20','r21','r22','r23','r24','r25','r26','r27','r28','r29','r30','r31','r32'])
dataD1 = pd.read_table('02_常_会莹2-5校准版4.2mm_Left_2022-08-17_08-59-11.dst', delim_whitespace=True, header=None,names=['r1','r2','r3','r4','r5','r6','r7','r8','r9','r10','r11','r12','r13','r14','r15','r16','r17','r18','r19','r20','r21','r22','r23','r24','r25','r26','r27','r28','r29','r30','r31','r32'])
dataH1 = dataH1.fillna(0)
dataD1 = dataD1.fillna(0)
dataH2 = pd.read_table('01_常_会莹2-5校准版4.2mm_Left_2023-02-28_09-13-09.hgt', delim_whitespace=True, header=None,names=['r1','r2','r3','r4','r5','r6','r7','r8','r9','r10','r11','r12','r13','r14','r15','r16','r17','r18','r19','r20','r21','r22','r23','r24','r25','r26','r27','r28','r29','r30','r31','r32'])
dataD2 = pd.read_table('01_常_会莹2-5校准版4.2mm_Left_2023-02-28_09-13-09.dst', delim_whitespace=True, header=None,names=['r1','r2','r3','r4','r5','r6','r7','r8','r9','r10','r11','r12','r13','r14','r15','r16','r17','r18','r19','r20','r21','r22','r23','r24','r25','r26','r27','r28','r29','r30','r31','r32'])
dataH2 = dataH2.fillna(0)
dataD2 = dataD2.fillna(0)

