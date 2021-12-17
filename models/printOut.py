#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 23:54:44 2019

@author: haoxiangyang
"""

import pickle
import sys
from os import path
import csv
import datetime

from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdates
import matplotlib.cbook as cbook
from matplotlib import rc
import matplotlib.patches as patches
import matplotlib.colors as pltcolors
import numpy as np
import textwrap

import pandas as pd
import time
import argparse
import calendar as py_cal
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdates
import matplotlib.cbook as cbook
from matplotlib import rc
import matplotlib.patches as patches
import matplotlib.colors as pltcolors
from collections import defaultdict
import copy

#%%
sys.path.append(path.abspath('/Users/haoxiangyang/Desktop/Git/daniel_Diesel'))  # home
sys.path.append(path.abspath('/home/haoxiang/daniel_Diesel'))  # Crunch

# load the alternative comparison
outDict = {}
outDict["surgeD"] = []
outDict["nomD"] = []
outDict["totalD"] = []
dataAlt = pickle.load(open("/Users/haoxiangyang/Desktop/Git/daniel_Diesel/output/Test_FR50_H96_F72_R24_N24_REAL.p","rb"))
surgeT = 0
nomT = 0
Tlen = 456
for i in range(Tlen):
    surgeT += dataAlt[1][0][i]
    nomT += dataAlt[1][1][i]
    outDict["surgeD"].append([dataAlt[4][i],surgeT])
    outDict["nomD"].append([dataAlt[4][i],nomT])
    outDict["totalD"].append([dataAlt[4][i],surgeT + nomT])
for dType in ["GEFS","NDFD","GAVG","REAL"]:
    dataAlt = pickle.load(open("/Users/haoxiangyang/Desktop/Git/daniel_Diesel/output/Test_FR50_H96_F72_R24_N24_{}.p".format(dType),"rb"))
    surgeT = 0
    nomT = 0
    for i in range(Tlen):
        surgeT += dataAlt[0][0][i]
        outDict["surgeD"][i].append(surgeT)
        nomT += dataAlt[0][1][i]
        outDict["nomD"][i].append(nomT)
        outDict["totalD"][i].append(surgeT + nomT)
    print("---------------- {} -------------------".format(dType))
    print(" & ", round(sum(dataAlt[0][0])/1000,1), " & ", round(sum(dataAlt[0][1])/1000,1), " & ", round((sum(dataAlt[0][0]) + sum(dataAlt[0][1]))/1000,1),\
          " & ", round(sum(dataAlt[2][0]),1), " & ", round(sum(dataAlt[2][1]),1), " & ", round(sum(dataAlt[2][1])+sum(dataAlt[2][0]),1))

title = ["Datetime","TotalD","GEFS","NDFD","GAVG","REAL"]
for i in ["surge","nom","total"]:
    printAdd = "/Users/haoxiangyang/Desktop/Git/daniel_Diesel/output/{}24.csv".format(i)
    fo = open(printAdd,'w',newline = '')
    csvWriter = csv.writer(fo,dialect = 'excel')
    csvWriter.writerow(title)
    csvWriter.writerows(outDict["{}D".format(i)])
    fo.close()
    
# print alternative comparion for R = 12, N = 12
outDict = {}
outDict["surgeD"] = []
outDict["nomD"] = []
outDict["totalD"] = []
dataAlt = pickle.load(open("/Users/haoxiangyang/Desktop/Git/daniel_Diesel/output/Test_FR50_H96_F72_R12_N12_REAL.p","rb"))
surgeT = 0
nomT = 0
Tlen = 444
for i in range(Tlen):
    surgeT += dataAlt[1][0][i]
    nomT += dataAlt[1][1][i]
    outDict["surgeD"].append([dataAlt[4][i],surgeT])
    outDict["nomD"].append([dataAlt[4][i],nomT])
    outDict["totalD"].append([dataAlt[4][i],surgeT + nomT])
for dType in ["GEFS","NDFD","GAVG","REAL"]:
    dataAlt = pickle.load(open("/Users/haoxiangyang/Desktop/Git/daniel_Diesel/output/Test_FR50_H96_F72_R12_N12_{}.p".format(dType),"rb"))
    surgeT = 0
    nomT = 0
    for i in range(Tlen):
        surgeT += dataAlt[0][0][i]
        outDict["surgeD"][i].append(surgeT)
        nomT += dataAlt[0][1][i]
        outDict["nomD"][i].append(nomT)
        outDict["totalD"][i].append(surgeT + nomT)
    print("---------------- {} -------------------".format(dType))
    print(" & ", round(sum(dataAlt[0][0])/1000,1), " & ", round(sum(dataAlt[0][1])/1000,1), " & ", round((sum(dataAlt[0][0]) + sum(dataAlt[0][1]))/1000,1),\
          " & ", round(sum(dataAlt[2][0]),1), " & ", round(sum(dataAlt[2][1]),1), " & ", round(sum(dataAlt[2][1])+sum(dataAlt[2][0]),1))

title = ["Datetime","TotalD","GEFS","NDFD","GAVG","REAL"]
for i in ["surge","nom","total"]:
    printAdd = "/Users/haoxiangyang/Desktop/Git/daniel_Diesel/output/{}12.csv".format(i)
    fo = open(printAdd,'w',newline = '')
    csvWriter = csv.writer(fo,dialect = 'excel')
    csvWriter.writerow(title)
    csvWriter.writerows(outDict["{}D".format(i)])
    fo.close()
    
    
print("---------------- Best -------------------")
dataPPI = pickle.load(open("/Users/haoxiangyang/Desktop/Git/daniel_Diesel/output/Test_FR50_H528_F72_R24_N24_REAL.p","rb"))
print(" & ", round(sum(dataPPI[0][0])/1000,1), " & ", round(sum(dataPPI[0][1])/1000,1), " & ", round((sum(dataPPI[0][0]) + sum(dataPPI[0][1]))/1000,1),\
      " & ", round(sum(dataPPI[2][0]),1), " & ", round(sum(dataPPI[2][1]),1), " & ", round(sum(dataPPI[2][1])+sum(dataPPI[2][0]),1))

# load the S comparison
for i in [24,48,72]:
    dataF = pickle.load(open("/Users/haoxiangyang/Desktop/Git/daniel_Diesel/output/Test_FR50_H96_F{}_R24_N24_GEFS.p".format(i),"rb"))
    print("---------------- {} -------------------".format(i))
    print(" & ", round(sum(dataF[0][0])/1000,1), " & ", round(sum(dataF[0][1])/1000,1), " & ", round((sum(dataF[0][0]) + sum(dataF[0][1]))/1000,1),\
      " & ", round(sum(dataF[2][0]),1), " & ", round(sum(dataF[2][1]),1), " & ", round(sum(dataF[2][1])+sum(dataF[2][0]),1))

# load the F comparison
for i in [96,120,144]:
    dataS = pickle.load(open("/Users/haoxiangyang/Desktop/Git/daniel_Diesel/output/Test_FR50_H{}_F{}_R24_N24_GEFS.p".format(i,i-72),"rb"))
    print("---------------- {} -------------------".format(i))
    print(" & ", round(sum(dataS[0][0])/1000,1), " & ", round(sum(dataS[0][1])/1000,1), " & ", round((sum(dataS[0][0]) + sum(dataS[0][1]))/1000,1),\
      " & ", round(sum(dataS[2][0]),1), " & ", round(sum(dataS[2][1]),1), " & ", round(sum(dataS[2][1])+sum(dataS[2][0]),1))

# load the H comparison
for i in [96,72,48]:
    dataH = pickle.load(open("/Users/haoxiangyang/Desktop/Git/daniel_Diesel/output/Test_FR50_H{}_F24_R24_N24_GEFS.p".format(i),"rb"))
    print("---------------- {} -------------------".format(i))
    print(" & ", round(sum(dataH[0][0])/1000,1), " & ", round(sum(dataH[0][1])/1000,1), " & ", round((sum(dataH[0][0]) + sum(dataH[0][1]))/1000,1),\
      " & ", round(sum(dataH[2][0]),1), " & ", round(sum(dataH[2][1]),1), " & ", round(sum(dataH[2][1])+sum(dataH[2][0]),1))

# load the R comparison
for i in [24,12,6]:
    dataR = pickle.load(open("/Users/haoxiangyang/Desktop/Git/daniel_Diesel/output/Test_FR50_H96_F24_R{}_N24_GEFS.p".format(i),"rb"))
    print("---------------- {} -------------------".format(i))
    print(" & ", round(sum(dataR[0][0])/1000,1), " & ", round(sum(dataR[0][1])/1000,1), " & ", round((sum(dataR[0][0]) + sum(dataR[0][1]))/1000,1),\
      " & ", round(sum(dataR[2][0]),1), " & ", round(sum(dataR[2][1]),1), " & ", round(sum(dataR[2][1])+sum(dataR[2][0]),1))

# load the N comparison
for i in [24,12,6]:
    dataN = pickle.load(open("/Users/haoxiangyang/Desktop/Git/daniel_Diesel/output/Test_FR50_H96_F24_R24_N{}_GEFS.p".format(i),"rb"))
    print("---------------- {} -------------------".format(i))
    print(" & ", round(sum(dataN[0][0])/1000,1), " & ", round(sum(dataN[0][1])/1000,1), " & ", round((sum(dataN[0][0]) + sum(dataN[0][1]))/1000,1),\
      " & ", round(sum(dataN[2][0]),1), " & ", round(sum(dataN[2][1]),1), " & ", round(sum(dataN[2][1])+sum(dataN[2][0]),1))

# load the alpha comparison
for i in [100,50,20]:
    dataalpha = pickle.load(open("/Users/haoxiangyang/Desktop/Git/daniel_Diesel/output/Test_FR{}_H96_F24_R24_N24_GEFS.p".format(i),"rb"))
    print("---------------- {} -------------------".format(i))
    print(" & ", round(sum(dataalpha[0][0])/1000,1), " & ", round(sum(dataalpha[0][1])/1000,1), " & ", round((sum(dataalpha[0][0]) + sum(dataalpha[0][1]))/1000,1),\
      " & ", round(sum(dataalpha[2][0]),1), " & ", round(sum(dataalpha[2][1]),1), " & ", round(sum(dataalpha[2][1])+sum(dataalpha[2][0]),1))

#%%
# load the alternative comparison
outDict = {}
outDict["surgeD"] = []
outDict["nomD"] = []
outDict["totalD"] = []
dataAlt = pickle.load(open("/Users/haoxiangyang/Desktop/Git/DFSC_FLORIDA/output/Test_FR50_H96_F72_R24_N24_REAL.p","rb"))
surgeT = 0
nomT = 0
Tlen = 456
for i in range(Tlen):
    surgeT += dataAlt[1][0][i]
    nomT += dataAlt[1][1][i]
    outDict["surgeD"].append([dataAlt[4][i],surgeT])
    outDict["nomD"].append([dataAlt[4][i],nomT])
    outDict["totalD"].append([dataAlt[4][i],surgeT + nomT])
for dType in ["GEFS","NDFD","GAVG","REAL"]:
    dataAlt = pickle.load(open("/Users/haoxiangyang/Desktop/Git/DFSC_FLORIDA/output/Test_FR50_H96_F72_R24_N24_{}.p".format(dType),"rb"))
    surgeT = 0
    nomT = 0
    for i in range(Tlen):
        surgeT += dataAlt[0][0][i]
        outDict["surgeD"][i].append(surgeT)
        nomT += dataAlt[0][1][i]
        outDict["nomD"][i].append(nomT)
        outDict["totalD"][i].append(surgeT + nomT)
    print("---------------- {} -------------------".format(dType))
    print(" & ", round(sum(dataAlt[0][0])/1000,1), " & ", round(sum(dataAlt[0][1])/1000,1), " & ", round((sum(dataAlt[0][0]) + sum(dataAlt[0][1]))/1000,1),\
          " & ", round(sum(dataAlt[2][0]),1), " & ", round(sum(dataAlt[2][1]),1), " & ", round(sum(dataAlt[2][1])+sum(dataAlt[2][0]),1))

title = ["Datetime","TotalD","GEFS","NDFD","GAVG","REAL"]
for i in ["surge","nom","total"]:
    printAdd = "/Users/haoxiangyang/Desktop/Git/DFSC_FLORIDA/output/{}24.csv".format(i)
    fo = open(printAdd,'w',newline = '')
    csvWriter = csv.writer(fo,dialect = 'excel')
    csvWriter.writerow(title)
    csvWriter.writerows(outDict["{}D".format(i)])
    fo.close()
#%%
# load the alternative comparison convex fit
outDict = {}
outDict["surgeD"] = []
outDict["nomD"] = []
outDict["totalD"] = []
dataAlt = pickle.load(open("/Users/haoxiangyang/Desktop/Git/DFSC_FLORIDA/output/Test_FR50_H96_F72_R24_N24_REAL.p","rb"))
surgeT = 0
nomT = 0
Tlen = 456
for i in range(Tlen):
    surgeT += dataAlt[1][0][i]
    nomT += dataAlt[1][1][i]
    outDict["surgeD"].append([dataAlt[4][i],surgeT])
    outDict["nomD"].append([dataAlt[4][i],nomT])
    outDict["totalD"].append([dataAlt[4][i],surgeT + nomT])
for dType in ["GEFS","NDFD","GAVG"]:
    dataAlt = pickle.load(open("/Users/haoxiangyang/Desktop/Git/DFSC_FLORIDA/output/Test_FR50_H96_F72_R24_N24_{}_c.p".format(dType),"rb"))
    surgeT = 0
    nomT = 0
    for i in range(Tlen):
        surgeT += dataAlt[0][0][i]
        outDict["surgeD"][i].append(surgeT)
        nomT += dataAlt[0][1][i]
        outDict["nomD"][i].append(nomT)
        outDict["totalD"][i].append(surgeT + nomT)
    print("---------------- {} -------------------".format(dType))
    print(" & ", round(sum(dataAlt[0][0])/1000,1), " & ", round(sum(dataAlt[0][1])/1000,1), " & ", round((sum(dataAlt[0][0]) + sum(dataAlt[0][1]))/1000,1),\
          " & ", round(sum(dataAlt[2][0]),1), " & ", round(sum(dataAlt[2][1]),1), " & ", round(sum(dataAlt[2][1])+sum(dataAlt[2][0]),1))

title = ["Datetime","TotalD","GEFS","NDFD","GAVG","REAL"]
for i in ["surge","nom","total"]:
    printAdd = "/Users/haoxiangyang/Desktop/Git/DFSC_FLORIDA/output/{}24_c.csv".format(i)
    fo = open(printAdd,'w',newline = '')
    csvWriter = csv.writer(fo,dialect = 'excel')
    csvWriter.writerow(title)
    csvWriter.writerows(outDict["{}D".format(i)])
    fo.close()
    
#%%  
    # load the alternative comparison
outDict = {}
outDict["surgeD"] = []
outDict["nomD"] = []
outDict["totalD"] = []
dataAlt = pickle.load(open("/Users/haoxiangyang/Desktop/Git/DFSC_FLORIDA/output/Test_FR50_H96_F72_R24_N24_REAL_2.p","rb"))
surgeT = 0
nomT = 0
Tlen = 456
for i in range(Tlen):
    surgeT += dataAlt[1][0][i]
    nomT += dataAlt[1][1][i]
    outDict["surgeD"].append([dataAlt[4][i],surgeT])
    outDict["nomD"].append([dataAlt[4][i],nomT])
    outDict["totalD"].append([dataAlt[4][i],surgeT + nomT])
for dType in ["GEFS","NDFD","GAVG","REAL"]:
    dataAlt = pickle.load(open("/Users/haoxiangyang/Desktop/Git/DFSC_FLORIDA/output/Test_FR50_H96_F72_R24_N24_{}_2.p".format(dType),"rb"))
    surgeT = 0
    nomT = 0
    for i in range(Tlen):
        surgeT += dataAlt[0][0][i]
        outDict["surgeD"][i].append(surgeT)
        nomT += dataAlt[0][1][i]
        outDict["nomD"][i].append(nomT)
        outDict["totalD"][i].append(surgeT + nomT)
    print("---------------- {} -------------------".format(dType))
    print(" & ", round(sum(dataAlt[0][0])/1000,1), " & ", round(sum(dataAlt[0][1])/1000,1), " & ", round((sum(dataAlt[0][0]) + sum(dataAlt[0][1]))/1000,1),\
          " & ", round(sum(dataAlt[2][0]),1), " & ", round(sum(dataAlt[2][1]),1), " & ", round(sum(dataAlt[2][1])+sum(dataAlt[2][0]),1))

title = ["Datetime","TotalD","GEFS","NDFD","GAVG","REAL"]
for i in ["surge","nom","total"]:
    printAdd = "/Users/haoxiangyang/Desktop/Git/DFSC_FLORIDA/output/{}24_2.csv".format(i)
    fo = open(printAdd,'w',newline = '')
    csvWriter = csv.writer(fo,dialect = 'excel')
    csvWriter.writerow(title)
    csvWriter.writerows(outDict["{}D".format(i)])
    fo.close()
    
#%%
# plot the bar plot for the cost 2
plt.rcParams["font.size"] = "22"
fig, ax1 = plt.subplots(1, 1, figsize=(17, 9))
ax1.xaxis.label.set_fontsize(24)
ax1.yaxis.label.set_fontsize(24)
ax1.ticklabel_format(style="sci", scilimits=(0,0))

ax2 = ax1.twinx()
ax2.spines["right"].set_visible(True)
ax2.yaxis.label.set_fontsize(24)
ax2.set_ylim(0,2000000)
ax2.ticklabel_format(style="sci", scilimits=(0,0))
textList = ["GEFS cost 1", "GEFS cost 2"]

ori_sd = np.array([154.1,481.9,169.1,4.1])
new_sd = np.array([151.8,482.3,169.1,4.1])
ori_nd = np.array([1019.7,714.4,1004.9,1169.3])
new_nd = np.array([1021.9,714.3,1004.9,1169.3])
ori_cost = np.array([1372391.8,1986186.9,1409690.5,1177643.3])
new_cost = np.array([1887784.5,4808748.9,2147873.6,1185936.8])

sd_l = np.array([139282,139539])
sd_h = np.array([14817,12307])
nd_l = np.array([1019740,1021944])
nd_h = np.array([0,0])
ori_cost = np.array([1372391.8,1887784.5])

X = np.arange(2)
barList1 = ax1.bar(X,sd_l,width = 0.2, color = 'red',alpha = 0.6)
barList2 = ax1.bar(X+0.2,sd_h, width = 0.2,color = 'red',alpha = 0.6,hatch = '/')
barList3 = ax2.bar(X+0.4,nd_l, width = 0.2,color = 'blue',alpha = 0.6)
barList4 = ax2.bar(X+0.6,nd_h, width = 0.2,color = 'blue',alpha = 0.6,hatch = '/')
#barList5 = ax2.bar(X+0.6,ori_cost, width = 0.15,color = 'gray',alpha = 0.6)

ax1.set_ylabel("Surge Shortage (barrels)")
ax2.set_ylabel("Nominal Shortage (barrels)")

ax1.set_ylim(0,150000)
ax1.set_xticks([0.3,1.3])
f = lambda x: textwrap.fill(x, 10)
ax1.set_xticklabels(map(f, textList))
ax1.legend([barList1,barList2,barList3,barList4],
           ['LCS','HCS','LCN','HCN'])
plot_filename = '/Users/haoxiangyang/Dropbox/Research_Documents/Hurricane/Writeup/shortage_bar.png'
plt.savefig(plot_filename, dpi=300)