#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 10:33:55 2019

@author: haoxiangyang
"""

# codes to plot the comparison between 
#   - real demand estimated from mapping the real wind speed (20% fulfillment rate)
#   - forecast demand estimated from mapping the NDFD wind speed forecast
#   - forecast demand estimated from mapping the average wind speed across 21 GEFS scenarios

import matplotlib.pyplot as plt
import numpy as np
import pickle
import datetime
realdDict = pickle.load(open("../data/realDemand.p","rb"))
ndfdDemand = pickle.load(open("../data/ndfdDemand_20.p","rb"))
avgDemand = pickle.load(open("../data/avgDemand_20.p","rb"))

startT = datetime.datetime(2017,9,7,0,0)
endT = datetime.datetime(2017,9,11,12,0)
currentT = startT
while currentT <= endT:
    figC, axC = plt.subplots(figsize=(15,10))
    for item in ([axC.xaxis.label, axC.yaxis.label, axC.title] +\
                 axC.get_xticklabels() + axC.get_yticklabels()):
        item.set_fontsize(20)
    
    axC.set_title(str(currentT))
    axC.title.set_fontsize(20)
    axC.set_xlabel('Hours')
    # Make the y-axis label, ticks and tick labels match the line color.
    axC.set_ylabel('Surge Demand (barrels)')
    axC.set_ybound(0,1)
    
    plt.plot(np.array(list(range(len(realdDict[0][currentT].keys()))))*6, [realdDict[0][currentT][j]['Miami-Dade County'] for j in realdDict[0][currentT].keys()], label = 'Real Demand',linewidth = 6)
    plt.plot(np.array(list(range(len(ndfdDemand[0][currentT].keys()))))*6, [ndfdDemand[0][currentT][j]['Miami-Dade County'] for j in ndfdDemand[0][currentT].keys()], label = 'NDFD',linewidth = 6)
    plt.plot(np.array(list(range(len(avgDemand[0][currentT].keys()))))*6, [avgDemand[0][currentT][j]['Miami-Dade County'] for j in sorted(avgDemand[0][currentT].keys())], label = 'Average GEFS',linewidth = 6)
    axC.legend(loc = 'upper right', fontsize = 20)
    plt.show()
    currentT += datetime.timedelta(hours = 12)