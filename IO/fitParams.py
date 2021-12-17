#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 23:44:04 2019

@author: haoxiangyang
"""

# scripts to fit mapping models
import os
os.chdir("/Users/haoxiangyang/Desktop/Git/DFSC_FLORIDA/IO")
import csv
import datetime
import re
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from sklearn import linear_model
import pickle
from dataProcess import *

countyList = ["ALACHUA","BAKER","BAY","BRADFORD","BREVARD","BROWARD","CALHOUN",\
                  "CHARLOTTE","CITRUS","CLAY","COLLIER","COLUMBIA","DESOTO","DIXIE",\
                  "DUVAL","ESCAMBIA","FLAGLER","FRANKLIN","GADSDEN","GILCHRIST",\
                  "GLADES","GULF","HAMILTON","HARDEE","HENDRY","HERNANDO","HIGHLANDS",\
                  "HILLSBOROUGH","HOLMES","INDIAN RIVER","JACKSON","JEFFERSON",\
                  "LAFAYETTE","LAKE","LEE","LEON","LEVY","LIBERTY","MADISON","MANATEE",\
                  "MARION","MARTIN","MIAMI-DADE","MONROE","NASSAU","OKALOOSA",\
                  "OKEECHOBEE","ORANGE","OSCEOLA","PALM BEACH","PASCO","PINELLAS",\
                  "POLK","PUTNAM","SANTA ROSA","SARASOTA","SEMINOLE","ST. JOHNS",\
                  "ST. LUCIE","SUMTER","SUWANNEE","TAYLOR","UNION","VOLUSIA",\
                  "WAKULLA","WALTON","WASHINGTON"]
countyCode = {"00235":"Citrus County","00325":"Gadsden County","00326":"Nassau County",\
              "00360":"Okeechobee County","00415":"Bradford County","00482":"Sarasota County",\
              "00485":"Flagler County","03818":"Jackson County","12812":"Charlotte County",\
              "12815":"Orange County","12816":"Alachua County","12818":"Hernando County",\
              "12819":"Lake County","12832":"Franklin County","12833":"Dixie County",\
              "12834":"Volusia County","12836":"Monroe County","12838":"Brevard County",\
              "12839":"Miami-Dade County","12842":"Hillsborough County","12843":"Indian River County",\
              "12844":"Palm Beach County","12849":"Broward County","12854":"Seminole County",\
              "12861":"Marion County","12871":"Manatee County","12873":"Pinellas County",\
              "12883":"Polk County","12894":"Lee County","12895":"St. Lucie County",\
              "12897":"Collier County","13884":"Okaloosa County","13899":"Escambia County",\
              "53848":"Santa Rosa County","53862":"Taylor County","63871":"Holmes County",\
              "73805":"Bay County","92813":"Osceola County","92814":"St. Johns County",\
              "92815":"Martin County","92817":"Sumter County","92827":"Highlands County",\
              "93805":"Leon County","00486":"Pasco County"}
countyName = ["Alachua County","Baker County","Bay County","Bradford County","Brevard County",\
              "Broward County","Calhoun County","Charlotte County","Citrus County","Clay County",\
              "Collier County","Columbia County","DeSoto County","Dixie County","Duval County",\
              "Escambia County","Flagler County","Franklin County","Gadsden County","Gilchrist County",\
              "Glades County","Gulf County","Hamilton County","Hardee County","Hendry County",\
              "Hernando County","Highlands County","Hillsborough County","Holmes County",
              "Indian River County","Jackson County","Jefferson County","Lafayette County",\
              "Lake County","Lee County","Leon County","Levy County","Liberty County","Madison County",\
              "Manatee County","Marion County","Martin County","Miami-Dade County","Monroe County",\
              "Nassau County","Okaloosa County","Okeechobee County","Orange County","Osceola County",\
              "Palm Beach County","Pasco County","Pinellas County","Polk County","Putnam County",\
              "St. Johns County","St. Lucie County","Santa Rosa County","Sarasota County",\
              "Seminole County","Sumter County","Suwannee County","Taylor County","Union County",\
              "Volusia County","Wakulla County","Walton County","Washington County"]
countyNameCap = [i[:-7].upper() for i in countyName]

# obtain the county-based wind data
lcdData = lcdTotalParser("/Users/haoxiangyang/Dropbox/Research_Documents/Hurricane/Data/Irma_LCD/LCD_FL.csv",countyCode)
totalData = pickle.load(open('/Users/haoxiangyang/Desktop/Git/DFSC_FLORIDA/data/power_outage_data.p', 'rb'))
windNDFD,windLoc = pickle.load(open('/Users/haoxiangyang/Desktop/Git/DFSC_FLORIDA/data/windNDFD.p', 'rb'))
gustNDFD,gustLoc = pickle.load(open('/Users/haoxiangyang/Desktop/Git/DFSC_FLORIDA/data/gustNDFD.p', 'rb'))

maxt = datetime.datetime(2017,9,25,0,0)
mint = min(totalData.keys())
cset = [i for i in countyCode.values()]

yearHurricane = 2017
t_step = 6
totalDemand = 2.383e11/(365*24/t_step)

#%%
# find the how long it takes to recover
stageTDict = {}
recoveryTDict = {}
peakDict = {}
xTemporal = np.array([])
yTemporal = np.array([])
zTemporal = np.array([])

for i in range(len(cset)):
    c = cset[i]
    x = np.array([])
    y = np.array([])
    for tp in sorted(totalData.keys()):
        #x = np.append(x,(tp - mint)/(maxt - mint))
        if (tp <= maxt)and(tp >= mint):
            x = np.append(x,tp)
            y = np.append(y,totalData[tp][c][0]/totalData[tp][c][1])
    # find the peak time
    peakT = y.argmax()
    if y[peakT] >= 0.2:
        lastGoodB = 0
        firstGoodA = len(y) - 1
        for loc in range(peakT):
            if y[loc] < 0.01:
                lastGoodB = loc
        for loc in range(peakT,len(y)):
            if y[loc] > 0.01:
                firstGoodA = loc
        if firstGoodA < len(y) - 1:
            firstGoodA += 1
        stageTDict[c] = x[peakT] - x[lastGoodB]
        zTemporal = np.append(zTemporal,stageTDict[c].total_seconds())
        recoveryTDict[c] = x[firstGoodA] - x[peakT]
        yTemporal = np.append(yTemporal,recoveryTDict[c].total_seconds())
        peakDict[c] = y[peakT]
        xTemporal = np.append(xTemporal,peakDict[c])
        
reg = linear_model.LinearRegression()
reg.fit(np.transpose(np.matrix(xTemporal)),yTemporal)

#%%
plMax = []
for i in range(len(cset)):
    c = cset[i]
    cInd = windLoc.index(c)
    x = np.array([])
    y = np.array([])
    yMax = -1
    tMax = list(sorted(totalData.keys()))[0]
    for tp in sorted(totalData.keys()):
        #x = np.append(x,(tp - mint)/(maxt - mint))
        if (tp <= maxt)and(tp >= mint):
            x = np.append(x,tp)
            y = np.append(y,totalData[tp][c][0]/totalData[tp][c][1])
            if totalData[tp][c][0]/totalData[tp][c][1] > yMax:
                yMax = totalData[tp][c][0]/totalData[tp][c][1]
                tMax = tp
    recBool = True
    recordTime = list(sorted(totalData.keys()))
    recInd = recordTime.index(tMax)
    while recBool:
        remainOut = [totalData[ttp][c][0]/totalData[ttp][c][1] for ttp in totalData.keys() if ttp >= recordTime[recInd]]
        if max(remainOut) <= 0.1:
            recBool = False
        else:
            recInd += 1
    if yMax > 0:
        plMax.append((c,yMax,recordTime[recInd] - tMax))
    xPart = np.array([])
    yPart = np.array([])
    for tp in sorted(totalData.keys()):
        #x = np.append(x,(tp - mint)/(maxt - mint))
        if (tp <= maxt)and(tp >= mint)and(tp >= tMax):
            tDiff = tp - tMax
            xPart = np.append(xPart,tDiff.total_seconds()/86400)
            yPart = np.append(yPart,totalData[tp][c][0]/totalData[tp][c][1])


# plot the fit between max outage and recovery time
fig3, ax3 = plt.subplots(figsize=(15,10))
for item in ([ax3.xaxis.label, ax3.yaxis.label] +\
     ax3.get_xticklabels() + ax3.get_yticklabels()):
    item.set_fontsize(20)

ax3.set_xlabel('time (hours)',fontsize = 24)
# Make the y-axis label, ticks and tick labels match the line color.
ax3.set_ylabel('Power Loss',fontsize = 24)
ax3.set_ybound(0,1)

# Fit the maximum outage vs. recovery time
xrecOri = np.array([plMax[i][1] for i in range(len(plMax))])
yrecOri = np.array([plMax[i][2].total_seconds()/3600 for i in range(len(plMax))])
xrec = np.array([plMax[i][1] for i in range(len(plMax))])
yrec = np.array([plMax[i][2].total_seconds()/3600 for i in range(len(plMax))])
regrec = linear_model.LinearRegression()
regrec.fit(np.transpose(np.matrix(xrec)),yrec)
regrecScore = regrec.score(np.transpose(np.matrix(xrec)),yrec)
xrecTest = np.arange(0,1,0.01)
yrecTest = regrec.intercept_ + regrec.coef_*xrecTest
plt.plot(yrec,xrec,'o',color = "black",markersize = 10)
plt.plot(yrecTest,xrecTest,linewidth = 6)

ax3.legend(("Data","Fitted"),loc = 'upper left',fontsize = 20)

fig3.savefig("/Users/haoxiangyang/Dropbox/Research_Documents/Hurricane/Writeup/recoveryMapping.png", dpi=300)
dieCoeff = t_step*3600/(regrec.coef_[0]/100)

# Fit a quadratic model
fig3, ax3 = plt.subplots(figsize=(15,10))
for item in ([ax3.xaxis.label, ax3.yaxis.label] +\
     ax3.get_xticklabels() + ax3.get_yticklabels()):
    item.set_fontsize(20)

ax3.set_xlabel('time (hours)',fontsize = 24)
# Make the y-axis label, ticks and tick labels match the line color.
ax3.set_ylabel('Power Loss',fontsize = 24)
ax3.set_ybound(0,1)

xrec2 = xrec**2
xList = []
for i in range(len(xrec)):
    xList.append([xrec[i],xrec2[i]])
xmat = np.matrix(xList)
regrec2 = linear_model.LinearRegression()
regrec2.fit(xmat,yrec)
yrecTest2 = regrec.coef_[0]*xrecTest + regrec2.coef_[1]*xrecTest**2 + regrec.intercept_
plt.plot(yrec,xrec,'o',color = "black",markersize = 10)
plt.plot(yrecTest2,xrecTest,linewidth = 6)
ax3.legend(("Data","Fitted"),loc = 'upper left',fontsize = 20)

fig3.savefig("/Users/haoxiangyang/Dropbox/Research_Documents/Hurricane/Writeup/recoveryMapping_quad.png", dpi=300)
dieCoeff = t_step*3600/(regrec.coef_[0]/100)

#xrec2 = xrecOri**2
#yrec2 = yrecOri
#regrec2 = linear_model.LinearRegression()
#regrec2.fit(np.transpose(np.matrix(xrec2)),yrec2)
#regrecScore2 = regrec2.score(np.transpose(np.matrix(xrec2)),yrec2)
#xrecTest2 = np.arange(0,1,0.01)**2
#yrecTest2 = regrec2.intercept_ + regrec2.coef_*xrecTest2
#plt.plot(xrec2,yrec2,'o',color = "black",markersize = 10)
#plt.plot(xrecTest2,yrecTest2,linewidth = 6)
#
#xrec3 = xrec
#yrec3 = np.log(yrec)
#regrec3 = linear_model.LinearRegression()
#regrec3.fit(np.transpose(np.matrix(xrec3)),yrec3)
#regrecScore3 = regrec3.score(np.transpose(np.matrix(xrec3)),yrec3)
#xrecTest3 = np.arange(0,1,0.01)
#yrecTest3 = np.exp(regrec3.intercept_ + regrec3.coef_*xrecTest3)
#plt.plot(xrec3,yrec3,'o',color = "black",markersize = 10)
#plt.plot(xrecTest3,yrecTest3,linewidth = 6)

#%%
# wind vs. power loss at any given point before max power loss
xmAll = np.array([])
ymAll = np.array([])
zmAll = np.array([])
windBool = True
windMax = []
surgeMax = []
for i in range(len(cset)):
    c = cset[i]
    cInd = windLoc.index(c)
    cCap = countyNameCap[countyName.index(c)]
    x = np.array([])
    y = np.array([])
    for tp in sorted(totalData.keys()):
        # record the time points and the outage
        if (tp <= maxt)and(tp >= mint):
            x = np.append(x,tp)
            y = np.append(y,totalData[tp][c][0]/totalData[tp][c][1])
    xw = np.array([])
    yw = np.array([])
    if windBool:
        for tp in sorted(windNDFD['WindSpd'].keys()):
            ttp = min(windNDFD['WindSpd'][tp].keys())
            if (ttp <= maxt)and(ttp >= mint):
                xw = np.append(xw,ttp)
                yw = np.append(yw,windNDFD['WindSpd'][tp][ttp][cInd])
    else:
        for tp in sorted(gustNDFD['WindGust'].keys()):
            ttp = min(gustNDFD['WindGust'][tp].keys())
            if (ttp <= maxt)and(ttp >= mint):
                xw = np.append(xw,ttp)
                if tp in windNDFD['WindSpd'].keys():
                    if ttp in windNDFD['WindSpd'][tp].keys():
                        yTemp = max(gustNDFD['WindGust'][tp][ttp][cInd],windNDFD['WindSpd'][tp][ttp][cInd])
                    else:
                        yTemp = gustNDFD['WindGust'][tp][ttp][cInd]
                else:
                    yTemp = gustNDFD['WindGust'][tp][ttp][cInd]
                yw = np.append(yw,yTemp)
    ymaxInd = np.argmax(y)
    for xind in range(ymaxInd + 1):
        if y[xind] > 0.05:
            indCurrList = [i for i in range(len(xw)) if xw[i] < x[xind]]
            yCurrList = yw[indCurrList]
            ywmax = max(yCurrList)
            xmAll = np.append(xmAll,ywmax)
            ymAll = np.append(ymAll,y[xind])

#%%
# max wind power vs. max power loss
xm = np.array([])
ym = np.array([])
zm = np.array([])
windBool = False
windMax = []
surgeMax = []
for i in range(len(cset)):
    c = cset[i]
    cInd = windLoc.index(c)
    cCap = countyNameCap[countyName.index(c)]
    x = np.array([])
    y = np.array([])
    for tp in sorted(totalData.keys()):
        #x = np.append(x,(tp - mint)/(maxt - mint))
        if (tp <= maxt)and(tp >= mint):
            x = np.append(x,tp)
            y = np.append(y,totalData[tp][c][0]/totalData[tp][c][1])
    #plt.plot(x,y)
    
    # plot the county wind plot
#    dataW = lcdData[c]
#    dataWsorted = sorted(dataW,key = lambda k:k[0])
    xw = np.array([])
    yw = np.array([])
#    for item in dataWsorted:
#        if (item[0] <= maxt)and(item[0] >= mint):
#            if item[1] != None:
#                if item[2] != None:
#                    windSFinal = item[2]
#                else:
#                    windSFinal = item[1]
#                #xw = np.append(xw,(item[0] - mint)/(maxt - mint))
#                xw = np.append(xw,item[0])
#                yw = np.append(yw,windSFinal)
    # add NDFD data here
    if windBool:
        for tp in sorted(windNDFD['WindSpd'].keys()):
            ttp = min(windNDFD['WindSpd'][tp].keys())
            if (ttp <= maxt)and(ttp >= mint):
                xw = np.append(xw,ttp)
                yw = np.append(yw,windNDFD['WindSpd'][tp][ttp][cInd])
        if (len(y) != 0)and(len(yw) != 0):
            ym = np.append(ym,max(y))
            xm = np.append(xm,max(yw))
            windMax.append((c,max(y),max(yw)))
    else:
        for tp in sorted(gustNDFD['WindGust'].keys()):
            ttp = min(gustNDFD['WindGust'][tp].keys())
            if (ttp <= maxt)and(ttp >= mint):
                xw = np.append(xw,ttp)
                if tp in windNDFD['WindSpd'].keys():
                    if ttp in windNDFD['WindSpd'][tp].keys():
                        yTemp = max(gustNDFD['WindGust'][tp][ttp][cInd],windNDFD['WindSpd'][tp][ttp][cInd])
                    else:
                        yTemp = gustNDFD['WindGust'][tp][ttp][cInd]
                else:
                    yTemp = gustNDFD['WindGust'][tp][ttp][cInd]
                yw = np.append(yw,yTemp)
        if (len(y) != 0)and(len(yw) != 0):
            ym = np.append(ym,max(y))
            xm = np.append(xm,max(yw))
            windMax.append((c,max(y),max(yw)))
            
reg = linear_model.LinearRegression()
reg.fit(np.transpose(np.matrix(xm)),ym)
reg1 = linear_model.LinearRegression()
reg1.fit(np.transpose(np.matrix(xm)),-np.log(1/ym - 1))
xt = np.array(range(0,140,2))
yt = 1/(1 + np.exp(-(xt*reg1.coef_ + reg1.intercept_)))
plt.scatter(xm,ym,color = "black")
plt.plot(xt,yt)

#%%
# anchor down the (0,0) point
xmp = xm.copy()
ymp = ym.copy()
#for i in range(0,40,5):
ymp = np.append(ymp,0.00001)
xmp = np.append(xmp,0)
fig2, ax2 = plt.subplots(figsize=(15,10))
for item in ([ax2.xaxis.label, ax2.yaxis.label] +\
             ax2.get_xticklabels() + ax2.get_yticklabels()):
    item.set_fontsize(20)

ax2.set_xlabel('Gust Speed (mph)',fontsize = 24)
# Make the y-axis label, ticks and tick labels match the line color.
ax2.set_ylabel('Power Loss',fontsize = 24)
ax2.set_ybound(0,1)
ax2.tick_params(axis = "y",labelsize = 20)
ax2.tick_params(axis = "x",labelsize = 20)
reg2 = linear_model.LinearRegression()
reg2.fit(np.transpose(np.matrix(xmp)),-np.log(1/ymp - 1))
xt2 = np.array(range(0,140,2))
yt2p = xt2*reg2.coef_ + reg2.intercept_
yt2 = 1/(1 + np.exp(-(xt2*reg2.coef_ + reg2.intercept_)))
plt.plot(xmp,ymp,'o',color = "black",markersize = 10)
plt.plot(xt2,yt2,linewidth = 6)
ax2.legend(("Data","Fitted"),loc = 'upper left',fontsize = 20)

fig2.savefig("/Users/haoxiangyang/Dropbox/NU Documents/Hurricane/Writeup/outageMapping.png", dpi=300)


#mapFunc = [-6.3876227823537075,0.08887279]
mapFunc = [reg2.intercept_,reg2.coef_[0]]

#plt.scatter(xmp,-np.log(1/ymp - 1))
#plt.plot(xt2,yt2p)
#
#plt.scatter(xmp,ymp)
#plt.scatter(xmp,-np.log(1/ymp - 1))
#plt.plot(xt2,yt2p)
#
#
#fo = open("WindvLoss.csv","w",newline = '')
#csvWriter = csv.writer(fo,dialect = 'excel')
#
#for i in range(len(xm)):
#    csvWriter.writerow([xm[i],zm[i],ym[i]])
#fo.close()