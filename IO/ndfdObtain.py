
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 16:16:13 2019

@author: haoxiangyang
"""
import os
os.chdir("/Users/haoxiangyang/Desktop/Git/daniel_Diesel/IO/")
from ndfdParser import *
import pickle
from geopy import geocoders
gn = geocoders.GeoNames(username = 'haoxiangyang89')
import re
import csv
import numpy as np
#==============================================================================
# import county names
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

#%%
# obtain the NDFD data for wind/gust speed
windNDFD,windLoc = extractNDFD('/Users/haoxiangyang/Desktop/outWindSpd/output_sustained/')
# change knots to mph
for i in windNDFD['WindSpd'].keys():
    for j in windNDFD['WindSpd'][i].keys():
        windNDFD['WindSpd'][i][j] = [round(k*1.15078,4) for k in windNDFD['WindSpd'][i][j]]
gustNDFD,gustLoc = extractNDFD('/Users/haoxiangyang/Desktop/outWindSpd/output_gust/')
for i in gustNDFD['WindGust'].keys():
    for j in gustNDFD['WindGust'][i].keys():
        gustNDFD['WindGust'][i][j] = [round(k*1.15078,4) for k in gustNDFD['WindGust'][i][j]]

with open('/Users/haoxiangyang/Desktop/Git/daniel_Diesel/data/windNDFD.p', 'wb') as fp:
    pickle.dump([windNDFD,windLoc], fp, protocol=pickle.HIGHEST_PROTOCOL)
with open('/Users/haoxiangyang/Desktop/Git/daniel_Diesel/data/gustNDFD.p', 'wb') as fp:
    pickle.dump([gustNDFD,gustLoc], fp, protocol=pickle.HIGHEST_PROTOCOL)

#%%
# obtain predicted NDFD demand
    
fitParam = [-6.3876227823537075,0.08887279]
recoverParam = [-7.1017645909936817,171.58803196]
cutoffVal = -recoverParam[0]/recoverParam[1]
t_step = 6
# from the outage percentage compute the real demand
totalDemand = 2.383e11/(365*24/t_step)
demandElectricity = {}
for c in countyName:
    demandElectricity[c] = fl_df.Population[c]/sum(fl_df.Population)*totalDemand
    startT = datetime.datetime(2017,9,6,0,0)
    endT = datetime.datetime(2017,9,24,18,0)
    refTList = []
    currentT = startT
    while currentT <= endT:
        refTList.append(currentT)
        currentT += datetime.timedelta(hours = 6)

fulRateList = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
for fr in fulRateList:
    # NDFDdata,locList,refTList,fitParam,recoverParam,realDemand,cutoff = 0.0,demandDict = {},fulfillRate = 0.2
    outPdict,dDict = obtainPredNDFD(gustNDFD,gustLoc,refTList,fitParam,recoverParam[1],realDemand,cutoffVal,demandElectricity,fr)
    with open('../data/predDemand/predNDFD_{}.p'.format(int(fr*100)), 'wb') as fp:
        pickle.dump(outPdict, fp, protocol=pickle.HIGHEST_PROTOCOL)
#%%
# obtain the county seats and their Lat/Long
countyFile = open('/Users/haoxiangyang/Desktop/outSurgeProb/countySeat.txt','rb')
countyTxt = countyFile.read()
countyTxt = countyTxt.decode('utf-8')
countySeat = re.findall('Name= ([A-Za-z .\-]+?) \|Seat= ([A-Za-z .\-]+) ',countyTxt)
countySeatList = []
for item in countySeat:
    countyInfo = gn.geocode(item[0] + ", FL")
    countySeatList.append([item[0] + ' County',countyInfo.latitude,countyInfo.longitude])
countyFile.close()
countySeatFile = open('/Users/haoxiangyang/Desktop/outSurgeProb/pntsFL.txt','w',newline = '')
csvWriter = csv.writer(countySeatFile,dialect = 'excel')
csvWriter.writerows(countySeatList)
countySeatFile.close()

# obtain the pSurge data
surgeData,surgeLoc = extractNDFD('/Users/haoxiangyang/Desktop/outSurgeProb/output/')
with open('/Users/haoxiangyang/Desktop/Git/daniel_Diesel/data/pSurge.p', 'wb') as fp:
    pickle.dump([surgeData,surgeLoc], fp, protocol=pickle.HIGHEST_PROTOCOL)

#%%
# obtain the NDFD data for probability generation
windProbGridOri,windLocGridOri = extractNDFD('/Users/haoxiangyang/Desktop/outWindProb/output_Grid/')
testTime1 = list(windProbGridOri['ProbWindSpd64c'].keys())[0]
testTime2 = list(windProbGridOri['ProbWindSpd64c'][testTime1].keys())[0]
windLocGrid = []
windLocGridInd = []
for i in range(len(windLocGridOri)):
    if windProbGridOri['ProbWindSpd64c'][testTime1][testTime2][i] != 9999:
        windLocGrid.append(windLocGridOri[i])
        windLocGridInd.append(i)
windProbGrid = {}
for i in windProbGridOri.keys():
    windProbGrid[i] = {}
    for j in windProbGridOri[i].keys():
        windProbGrid[i][j] = {}
        for k in windProbGridOri[i][j].keys():
            windProbGrid[i][j][k] = [windProbGridOri[i][j][k][l] for l in windLocGridInd]

with open('/Users/haoxiangyang/Desktop/Git/daniel_Diesel/data/windProbGrid.p', 'wb') as fp:
    pickle.dump([windProbGrid,windLocGrid,windLocGridInd], fp, protocol=pickle.HIGHEST_PROTOCOL)
    
#%%
# obtain the NDFD data for probability fitting process
windProbOri,windLocOri = extractNDFD('/Users/haoxiangyang/Desktop/outWindProb/output/')
testTime1 = list(windProbOri['ProbWindSpd64c'].keys())[0]
testTime2 = list(windProbOri['ProbWindSpd64c'][testTime1].keys())[0]
windLoc = []
windLocInd = []
for i in range(len(windLocOri)):
    if windProbOri['ProbWindSpd64c'][testTime1][testTime2][i] != 9999:
        windLoc.append(windLocOri[i])
        windLocInd.append(i)
windProb = {}
for i in windProbOri.keys():
    windProb[i] = {}
    for j in windProbOri[i].keys():
        windProb[i][j] = {}
        for k in windProbOri[i][j].keys():
            windProb[i][j][k] = [windProbOri[i][j][k][l] for l in windLocInd]

with open('../data/windProb.p', 'wb') as fp:
    pickle.dump([windProb,windLoc,windLocInd], fp, protocol=pickle.HIGHEST_PROTOCOL)

# obtain the relationship between the NDFD wind speed pointwise estimation vs. wind speed probability estimation
xList = np.array([])
yList = {}
for i in [34,50,64]:
    yList[i] = np.array([])
for j in windProb['ProbWindSpd64i'].keys():
    for k in windProb['ProbWindSpd64i'][j]:
        if j in windNDFD['WindSpd'].keys():
            jKey = j
        else:
            prevTime = [jj for jj in windNDFD['WindSpd'].keys() if jj < j]
            if prevTime != []:
                jKey = max(prevTime)
            else:
                jKey = None
        if jKey != None:
            if k in windNDFD['WindSpd'][jKey].keys():
                xList = np.append(xList,windNDFD['WindSpd'][jKey][k])
                for i in [34,50,64]:
                    iKey = 'ProbWindSpd{}i'.format(i)
                    yList[i] = np.append(yList[i],windProb[iKey][j][k])
                    for l in windLocInd:
                        if (windProb[iKey][j][k][l] >= 80)and(windNDFD['WindSpd'][jKey][k][l] <= 20):
                            print(windLoc[l],windProb[iKey][j][k][l],windNDFD['WindSpd'][jKey][k][l],j,jKey,k,i)
                            
plt.scatter(xList,yList[34])
xTest = np.arange(0,140,1)
yTest = reg.coef_*xTest + reg.intercept_
plt.plot(xTest,yTest,color = "k")