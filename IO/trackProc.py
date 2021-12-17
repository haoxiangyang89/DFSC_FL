#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 22:08:32 2018

@author: haoxiangyang
"""
import datetime
import pickle
import haversine
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

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
# interpret the locational information
def interpLoc(advData,advList,startT,endT,lambdaT):
    startLoc = (advData[advList[startT]][1][0],advData[advList[startT]][1][1])
    endLoc = (advData[advList[endT]][1][0],advData[advList[endT]][1][1])
    lat = round(endLoc[0]*lambdaT + startLoc[0]*(1 - lambdaT),2)
    long = round(endLoc[1]*lambdaT + startLoc[1]*(1 - lambdaT),2)
    windStr = round(advData[advList[endT]][1][5]*lambdaT + advData[advList[startT]][1][5]*(1 - lambdaT),2)
    gustStr = round(advData[advList[endT]][1][6]*lambdaT + advData[advList[startT]][1][6]*(1 - lambdaT),2)
    return lat,long,windStr,gustStr

# process the track data of a hurricane
# starting from the first advisory, output the location of each hour
# if not recorded, select the point between two recorded centers
# the distance is proportional to the time elapsed
def obtainTrack(advData):
    trackInfo = {}
    advList = sorted(list(advData.keys()))
    startTime = advData[advList[0]][0]
    endTime = advData[advList[len(advList) - 1]][0]
    timeList = [advData[i][0] for i in advList]
    currentT = startTime
    tDelta = datetime.timedelta(hours = 1)
    while currentT <= endTime:
        # if the current time is recorded
        if currentT in timeList:
            timeInd = timeList.index(currentT)
            lat = round(advData[advList[timeInd]][1][0],2)
            long = round(advData[advList[timeInd]][1][1],2)
            windStr = advData[advList[timeInd]][1][5]
            gustStr = advData[advList[timeInd]][1][6]
            trackInfo[currentT] = (lat,long,windStr,gustStr)
        else:
            startT = max([i for i in range(len(timeList)) if timeList[i] < currentT])
            endT = min([i for i in range(len(timeList)) if timeList[i] > currentT])
            lambdaT = (currentT - timeList[startT])/(timeList[endT] - timeList[startT])
            lat,lon,windStr,gustStr = interpLoc(advData,advList,startT,endT,lambdaT)
            trackInfo[currentT] = (lat,long,windStr,gustStr)
        currentT += tDelta
    
    return trackInfo

#%%
advData_file='../data/advData.p'
advData = pickle.load(open(advData_file, 'rb'))
trackInfo = obtainTrack(advData)
with open('/Users/haoxiangyang/Desktop/Git/daniel_Diesel/data/trackInfo_Irma.p', 'wb') as fp:
    pickle.dump(trackInfo, fp, protocol=pickle.HIGHEST_PROTOCOL)

plData = pickle.load(open('/Users/haoxiangyang/Desktop/Git/daniel_Diesel/data/power_outage_data.p', 'rb'))
#%%
windData_file='../data/windData.p'
windData,gustData,errorList,errorDict,lcdData,windNDFD = pickle.load(open(windData_file,'rb'))
netwrok_file='../data/floridaNetObj.p'
fl_df, fl_edges = pickle.load(open(netwrok_file, 'rb'))
countyLat = {}
countyLong = {}
for i in range(len(fl_df.County)):
    countyLat[fl_df.County[i]] = fl_df.latitude[i]
    countyLong[fl_df.County[i]] = fl_df.longitude[i]

# obtain the closest point of the hurricane to each county
closestPt = {}
for c in countyName:
    cPtD = 9999999
    for tt in trackInfo.keys():
        centerInfo = trackInfo[tt]
        if haversine.haversine((centerInfo[0],centerInfo[1]),(countyLat[c],countyLong[c])) < cPtD:
            cPtD = haversine.haversine((centerInfo[0],centerInfo[1]),(countyLat[c],countyLong[c]))
            closestPt[c] = tt
            
# obtain the distance for the wind data before the hurricane passing by
distanceDict = {}
distanceList = []
for predTime in gustData.keys():
    if (predTime in trackInfo.keys())and(gustData[predTime] != {}):
        centerInfo = trackInfo[predTime]
        distanceDict[predTime] = {}
        for c in countyName:
            if predTime <= closestPt[c]:
                distanceDict[predTime][c] = haversine.haversine((centerInfo[0],centerInfo[1]),(countyLat[c],countyLong[c]))
                distanceList.append([gustData[predTime][c],distanceDict[predTime][c]])
            
distanceListSort = sorted(distanceList,key=lambda x: x[1])
reg = linear_model.LinearRegression()
X = [distanceListSort[i][1] for i in range(len(distanceListSort)) if distanceListSort[i][1] <= 300]
y = [distanceListSort[i][0] for i in range(len(distanceListSort)) if distanceListSort[i][1] <= 300]
reg.fit(np.transpose(np.matrix(X)),y)
plt.scatter(X,y)
X1 = np.array(range(300))
y1 = reg.intercept_+reg.coef_[0]*X1
plt.plot(X1,y1,'k-')

# Fit the one norm distance to the wind data
dirFitDict = {}
dirFitList = []
for predTime in windData.keys():
    if (predTime in trackInfo.keys())and(windData[predTime] != {}):
        centerInfo = trackInfo[predTime]
        dirFitDict[predTime] = {}
        for c in countyName:
            dirFitDict[predTime][c] = (centerInfo[0],centerInfo[1],countyLat[c],countyLong[c])
            dirFitList.append([windData[predTime][c],[abs(centerInfo[0] - countyLat[c]),abs(centerInfo[1] - countyLong[c])],haversine.haversine((centerInfo[0],centerInfo[1]),(countyLat[c],countyLong[c]))])
dirFitListSort = sorted(dirFitList,key=lambda x: x[2])
regd = linear_model.LinearRegression()
Xd = [dirFitListSort[i][1] for i in range(400)]
yd = [dirFitListSort[i][0] for i in range(400)]
regd.fit(np.matrix(Xd),yd)

# directly fit the demand to the distance^2
distanceDict = {}
distanceList = []
for predT in plData.keys():
    if predT.minute >= 30:
        predTime = datetime.datetime(predT.year,predT.month,predT.day,predT.hour,0) + datetime.timedelta(hours = 1)
    else:
        predTime = datetime.datetime(predT.year,predT.month,predT.day,predT.hour,0)
    if (predTime in trackInfo.keys())and(plData[predT] != {}):
        centerInfo = trackInfo[predTime]
        distanceDict[predTime] = {}
        for c in countyName:
            cCap = c[:-7].upper()
            if cCap != 'DUVAL':
                if predTime <= closestPt[c]:
                    distanceDict[predTime][c] = haversine.haversine((centerInfo[0],centerInfo[1]),(countyLat[c],countyLong[c]))
                    distanceList.append([plData[predT][cCap][0]/plData[predT][cCap][1],distanceDict[predTime][c]])


max_dist_plot = 1000
plt_data = np.array(distanceList)  
plt_data = plt_data[plt_data[:,1]<max_dist_plot]  
#plt_data = sorted(distanceList,key=lambda x: x[1])
reg = linear_model.LinearRegression()
X = [plt_data[i][1] for i in range(len(plt_data))]
y = [plt_data[i][0] for i in range(len(plt_data))]
reg.fit(np.transpose(np.matrix(X)),y)
plt.scatter(X,y)
X1 = np.array(range(max_dist_plot))
y1 = reg.intercept_+reg.coef_[0]*X1
plt.plot(X1,y1,'k-')
