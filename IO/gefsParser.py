#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 22:56:27 2018

@author: haoxiangyang
"""

import csv
import matplotlib.pyplot as plt
import datetime
import os
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import urllib.request
from bs4 import BeautifulSoup
import pickle
import seaborn as sns
from gurobipy import *
import itertools
from math import sin,cos,asin,sqrt
from NetworkBuilder import load_florida_network
# import pdb
# pdb.set_trace()

#%%
def getCountyName():
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
    return countyName,countyNameCap

def haversineEuclidean(lat1, lon1, lat2, lon2):
    '''
    Latitudes and longitudes in radians
    '''
    R = 6371 #Earth radius in Km
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    
    dLat = lat2 - lat1;
    dLon = lon2 - lon1;
    a = sin(dLat/2)**2 + (sin(dLon / 2)** 2)*cos(lat1)*cos(lat2);
    c = 2*asin(sqrt(a));
    return R * c;

def readGEFSheader(headerExplain):
    target_url = headerExplain
    html = urllib.request.urlopen(target_url).read()
    htmlRaw = html.decode('utf-8')
    soup = BeautifulSoup(htmlRaw, 'html.parser')
    lineObj = soup.find_all(name = 'tr')
    dictGEFS = {}
    for item in lineObj:
        entryList = [entry for entry in item.children if entry != '\n']
        if len(entryList) == 4:
            dictGEFS[entryList[3].text.strip()] = (entryList[1].text,entryList[2].text.strip())
    return dictGEFS
            
# parse the information from GEFS ensemble data
def readGEFS(filePath,GEFSdata = {},titleFilter = []):
    fileList = os.listdir(filePath)
    if GEFSdata == {}:
        GEFSdata = {'Wind':{},'UWind':{},'VWind':{},'Gust':{},'Pres':{}}
        for i in range(21):
            GEFSdata['Wind'][i] = {}
            GEFSdata['UWind'][i] = {}
            GEFSdata['VWind'][i] = {}
            GEFSdata['Gust'][i] = {}
            GEFSdata['Pres'][i] = {}
    for fileName in fileList:
        # the scenario no
        try:
            fileNameInfoList = fileName.split('_')
            scenNo = int(fileNameInfoList[5][0:2])
            fi = open(os.path.join(filePath,fileName),'r')
            csvReader = csv.reader(fi)
            # reference time and predicted time
            refTime = datetime.datetime(int(fileNameInfoList[2][0:4]),int(fileNameInfoList[2][4:6]),\
                                        int(fileNameInfoList[2][6:8]),int(fileNameInfoList[3][0:2]),\
                                        int(fileNameInfoList[3][2:4]))
            predictTime = refTime + datetime.timedelta(hours = int(fileNameInfoList[4]))
            counter = 0
            windU = []
            windV = []
            windList = []
            pressure = []
            gust = []
            rawData = []
            for item in csvReader:
                if counter == 0:
                    counter += 1
                    if titleFilter != []:
                        title = [item[i].strip() for i in range(5,len(item)) if item[i].strip() in titleFilter]
                        titleInd = [i for i in range(5,len(item)) if item[i].strip() in titleFilter]
                    else:
                        title = [item[i].strip() for i in range(5,len(item))]
                        titleInd = list(range(5,len(item)))
                else:
                    rawData.append(item)
                    if (item[0].strip() == 'PRES')and(item[2].strip() == '0[-] MSL="Mean sea level"'):
                        pressure = [float(item[i]) for i in titleInd]
                    elif (item[0].strip() == 'GUST'):
                        gust = [round(float(item[i])*1.15078,4) for i in titleInd]
                    elif (item[0].strip() == 'UGRD')and(item[2].strip() == '10[m] HTGL="Specified height level above ground"'):
                        windU = [float(item[i])*2.23694 for i in titleInd]
                    elif (item[0].strip() == 'VGRD')and(item[2].strip() == '10[m] HTGL="Specified height level above ground"'):
                        windV = [float(item[i])*2.23694 for i in titleInd]
                    
                    if (windU != [])and(windV != [])and(len(windU) == len(windV)):
                        lenWind = len(windU)
                        windList = [round(np.sqrt(windU[i]**2 + windV[i]**2),4) for i in range(lenWind)]
            fi.close()
            # wind
            if windList != []:
                if not(refTime in GEFSdata['Wind'][scenNo].keys()):
                    GEFSdata['Wind'][scenNo][refTime] = {}
                    GEFSdata['UWind'][scenNo][refTime] = {}
                    GEFSdata['VWind'][scenNo][refTime] = {}
                    if not(predictTime in GEFSdata['Wind'][scenNo][refTime].keys()):
                        GEFSdata['Wind'][scenNo][refTime][predictTime] = windList
                        GEFSdata['UWind'][scenNo][refTime][predictTime] = windU
                        GEFSdata['VWind'][scenNo][refTime][predictTime] = windV
                else:
                    if not(predictTime in GEFSdata['Wind'][scenNo][refTime].keys()):
                        GEFSdata['Wind'][scenNo][refTime][predictTime] = windList
                        GEFSdata['UWind'][scenNo][refTime][predictTime] = windU
                        GEFSdata['VWind'][scenNo][refTime][predictTime] = windV
            # gust
            if gust != []:
                if not(refTime in GEFSdata['Gust'][scenNo].keys()):
                    GEFSdata['Gust'][scenNo][refTime] = {}
                    GEFSdata['Gust'][scenNo][refTime][predictTime] = gust
                else:
                    GEFSdata['Gust'][scenNo][refTime][predictTime] = gust
                        
            # pressure
            if pressure != []:
                if not(refTime in GEFSdata['Pres'][scenNo].keys()):
                    GEFSdata['Pres'][scenNo][refTime] = {}
                    GEFSdata['Pres'][scenNo][refTime][predictTime] = pressure
                else:
                    GEFSdata['Pres'][scenNo][refTime][predictTime] = pressure
        except:
            print(fileName)
                 
    return title,GEFSdata

def findCenter(xyList):
    xadd = [-1,0,1,-1,1,-1,0,1]
    yadd = [-1,-1,-1,0,0,1,1,1]
    xlen,ylen = xyList.shape
    centerList = []
    centerBool = False
    for i in range(xlen):
        for j in range(ylen):
            counterTot = 0
            counter = 0
            maxWind = 0
            for k in range(8):
                xnew = i + xadd[k]
                ynew = j + yadd[k]
                if (xnew >= 0)and(xnew < xlen)and(ynew >= 0)and(ynew < ylen):
                    counterTot += 1
                    if xyList[xnew,ynew] >= xyList[i,j]+15:
                        counter += 1
                    if xyList[xnew,ynew] > maxWind:
                        maxWind = xyList[xnew,ynew]
            # if it is not on the edge, and blocks surrounding it have greater wind speed
            if (counterTot >= 5):
                if counter >= counterTot:
                    if not(centerBool):
                        centerList = [i,j,maxWind]
                        centerBool = True
                    else:
                        if maxWind > centerList[2]:
                            centerList = [i,j,maxWind]
    if not(centerBool):
        xyListInd = [(xyList[i,j],i,j) for i in range(len(xyList)) for j in range(len(xyList[i]))]
        xyListInd = sorted(xyListInd, key = lambda x:x[0], reverse = True)
        centerBlock = np.array([xyListInd[seq] for seq in range(8) if xyListInd[seq][0] >= 73.6499])
        if len(centerBlock) > 0:
            centerBlockMean = np.mean(centerBlock,axis = 0)
            centerList = [centerBlockMean[1],centerBlockMean[2],np.max(centerBlock,axis = 0)[0]]
    return centerList

def parseTrack(centerList,tdelta = datetime.timedelta(hours = 1)):
    timeList = [item[0] for item in centerList]
    startTime = timeList[0]
    endTime = timeList[-1]
    currentT = startTime
    trackInfo = {}
    while currentT <= endTime:
        if currentT in timeList:
            timeInd = timeList.index(currentT)
            if len(centerList[timeInd][1]) != 0:
                lat = round(centerList[timeInd][1][0],2)
                long = round(centerList[timeInd][1][1],2)
                windStr = round(centerList[timeInd][1][2],2)
            else:
                lat = -90
                long = 0
                windStr = 0
        else:
            startT = max([i for i in range(len(timeList)) if timeList[i] < currentT])
            endT = min([i for i in range(len(timeList)) if timeList[i] > currentT])
            lambdaT = (currentT - timeList[startT])/(timeList[endT] - timeList[startT])
            if (len(centerList[startT][1]) != 0)and(len(centerList[endT][1]) != 0):
                lat = round(centerList[endT][1][0]*lambdaT + centerList[startT][1][0]*(1 - lambdaT),2)
                long = round(centerList[endT][1][1]*lambdaT + centerList[startT][1][1]*(1 - lambdaT),2)
                windStr = round(centerList[endT][1][2]*lambdaT + centerList[startT][1][2]*(1 - lambdaT),2)
            else:
                lat = -90
                long = 0
                windStr = 0
        trackInfo[currentT] = (lat,long,windStr)
        currentT += tdelta
    return trackInfo

def obtainCenter(GEFSdata,xList,yList,refTlist,locDict,gustWindOpt = 'Wind',baseLatLong = (20,-88),scenList = list(range(21))):
    # obtain the center list for each list
    centerDict = {}
    trackDict = {}
    for refTime in refTlist:
        centerDict[refTime] = {}
        trackDict[refTime] = {}
        for scenNo in scenList:
            # obtain the predicted track of the reference time
            centerList = []
            for predictTime in sorted(list(GEFSdata[gustWindOpt][scenNo][refTime].keys())):
                dataPlot = GEFSdata[gustWindOpt][scenNo][refTime][predictTime]
                xyList = np.zeros([len(xList),len(yList)])
                for i in range(len(dataPlot)):
                    xyList[locDict[i][0],locDict[i][1]] = dataPlot[i]
                center = findCenter(xyList)
                if len(center) == 3:
                    center[0] += baseLatLong[0]
                    center[1] += baseLatLong[1]
                centerList.append((predictTime,center))
            centerDict[refTime][scenNo] = centerList
            trackDict[refTime][scenNo] = parseTrack(centerList)
            
    return centerDict,trackDict

def farthestCenter(centerDict,scenList = list(range(21))):
    # return the farthest two track by summing up their distance
    timeList = sorted(list(centerDict.keys()))
    fList = {}
    distList = {}
    for t in timeList:
        distMax = -99999
        indMax = [-1,-1]
        distList[t] = {}
        for i in scenList:
            for j in scenList:
                if i < j:
                    distanceIJ = 0
                    distanceIJavg = 0
                    for k in range(20):
                        if (centerDict[t][i][k][1] != [])and(centerDict[t][j][k][1] != []):
                            distanceIJ += haversineEuclidean(centerDict[t][i][k][1][0],centerDict[t][i][k][1][1],\
                                                             centerDict[t][j][k][1][0],centerDict[t][j][k][1][1])
                            distanceIJavg += 1
                    if distanceIJavg != 0:
                        distList[t][i,j] = distanceIJ/distanceIJavg
                        if distanceIJ/distanceIJavg > distMax:
                            indMax = [i,j]
                            distMax = distanceIJ/distanceIJavg
        fList[t] = indMax
                            
    return fList,distList

#%%
def obtainProb(locSet,GEFSdata,GEFSdataGrid,NDFDdata,locDict,locLatLong,refTlist = [], scenList = list(range(21)),\
               timeRange = list(range(1,15)), timeH = datetime.timedelta(hours = 6), levelSet = (34,50,64), baseLatLong = (20,-88),endLatLong = (32,-70)):
    # read in the GEFS data (county and grid) and NDFD data
    # locDict is the location dictionary mapping the index to each grid point in 2D
    # baseLatLong is the 2D origin point
    # use iterative method to compute the probability of the scenarios
    
    # obtain the reference times and list of scenarios
    if refTlist == []:
        refTlist = list(GEFSdata['Wind'][0].keys())
    
    # obtain the indices of each set
    locSetInd = list(range(len(locSet)))
    timeRangeInd = list(range(len(timeRange)))
    scenListInd = list(range(len(scenList)))
    levelSetInd = list(range(len(levelSet)))
        
    # for each reference time
    centerDict = {}
    trackDict = {}
    pDict = {}
    for s in scenList:
        pDict[s] = {}
    pCondDict = {}
    xList = list(range(baseLatLong[0],endLatLong[0]))
    yList = list(range(baseLatLong[1],endLatLong[1]))

    for refTime in refTlist:
        # the closest NDFD time that has all predictions
        NDFDref = max([iKey for iKey in NDFDdata['ProbWindSpd34c'].keys() if ((timeH + refTime) in NDFDdata['ProbWindSpd34c'][iKey].keys())and(iKey <= refTime)])
        # for each scenario
        centerDict[refTime] = {}
        trackDict[refTime] = {}
        for scenInd in scenListInd:
            # obtain the predicted track of the reference time
            scenNo = scenList[scenInd]
            centerList = []
            for predictTime in sorted(list(GEFSdata['Wind'][scenNo][refTime].keys()))[:len(timeRange)+1]:
                dataPlot = GEFSdataGrid['Wind'][scenNo][refTime][predictTime]
                xyList = np.zeros([len(xList),len(yList)])
                for i in range(len(dataPlot)):
                    xyList[locDict[i][0],locDict[i][1]] = dataPlot[i]
                center = findCenter(xyList)
                if len(center) == 3:
                    center[0] += baseLatLong[0]
                    center[1] += baseLatLong[1]
                centerList.append((predictTime,center))
            centerDict[refTime][scenInd] = centerList
            trackDict[refTime][scenInd] = parseTrack(centerList)
            
        pPdf = {}
        for j in levelSetInd:
            for t in timeRangeInd:
                if timeRange[t]*timeH + refTime in NDFDdata['ProbWindSpd{}c'.format(levelSet[j])][NDFDref].keys():
                    for i in locSetInd:
                        pPdf[i,j,t] = round(NDFDdata['ProbWindSpd{}c'.format(levelSet[j])][NDFDref][timeRange[t]*timeH + refTime][i]/100,5)
        
        # construct the math programming model to solve for the probability of each scenario
        m1 = Model('obtain_ps')
        termBool = False
        ps1 = m1.addVars(scenListInd,lb = 0, ub = 1)
        pDiff1 = m1.addVars(list(itertools.product(locSetInd,levelSetInd,timeRangeInd)),lb = -1,ub = 1)
        
        # initialize the conditional probability
        pCond1 = {}
        for s in scenListInd:
            for t in timeRangeInd:
                for i in locSetInd:
                    windSpd = GEFSdata['Wind'][scenList[s]][refTime][refTime + timeRange[t]*timeH][i]
                    for j in levelSetInd:
                        if windSpd >= levelSet[j]*1.15078:
                            pCond1[i,j,t,s] = 1
                        else:
                            pCond1[i,j,t,s] = 0
        
        # add the constraints and the objective function
        consDiff1 = m1.addConstrs(sum(ps1[s]*pCond1[i,j,t,s] for s in scenListInd) + pDiff1[i,j,t] == pPdf[i,j,t]\
                                 for i in locSetInd\
                                 for j in levelSetInd\
                                 for t in timeRangeInd)
        consSum = m1.addConstr(sum(ps1[s] for s in scenListInd) == 1)
        m1.setObjective(sum(sum(sum(pDiff1[i,j,t]*pDiff1[i,j,t] for t in timeRangeInd) for j in levelSetInd) for i in locSetInd),GRB.MINIMIZE)
        m1.update()
        m1.optimize()
        
        m2 = Model('obtain_cond')
        pCond2 = m2.addVars(list(itertools.product(locSetInd,levelSetInd,timeRangeInd,scenListInd)), lb = 0, ub = 1)
        pDiff2 = m2.addVars(list(itertools.product(locSetInd,levelSetInd,timeRangeInd)),lb = -1,ub = 1)
        consDiff2 = m2.addConstrs(sum(round(ps1[s].X,6)*pCond2[i,j,t,s] for s in scenListInd) + pDiff2[i,j,t] == pPdf[i,j,t]\
                         for i in locSetInd\
                         for j in levelSetInd\
                         for t in timeRangeInd)
        consRank2 = m2.addConstrs(pCond2[i,j,t,s] >= pCond2[i,j+1,t,s]\
                                 for i in locSetInd\
                                 for j in levelSetInd[:-1]\
                                 for t in timeRangeInd\
                                 for s in scenListInd)

        disDict = {}
        consSeq2 = {}
        for i in locSetInd:
            for t in timeRangeInd:
                disDict[i,t] = {}
                for s in scenListInd:
                    # obtain the distance between the county and the center of the hurricane
                    disDict[i,t][s] = haversineEuclidean(locLatLong[locSet[i]][0],locLatLong[locSet[i]][1],\
                           trackDict[refTime][s][timeRange[t]*timeH + refTime][0],trackDict[refTime][s][timeRange[t]*timeH + refTime][1])
                disTemp = [(disDict[i,t][s],s) for s in scenListInd]
                disTemp = sorted(disTemp,key = lambda x:x[0])
                consSeq2[i,t] = m2.addConstrs(pCond2[i,j,t,disTemp[s][1]]*disTemp[s+1][0] - disTemp[s][0]*pCond2[i,j,t,disTemp[s+1][1]] >= 0\
                                        for j in levelSetInd\
                                        for s in range(len(disTemp) - 1))
        m2.setObjective(sum(sum(sum(pDiff2[i,j,t]*pDiff2[i,j,t] for t in timeRangeInd) for j in levelSetInd) for i in locSetInd),GRB.MINIMIZE)
        m2.update()
        m2.optimize()
        
        ps1Rec = []
        iterBool = True
        while iterBool:
            # record ps1
            ps1Rec.append([round(ps1[s].X,6) for s in scenListInd]) 
            # iterations, change constraints' coefficients
            for i in locSetInd:
                for j in levelSetInd:
                    for t in timeRangeInd:
                        for s in scenListInd:
                            m1.chgCoeff(consDiff1[i,j,t],ps1[s],pCond2[i,j,t,s].X)
            m1.update()
            m1.optimize()
            
            for i in locSetInd:
                for j in levelSetInd:
                    for t in timeRangeInd:
                        for s in scenListInd:
                            m2.chgCoeff(consDiff2[i,j,t],pCond2[i,j,t,s],ps1[s].X)
            m2.update()
            m2.optimize()
            if len(ps1Rec) > 1:
                ps1Prev = np.array(ps1Rec[-2])
                ps1Curr = np.array(ps1Rec[-1])
                if np.linalg.norm(ps1Prev - ps1Curr) <= 1e-4:
                    iterBool = False
        for s in scenListInd:
            pDict[scenList[s]][refTime] = ps1[s].X
        pCondDict[refTime] = {}
        for i,j,t,s in list(itertools.product(locSetInd,levelSetInd,timeRangeInd,scenListInd)):
            pCondDict[refTime][i,j,t,s] = pCond2[i,j,t,s].X
    return pDict,pCondDict

def obtainProb2(locSet,GEFSdata,GEFSdataGrid,NDFDdata,locDict,locLatLong,refTlist = [], scenList = list(range(21)),\
               timeRange = list(range(1,10)), timeH = datetime.timedelta(hours = 6), baseLatLong = (20,-88),endLatLong = (32,-70)):
    # obtain the scenario probability just to match the NDFD data
    # read in the GEFS data (county and grid) and NDFD data
    # locDict is the location dictionary mapping the index to each grid point in 2D
    # baseLatLong is the 2D origin point
    # use iterative method to compute the probability of the scenarios
    
    # obtain the reference times and list of scenarios
    if refTlist == []:
        refTlist = list(GEFSdata[0].keys())
    
    # obtain the indices of each set
    locSetInd = list(range(len(locSet)))
    timeRangeInd = list(range(len(timeRange)))
    scenListInd = list(range(len(scenList)))
        
    # for each reference time
    pDict = {}
    for s in scenList:
        pDict[s] = {}

    for refTime in refTlist:
        # the closest NDFD time that has all predictions
        NDFDref = max([iKey for iKey in NDFDdata.keys() if ((timeH + refTime) in NDFDdata[iKey].keys())])
            
        # construct the math programming model to solve for the probability of each scenario
        m1 = Model('obtain_ps')
        ps1 = m1.addVars(scenListInd,lb = 0, ub = 1)
        wDiff1 = m1.addVars(locSetInd,timeRangeInd,lb = -9999, ub = 9999)
        
        # add the constraints and the objective function
        consDiff1 = m1.addConstrs(sum(ps1[s]*GEFSdata[s][refTime][refTime + timeRange[t]*timeH][i] for s in scenListInd) + wDiff1[i,t] == NDFDdata[NDFDref][refTime + timeRange[t]*timeH][i]\
                                 for i in locSetInd\
                                 for t in timeRangeInd)
        consSum = m1.addConstr(sum(ps1[s] for s in scenListInd) == 1)
        m1.setObjective(sum(sum(wDiff1[i,t]*wDiff1[i,t] for t in timeRangeInd) for i in locSetInd),GRB.MINIMIZE)
        m1.update()
        m1.optimize()
        for s in scenListInd:
            pDict[scenList[s]][refTime] = ps1[s].X
    return pDict
    

def calProb(dataIn,fitParam,cutoff = 0.0):
    probArray = 1/(1+np.exp(-fitParam[0] - fitParam[1]*dataIn))
    for i in range(len(probArray)):
        if probArray[i] < cutoff:
            probArray[i] = 0.0
    return probArray

def mapFuel(powerloss):
    # input the power loss amount in kwh
    # output the diesel demand for the power loss in barrel
    # 14.1 kwh/gal, 42 gal/barrel
    barrelNeeded = powerloss/(14.1*42)
    return barrelNeeded

def obtainPredWind(GEFSdata,locList,refTList,fitParam,recoverParam,realDemand,cutoff = 0.0,demandDict = {},fulfillRate = 0.2,scenList = list(range(21)),recoverRate = 1/171.5880):
    # obtain the peak of the realDemand for each county
    peakDict = {}
    timeList = sorted(realDemand.keys())
    for c in locList:
        demandList = [realDemand[t][c] for t in timeList]
        peakDict[c] = timeList[demandList.index(max(demandList))]
    dDict = {}
    outPdict = {}
    for s in scenList:
        dDict[s] = {}
        outPdict[s] = {}
        for refTime in refTList:
            dDict[s][refTime] = {}
            outPdict[s][refTime] = {}
            for tp in GEFSdata['Gust'][s][refTime].keys():
                outPred = calProb(np.array(GEFSdata['Gust'][s][refTime][tp]),fitParam,cutoff)
                dPred = outPred
                dDict[s][refTime][tp] = {}
                outPdict[s][refTime][tp] = {}
                for l in range(len(locList)):
                    dDict[s][refTime][tp][locList[l]] = dPred[l]
            # for each location, identify the predicted peak wind
            for l in range(len(locList)):
                predTList = list(sorted(GEFSdata['Gust'][s][refTime].keys()))
                # if the prediction time is before the actual peak
                if refTime <= peakDict[locList[l]]:
                    probLoc = [dDict[s][refTime][tp][locList[l]] for tp in predTList]
                    probMaxInd = probLoc.index(max(probLoc))
                    maxTime = predTList[probMaxInd]
                    for tp in predTList:
                        if tp == refTime:
                            maxOut = realDemand[refTime][locList[l]]/mapFuel(demandDict[locList[l]])
                            # if it has not reached the peak yet in the GEFS prediction
                            if demandDict != {}:
                                outPdict[s][refTime][tp][locList[l]] = realDemand[refTime][locList[l]]*fulfillRate
                                #outPdict[s][refTime][tp][locList[l]] = mapFuel((maxOut*fulfillRate)*demandDict[locList[l]])
                            else:
                                outPdict[s][refTime][tp][locList[l]] = maxOut*fulfillRate
                        elif tp < maxTime:
                            maxOut = max(probLoc)
                            # if it has not reached the peak yet in the GEFS prediction
                            if demandDict != {}:
                                outPdict[s][refTime][tp][locList[l]] = mapFuel((dDict[s][refTime][tp][locList[l]]*fulfillRate)*demandDict[locList[l]])
                            else:
                                outPdict[s][refTime][tp][locList[l]] = dDict[s][refTime][tp][locList[l]]*fulfillRate
                        else:
                            # if it has reached the peak in the GEFS prediction
                            timeDiff = tp - maxTime
                            # the current outage rate
                            maxOut = max(max(probLoc),realDemand[refTime][locList[l]]/mapFuel(demandDict[locList[l]]))
                            if len(recoverParam) == 3:
                                if (maxOut**2*recoverParam[2]+maxOut*recoverParam[1]+recoverParam[0]) > 0:
                                    recoverRate = maxOut/(maxOut**2*recoverParam[2]+maxOut*recoverParam[1]+recoverParam[0])
                                else:
                                    recoverRate = 1
                            elif len(recoverParam) == 2:
                                if (maxOut*recoverParam[1]+recoverParam[0]) > 0:
                                    recoverRate = maxOut/(maxOut*recoverParam[1]+recoverParam[0])
                                else:
                                    recoverRate = 1
                            # if the ref time is after the peak shown in the realDemand
                            if demandDict != {}:
                                outPdict[s][refTime][tp][locList[l]] = mapFuel((max(maxOut - (timeDiff.total_seconds()/3600*recoverRate),0.0)*fulfillRate)*demandDict[locList[l]])
                            else:
                                outPdict[s][refTime][tp][locList[l]] = max(maxOut - (timeDiff.total_seconds()/3600*recoverRate),0.0)*fulfillRate
                # if the prediction time has passed the actual peak
                else:
                    maxTime = refTime
                    maxOut = realDemand[refTime][locList[l]]/mapFuel(demandDict[locList[l]])
                    if len(recoverParam) == 3:
                        if (maxOut**2*recoverParam[2]+maxOut*recoverParam[1]+recoverParam[0]) > 0:
                            recoverRate = maxOut/(maxOut**2*recoverParam[2]+maxOut*recoverParam[1]+recoverParam[0])
                        else:
                            recoverRate = 1
                    elif len(recoverParam) == 2:
                        if (maxOut*recoverParam[1]+recoverParam[0]) > 0:
                            recoverRate = maxOut/(maxOut*recoverParam[1]+recoverParam[0])
                        else:
                            recoverRate = 1
                    for tp in predTList:
                        timeDiff = tp - maxTime
                        # if the ref time is after the peak shown in the realDemand
                        if demandDict != {}:
                            outPdict[s][refTime][tp][locList[l]] = mapFuel((max(maxOut - (timeDiff.total_seconds()/3600*recoverRate),0.0)*fulfillRate)*demandDict[locList[l]])
                        else:
                            outPdict[s][refTime][tp][locList[l]] = max(maxOut - (timeDiff.total_seconds()/3600*recoverRate),0.0)*fulfillRate
#                outPdict[s][refTime][refTime][locList[l]] = realDemand[refTime][locList[l]]
    return outPdict,dDict

# auxiliary scripts for GEFS analysis
def headerPrint():
    # read the header of the GEFS data
    fi = open('/Users/haoxiangyang/Desktop/GEFS/GEFS_header.csv','r')
    csvReader = csv.reader(fi)
    GEFSheader = []
    dictGEFS = readGEFSheader('https://www.nco.ncep.noaa.gov/pmb/docs/on388/table2.html')
    for item in csvReader:
        if item[0] in dictGEFS.keys():
            GEFSheader.append([item[0],dictGEFS[item[0]][0],dictGEFS[item[0]][1]])
        else:
            GEFSheader.append([item[0],'',''])
            print(item[0])
    fi.close()
    fo = open('/Users/haoxiangyang/Desktop/GEFS/GEFS_header.csv','w',newline = '')
    csvWriter = csv.writer(fo,dialect = 'excel')
    csvWriter.writerows(GEFSheader)
    fo.close()
    
    
def printLocation(pntsGridLoc):
    # print the grid point list in a txt file for Florida
    # default pntsGridLoc is "../data/pntsFL_Grid.txt"
    fo = open(pntGridLoc,'w',newline = '')
    csvWriter = csv.writer(fo)
    counter = 0
    for x in range(20,32):
        for y in range(-88,-70):
            csvWriter.writerow([counter,x,y])
            counter += 1
    fo.close()
    
def locDictGen(gridPointAdd):
    fi = open(gridPointAdd,'r')
    csvReader = csv.reader(fi,dialect = 'excel')
    locDict = {}
    xList = list(range(20,32))
    yList = list(range(-88,-70))
    xyList = np.zeros([len(xList),len(yList)])
    for item in csvReader:
        xLat = int(item[1])
        yLong = int(item[2])
        xInd = xList.index(xLat)
        yInd = yList.index(yLong)
        locDict[int(item[0])] = (xInd,yInd)
    fi.close()
    return xyList,locDict
    
def heatMapGen(xyList,locDict,refTime,timeD,scenNo,T,simuPath,GEFSdataGrid):
    # demo of a heat map of a scenario
    # input:
    #   - the grid points of Florida (xyList,locDict)
    #   - the reference time
    #   - the duration of each time step
    #   - the scenario number to plot

    centerList = []
    xList = list(range(20,32))
    yList = list(range(-88,-70))
    for j in range(T):
        predictTime = refTime + timeD*j
        dataPlot = GEFSdataGrid['Gust'][scenNo][refTime][predictTime]
        for i in range(len(dataPlot)):
            xyList[locDict[i][0],locDict[i][1]] = dataPlot[i]
        
        figure, ax = plt.subplots(figsize=(15,10))
        sns.heatmap(xyList, linewidth=0.5, cmap = 'jet',vmin = 0,vmax = 100,ax = ax)
        ax.invert_yaxis()
        for item in ([ax.xaxis.label, ax.yaxis.label, ax.title]+\
                 ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(20)
        
        ax.set_xlabel('longitude')
        ax.set_yticklabels(list(reversed(xList)))
        ax.set_xticklabels(yList)
        # Make the y-axis label, ticks and tick labels match the line color.
        ax.set_ylabel('latitude')
        ax.title.set_text('{}'.format(predictTime))
        plt.show()
        figure = ax.get_figure()
    
        figure.savefig(simuPath + "Path{}_{}.png".format(scenNo,j), dpi=400)
        
        center = findCenter(xyList)
        centerList.append(center)
        print(predictTime,center)
    return centerList

def obtainCenterList(xyList,locDict,GEFSdataGrid,scenNo,refTime,timeD,T):
    centerList = []
    for i in range(T):
        predictTime = refTime + timeD*i
        dataPlot = GEFSdataGrid['Gust'][scenNo][refTime][predictTime]
        for i in range(len(dataPlot)):
            xyList[locDict[i][0],locDict[i][1]] = dataPlot[i]
        center = findCenter(xyList)
        centerList.append(center)
    return centerList
            
def tracksPlot(startT,endT,endTime,GEFSdataGrid,locDict,simuPath):
    # plot the ensemble track for a fixed time
    refTList = []
    currentT = startT
    while currentT <= endT:
        refTList.append(currentT)
        currentT += datetime.timedelta(hours = 6)
    # obtain the track centers
    baseLatLong = (20,-88)
    endLatLong = (32,-70)
    xList = list(range(baseLatLong[0],endLatLong[0]))
    yList = list(range(baseLatLong[1],endLatLong[1]))
    centerDict,trackDict = obtainCenter(GEFSdataGrid,xList,yList,refTList,locDict,'Gust')
    
    # plot the track of each scenario as time proceeds
    for s in range(21):
        fig, ax1 = plt.subplots(figsize=(7.5,5))
        plt.title(str(s))
        for t in centerDict.keys():
            xList = []
            yList = []
            for item in centerDict[t][s]:
                if (item[1] != [])and(item[0] <= endTime):
                    yList.append(item[1][0])
                    xList.append(item[1][1])
            ax1.plot(xList, yList)
            
        fig.tight_layout()
        plt.show()
        fig.savefig(os.path.join(simuPath,str(s)+".png"))
