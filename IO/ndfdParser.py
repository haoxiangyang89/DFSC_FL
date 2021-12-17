#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 16:16:40 2019

@author: haoxiangyang
"""

import os
import csv
import datetime
import numpy as np

# import pdb
# pdb.set_trace()

def mapFuel(powerloss):
    # input the power loss amount in kwh
    # output the diesel demand for the power loss in barrel
    # 14.1 kwh/gal, 42 gal/barrel
    barrelNeeded = powerloss/(14.1*42)
    return barrelNeeded


def calProb(dataIn,fitParam,cutoff = 0.0):
    probArray = 1/(1+np.exp(-fitParam[0] - fitParam[1]*dataIn))
    for i in range(len(probArray)):
        if probArray[i] < cutoff:
            probArray[i] = 0.0
    return probArray

def convTime(tString):
    tString = tString.strip()
    tYear = int(tString[0:4])
    tMonth = int(tString[4:6])
    tDay = int(tString[6:8])
    tHour = int(tString[8:10])
    return datetime.datetime(tYear,tMonth,tDay,tHour)

# parse NDFD-format data
def extractNDFD(filePath):
    # extract the entire set of NDFD-format data
    # can only process one geographic region
    listFile = os.listdir(filePath)
    dataProc = {}
    for itemR in listFile:
        fileAdd = os.path.join(filePath,itemR)
        fi = open(fileAdd,'r')
        csvReader = csv.reader(fi)
        counter = 0
        rawData = []
        try:
            for rowD in csvReader:
                if counter == 0:
                    counter += 1
                    title = rowD
                else:
                    rawData.append(rowD)
            fi.close()
            locList = [i.strip() for i in title[4:]]
            for item in rawData:
                dataName = item[0]
                refTime = convTime(item[2])
                predTime = convTime(item[3])
                dataEntry = [float(i.strip()) for i in item[4:]]
                if dataName in dataProc.keys():
                    if refTime in dataProc[dataName].keys():
                        dataProc[dataName][refTime][predTime] = dataEntry
                    else:
                        dataProc[dataName][refTime] = {}
                        dataProc[dataName][refTime][predTime] = dataEntry
                else:
                    dataProc[dataName] = {}
                    dataProc[dataName][refTime] = {}
                    dataProc[dataName][refTime][predTime] = dataEntry
        except:
            print(itemR)
                
    return dataProc,locList

def obtainPredNDFD(NDFDdata,locList,refTList,fitParam,recoverParam,realDemand,cutoff = 0.0,demandDict = {},fulfillRate = 0.2,recoverRate=1/171.5880):
    # obtain the peak of the realDemand for each county
    peakDict = {}
    timeList = sorted(realDemand.keys())
    for c in locList:
        demandList = [realDemand[t][c] for t in timeList]
        peakDict[c] = timeList[demandList.index(max(demandList))]
    dDict = {}
    outPdict = {}
    for refTime in refTList:
        dDict[refTime] = {}
        outPdict[refTime] = {}
        
        if refTime in NDFDdata['WindGust'].keys():
            # identify the predicted time
            predTList = [refTime]
            for predTime in NDFDdata['WindGust'][refTime].keys():
                if predTime.hour in [0,6,12,18]:
                    if not(predTime in predTList):
                        predTList.append(predTime)
                
            for tp in predTList:
                if tp == refTime:
                    outPred = np.zeros(len(locList))
                    dPred = outPred
                    dDict[refTime][tp] = {}
                    outPdict[refTime][tp] = {}
                    for l in range(len(locList)):
                        dDict[refTime][tp][locList[l]] = dPred[l]
                else:
                    outPred = calProb(np.array(NDFDdata['WindGust'][refTime][tp]),fitParam,cutoff)
                    dPred = outPred
                    dDict[refTime][tp] = {}
                    outPdict[refTime][tp] = {}
                    for l in range(len(locList)):
                        dDict[refTime][tp][locList[l]] = dPred[l]
        else:
            # find the two closest time, +/- 1 hour
            if refTime + datetime.timedelta(hours = 1) in NDFDdata['WindGust'].keys():
                refTimeNew = refTime + datetime.timedelta(hours = 1)
                predTList = [refTime]
                for predTime in NDFDdata['WindGust'][refTimeNew].keys():
                    if predTime.hour in [0,6,12,18]:
                        if not(predTime in predTList):
                            predTList.append(predTime)
                            
                for tp in predTList:
                    if tp == refTime:
                        outPred = np.zeros(len(locList))
                        dPred = outPred
                        dDict[refTime][tp] = {}
                        outPdict[refTime][tp] = {}
                        for l in range(len(locList)):
                            dDict[refTime][tp][locList[l]] = dPred[l]
                    else:
                        outPred = calProb(np.array(NDFDdata['WindGust'][refTimeNew][tp]),fitParam,cutoff)
                        dPred = outPred
                        dDict[refTime][tp] = {}
                        outPdict[refTime][tp] = {}
                        for l in range(len(locList)):
                            dDict[refTime][tp][locList[l]] = dPred[l]
            elif refTime - datetime.timedelta(hours = 1) in NDFDdata['WindGust'].keys():
                refTimeNew = refTime - datetime.timedelta(hours = 1)
                predTList = [refTime]
                for predTime in NDFDdata['WindGust'][refTimeNew].keys():
                    if predTime.hour in [0,6,12,18]:
                        if not(predTime in predTList):
                            predTList.append(predTime)
                        
                for tp in predTList:
                    if tp == refTime:
                        outPred = np.zeros(len(locList))
                        dPred = outPred
                        dDict[refTime][tp] = {}
                        outPdict[refTime][tp] = {}
                        for l in range(len(locList)):
                            dDict[refTime][tp][locList[l]] = dPred[l]
                    else:
                        outPred = calProb(np.array(NDFDdata['WindGust'][refTimeNew][tp]),fitParam,cutoff)
                        dPred = outPred
                        dDict[refTime][tp] = {}
                        outPdict[refTime][tp] = {}
                        for l in range(len(locList)):
                            dDict[refTime][tp][locList[l]] = dPred[l]
            elif refTime - datetime.timedelta(hours = 2) in NDFDdata['WindGust'].keys():
                refTimeNew = refTime - datetime.timedelta(hours = 2)
                predTList = [refTime]
                for predTime in NDFDdata['WindGust'][refTimeNew].keys():
                    if predTime.hour in [0,6,12,18]:
                        if not(predTime in predTList):
                            predTList.append(predTime)
                        
                for tp in predTList:
                    if tp == refTime:
                        outPred = np.zeros(len(locList))
                        dPred = outPred
                        dDict[refTime][tp] = {}
                        outPdict[refTime][tp] = {}
                        for l in range(len(locList)):
                            dDict[refTime][tp][locList[l]] = dPred[l]
                    else:
                        outPred = calProb(np.array(NDFDdata['WindGust'][refTimeNew][tp]),fitParam,cutoff)
                        dPred = outPred
                        dDict[refTime][tp] = {}
                        outPdict[refTime][tp] = {}
                        for l in range(len(locList)):
                            dDict[refTime][tp][locList[l]] = dPred[l]
                
        # for each location, identify the predicted peak wind
        for l in range(len(locList)):
            if refTime <= peakDict[locList[l]]:
                probLoc = [dDict[refTime][tp][locList[l]] for tp in predTList]
                probMaxInd = probLoc.index(max(probLoc))
                maxTime = predTList[probMaxInd]
                for tp in predTList:
                    if tp == refTime:
                        maxOut = realDemand[refTime][locList[l]]/mapFuel(demandDict[locList[l]])
                        # if it has not reached the peak yet in the N prediction
                        if demandDict != {}:
                            outPdict[refTime][tp][locList[l]] = realDemand[refTime][locList[l]]*fulfillRate
                            #outPdict[refTime][tp][locList[l]] = mapFuel((maxOut*fulfillRate)*demandDict[locList[l]])
                        else:
                            outPdict[refTime][tp][locList[l]] = maxOut*fulfillRate
                    elif tp < maxTime:
                        maxOut = max(probLoc)
                        # if it has not reached the peak yet in the NDFD prediction
                        if demandDict != {}:
                            outPdict[refTime][tp][locList[l]] = mapFuel((dDict[refTime][tp][locList[l]]*fulfillRate)*demandDict[locList[l]])
                        else:
                            outPdict[refTime][tp][locList[l]] = dDict[refTime][tp][locList[l]]*fulfillRate
                    else:
                        # if it has reached the peak in the NDFD prediction
                        timeDiff = tp - maxTime
                        # the current outage rate
                        maxOut = max(max(probLoc),realDemand[refTime][locList[l]]/mapFuel(demandDict[locList[l]]))
                        #maxOut = max(max(probLoc),max(realDemand[refTime][locList[l]]/mapFuel(demandDict[locList[l]]) - 1,0.0))
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
                            outPdict[refTime][tp][locList[l]] = mapFuel((max(maxOut - (timeDiff.total_seconds()/(3600*recoverRate)),0.0)*fulfillRate)*demandDict[locList[l]])
                        else:
                            outPdict[refTime][tp][locList[l]] = max(maxOut - (timeDiff.total_seconds()/(3600*recoverRate)),0.0)*fulfillRate
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
#                        print(l,refTime,tp)
                        outPdict[refTime][tp][locList[l]] = mapFuel((max(maxOut - (timeDiff.total_seconds()/(3600*recoverRate)),0.0)*fulfillRate)*demandDict[locList[l]])
                    else:
                        outPdict[refTime][tp][locList[l]] = max(maxOut - (timeDiff.total_seconds()/(3600*recoverRate)),0.0)*fulfillRate
#                outPdict[refTime][refTime][locList[l]] = realDemand[refTime][locList[l]]
    return outPdict,dDict