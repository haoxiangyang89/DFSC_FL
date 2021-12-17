#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 13:11:21 2018

@author: haoxiangyang
"""

# This is the CAOE module for:
#   - reading the power loss data from txt files (done)
#   - reading the LCD observation data from csv files (done)
#   - reading the HWM observation data from a csv file (done)
#   - reading the NDFD forecast data (done)
#   - reading the real time wind/flood data (for future implementation of the response tool)
#   - reading the P-Surge forecast data
#   - map the outage to fuel demand
#   - generate demand for simulation/optimization use

import os
os.chdir("/Users/haoxiangyang/Desktop/Git/DFSC_FLORIDA/IO")
import csv
import datetime
import re
import numpy as np
from sklearn import linear_model
import pickle

def plPDFAnalyzer(countyList,year,fileAdd):
    # readin the Irma outage data in pdf format for mapping the demand relationship
    # input countyList: the list of counties
    #       year: the year of hurricane
    #       fileAdd: the path of all txt files translated from pdf files 
    # output totalData[datetime][county name]: power outage dictionary
    totalData = {}
    fileList = os.listdir(fileAdd)
    for fA in fileList:
        # if it is a txt file
        if ".txt" in fA:
            fullfA = os.path.join(fileAdd,fA)
            fi = open(fullfA,"r")
            rawStr = fi.read()
            fi.close()
            # extract the report time
            reportTraw = re.findall("([0-9]+)/([0-9]+)/([0-9]+)\ ([0-9]+)\:([0-9]+)",rawStr)[0]
            reportT = datetime.datetime(int(reportTraw[2]),int(reportTraw[0]),int(reportTraw[1]),int(reportTraw[3]),int(reportTraw[4]))
            totalData[reportT] = {}
            for c in countyList:
                # extract the county level power loss data
                pattern = c + "(.+)\n"
                countyDraw = re.findall(pattern,rawStr)[0]
                countyDList = countyDraw.split(" ")
                if c == 'DESOTO':
                    cTrans = 'DeSoto County'
                else:
                    cTrans = c.title() + ' County'
                totalData[reportT][cTrans] = [int(countyDList[-4].replace(',','')),int(countyDList[-3].replace(',',''))]
    return totalData

def lcdParser(fileAdd,loc):
    # read in the lcd csv file
    fileName = loc + "_Irma_LCD.csv"
    fi = open(os.path.join(fileAdd,fileName),"r")
    csvReader = csv.reader(fi)
    counter = 0
    rawData = []
    for item in csvReader:
        if counter == 0:
            counter += 1
            title = item
        else:
            rawData.append(item)
    fi.close()
    # for each data entry, extract useful information and store them in an array
    dataW = []
    for item in rawData:
        dateRaw = item[title.index("DATE")]
        reportTraw = re.findall("([0-9]+)-([0-9]+)-([0-9]+)\ ([0-9]+)\:([0-9]+)",dateRaw)[0]
        reportT = datetime.datetime(int(reportTraw[0]),int(reportTraw[1]),int(reportTraw[2]),int(reportTraw[3]),int(reportTraw[4]))
        windSRaw = item[title.index("HOURLYWindSpeed")]
        if windSRaw != '':
            windS = int(windSRaw)
        else:
            windS = None
        gustSRaw = item[title.index("HOURLYWindGustSpeed")]
        if gustSRaw != '':
            gustS = int(gustSRaw)
        else:
            gustS = None
        dataW.append((reportT,windS,gustS))
    return dataW

def lcdTotalParser(fileAdd,countyDict):
    # read in the lcd wind data for mapping the demand relationship
    # input fileAdd: address of the total lcd data csv file
    #       countyDict: the dictionary to map between county and its lcd code
    # output dataDict[county name]: a dictionary with lcd wind observations
    fi = open(fileAdd,"r")
    csvReader = csv.reader(fi)
    counter = 0
    rawData = []
    for item in csvReader:
        if counter == 0:
            counter += 1
            title = item
        else:
            rawData.append(item)
    fi.close()
    
    dataDict = {}
    # obtain county-specific data
    for ckey in countyDict.keys():
        rawDataC = []
        for item in rawData:
            if item[0] == "WBAN:" + ckey:
                rawDataC.append(item)
        # for each data entry, extract useful information and store them in an array
        dataW = []
        for item in rawDataC:
            dateRaw = item[title.index("DATE")]
            reportTraw = re.findall("([0-9]+)/([0-9]+)/([0-9]+)\ ([0-9]+)\:([0-9]+)",dateRaw)[0]
            reportT = datetime.datetime(int(reportTraw[2])+2000,int(reportTraw[0]),int(reportTraw[1]),int(reportTraw[3]),int(reportTraw[4]))
            windSRaw = item[title.index("HOURLYWindSpeed")]
            if windSRaw != '':
                windS = int(windSRaw)
            else:
                windS = None
            gustSRaw = item[title.index("HOURLYWindGustSpeed")]
            if gustSRaw != '':
                gustS = int(gustSRaw)
            else:
                gustS = None
            dataW.append((reportT,windS,gustS))
        dataDict[countyDict[ckey]] = dataW
        
    # return a dictionary of spectral wind data
    return dataDict

def hwmReader(HWMAdd,countyList):
    # read in the high water mark data for mapping the demand relationship
    # input HWMAdd: address of the high water mark csv file
    # output hwmPeakData[county name]: the county
    fih = open(HWMAdd,"r")
    csvReaderh = csv.reader(fih)
    
    counter = 0
    datah = []
    for item in csvReaderh:
        if counter == 0:
            titleh = item
            counter += 1
        else:
            if item[titleh.index("stateName")] == "FL":
                datah.append(item)
    fih.close()
    
    # create empty data dictionaries for high water mark and peak
    hwmData = {}
    datumElev = {}
    for c in countyList:
        hwmData[c] = [0.0]
        datumElev[c] = []
        
    # append the high water mark data to each county
    countyInd = titleh.index("countyName")
    datumInd = titleh.index("verticalDatumName")
    dataInd = titleh.index("height_above_gnd")
    datumDInd = titleh.index("elev_ft")
    for item in datah:
        c = item[countyInd]
        # only use NAVD88 datum
        if (item[datumDInd] != '')and(item[dataInd] != '')and(item[datumInd] == 'NAVD88'):
            hwmData[c].append(float(item[dataInd]))
            datumElev[c].append(float(item[datumDInd]) - float(item[dataInd]))
            
    hwmPeakData = {}
    for c in countyList:
        hwmPeakData[c] = max(hwmData[c])
    
    # return a dictionary of surge peak data
    return hwmPeakData

def ndfdWindReader(fileAdd,countyList,isPath = True):
    # read the NDFD data for future forecast
    # input fileAdd: the file address
    #       countyList: the list of counties
    #       isPath: whether fileAdd is a folder and we are reading every file in the directory
    # output windNDFD: dictionary of NDFD wind speed forecast
    if isPath:
        windFileList = os.listdir(fileAdd)
        windNDFD = {}
        for fA in windFileList:
            # if it is a txt file
            if ".txt" in fA:
                fullfA = os.path.join(fileAdd,fA) 
                fi = open(fullfA,"r")
                rawStr = fi.read()
                fi.close()
                dataRaw = re.findall("(.+)\n",rawStr)
                title = dataRaw[0].split(',')
                for i in range(len(title)):
                    title[i] = title[i].strip()
                # time point and predicted time point
                tpStr = dataRaw[1].split(',')[2].strip()
                tp = datetime.datetime(int(tpStr[0:4]),int(tpStr[4:6]),int(tpStr[6:8]),int(tpStr[8:10]),int(tpStr[10:12]))
                windNDFD[tp] = {}
                for l in range(1,len(dataRaw)):
                    ptpStr = dataRaw[l].split(',')[3].strip()
                    ptp = datetime.datetime(int(ptpStr[0:4]),int(ptpStr[4:6]),int(ptpStr[6:8]),int(ptpStr[8:10]),int(ptpStr[10:12]))
                    windNDFD[tp][ptp] = {}
                    preData = dataRaw[l].split(',')
                    for c in countyList:
                        cLoc = title.index(c)
                        if float(preData[cLoc].strip()) <= 1000.0:
                            windNDFD[tp][ptp][c] = float(preData[cLoc].strip())*1.15078
    else:
        windNDFD = {}
        fullfA = fileAdd
        fi = open(fullfA,"r")
        rawStr = fi.read()
        fi.close()
        dataRaw = re.findall("(.+)\n",rawStr)
        title = dataRaw[0].split(',')
        for i in range(len(title)):
            title[i] = title[i].strip()
        # time point and predicted time point
        tpStr = dataRaw[1].split(',')[2].strip()
        tp = datetime.datetime(int(tpStr[0:4]),int(tpStr[4:6]),int(tpStr[6:8]),int(tpStr[8:10]),int(tpStr[10:12]))
        windNDFD[tp] = {}
        for l in range(1,len(dataRaw)):
            ptpStr = dataRaw[l].split(',')[3].strip()
            ptp = datetime.datetime(int(ptpStr[0:4]),int(ptpStr[4:6]),int(ptpStr[6:8]),int(ptpStr[8:10]),int(ptpStr[10:12]))
            windNDFD[tp][ptp] = {}
            preData = dataRaw[l].split(',')
            for c in countyList:
                cLoc = title.index(c)
                if float(preData[cLoc].strip()) <= 1000.0:
                    windNDFD[tp][ptp][c] = float(preData[cLoc].strip())*1.15078    
    return windNDFD

def ndfdWindReader_old(fileAdd,countyCode):
    # read the NDFD data
    windFileList = os.listdir(fileAdd)
    windNDFD = {}
    for fA in windFileList:
        # if it is a txt file
        if ".txt" in fA:
            fullfA = os.path.join(fileAdd,fA) 
            fi = open(fullfA,"r")
            rawStr = fi.read()
            fi.close()
            dataRaw = re.findall("(.+)\n",rawStr)
            title = dataRaw[0].split(',')
            for i in range(len(title)):
                title[i] = title[i].strip()
            # time point and predicted time point
            tpStr = dataRaw[1].split(',')[2].strip()
            tp = datetime.datetime(int(tpStr[0:4]),int(tpStr[4:6]),int(tpStr[6:8]),int(tpStr[8:10]),int(tpStr[10:12]))
            windNDFD[tp] = {}
            for l in range(1,len(dataRaw)):
                ptpStr = dataRaw[l].split(',')[3].strip()
                ptp = datetime.datetime(int(ptpStr[0:4]),int(ptpStr[4:6]),int(ptpStr[6:8]),int(ptpStr[8:10]),int(ptpStr[10:12]))
                windNDFD[tp][ptp] = {}
                preData = dataRaw[l].split(',')
                for c in countyCode.keys():
                    cLoc = title.index("WBAN:"+c)
                    if float(preData[cLoc].strip()) <= 1000.0:
                        windNDFD[tp][ptp][countyCode[c]] = float(preData[cLoc].strip())*1.15078
                
    return windNDFD


def pSurgeReader(fileAdd,countyList,pSurge = {},isPath = True,gtx = 0):
    # read the pSurge data for future forecast
    # input fileAdd: the file address
    #       countyList: the list of counties
    #       pSurge: previously stored pSurge data
    #       isPath: whether fileAdd is a folder and we are reading every file in the directory
    #       gtx: readin greater than x ft probability
    # output pSurge: dictionary of pSurge forecast
    if isPath:
        psurgeFileList = os.listdir(fileAdd)
        for fA in psurgeFileList:
            # if it is a txt file
            if ".txt" in fA:
                fullfA = os.path.join(fileAdd,fA) 
                fi = open(fullfA,"r")
                rawStr = fi.read()
                fi.close()
                dataRaw = re.findall("(.+)\n",rawStr)
                title = dataRaw[0].split(',')
                gtxRaw = dataRaw[1].split(',')[0].strip()
                gtx = int(re.findall("([0-9]+)",gtxRaw)[0])
                for i in range(len(title)):
                    title[i] = title[i].strip()
                # time point and predicted time point
                tpStr = dataRaw[1].split(',')[2].strip()
                tp = datetime.datetime(int(tpStr[0:4]),int(tpStr[4:6]),int(tpStr[6:8]),int(tpStr[8:10]),int(tpStr[10:12]))
                if not(tp in pSurge.keys()):
                    pSurge[tp] = {}
                for l in range(1,len(dataRaw)):
                    ptpStr = dataRaw[l].split(',')[3].strip()
                    ptp = datetime.datetime(int(ptpStr[0:4]),int(ptpStr[4:6]),int(ptpStr[6:8]),int(ptpStr[8:10]),int(ptpStr[10:12]))
                    if not(ptp in pSurge[tp].keys()):
                        pSurge[tp][ptp] = {}
                    preData = dataRaw[l].split(',')
                    for c in countyList:
                        cLoc = title.index(c)
                        if not(c in pSurge[tp][ptp].keys()):
                            pSurge[tp][ptp][c] = {}
                        if float(preData[cLoc].strip()) <= 1000.0:
                            pSurge[tp][ptp][c][gtx] = float(preData[cLoc].strip())/100
                        else:
                            pSurge[tp][ptp][c][gtx] = 0.0
    else:
        fullfA = fileAdd
        fi = open(fullfA,"r")
        rawStr = fi.read()
        fi.close()
        dataRaw = re.findall("(.+)\n",rawStr)
        title = dataRaw[0].split(',')
        for i in range(len(title)):
            title[i] = title[i].strip()
        # time point and predicted time point
        tpStr = dataRaw[1].split(',')[2].strip()
        tp = datetime.datetime(int(tpStr[0:4]),int(tpStr[4:6]),int(tpStr[6:8]),int(tpStr[8:10]),int(tpStr[10:12]))
        if not(tp in pSurge.keys()):
            pSurge[tp] = {}
        for l in range(1,len(dataRaw)):
            ptpStr = dataRaw[l].split(',')[3].strip()
            ptp = datetime.datetime(int(ptpStr[0:4]),int(ptpStr[4:6]),int(ptpStr[6:8]),int(ptpStr[8:10]),int(ptpStr[10:12]))
            if not(ptp in pSurge[tp].keys()):
                pSurge[tp][ptp] = {}
            preData = dataRaw[l].split(',')
            for c in countyList:
                cLoc = title.index(c)
                if not(c in pSurge[tp][ptp].keys()):
                    pSurge[tp][ptp][c] = {}
                if float(preData[cLoc].strip()) <= 1000.0:
                    pSurge[tp][ptp][c][gtx] = float(preData[cLoc].strip())/100
                else:
                    pSurge[tp][ptp][c][gtx] = 0.0
    return pSurge

def usgsReader(HWMAdd,countyList):
    # read the HWM data for mapping the relationship
    fih = open(HWMAdd,"r")
    csvReaderh = csv.reader(fih)
    
    counter = 0
    datah = []
    for item in csvReaderh:
        if counter == 0:
            titleh = item
            counter += 1
        else:
            if item[titleh.index("stateName")] == "FL":
                datah.append(item)
    fih.close()
    
    # create empty data dictionaries for high water mark and peak
    hwmData = {}
    datumElev = {}
    for c in countyList:
        hwmData[c] = [0.0]
        datumElev[c] = []
        
    # append the high water mark data to each county
    countyInd = titleh.index("countyName")
    datumInd = titleh.index("verticalDatumName")
    dataInd = titleh.index("height_above_gnd")
    datumDInd = titleh.index("elev_ft")
    for item in datah:
        c = item[countyInd].replace(" County","").upper()
        if c == 'DE SOTO':
            c = 'DESOTO'
        # only use NAVD88 datum
        if (item[datumDInd] != '')and(item[dataInd] != '')and(item[datumInd] == 'NAVD88'):
            hwmData[c].append(float(item[dataInd]))
            datumElev[c].append(float(item[datumDInd]) - float(item[dataInd]))
            
    hwmPeakData = {}
    for c in countyList:
        hwmPeakData[c] = max(hwmData[c])
        
    return hwmData

def mapFuel(powerloss):
    # input the power loss amount in kwh
    # output the diesel demand for the power loss in barrel
    # 14.1 kwh/gal, 42 gal/barrel
    barrelNeeded = powerloss/(14.1*42)
    return barrelNeeded

def forecastDemand(countyList,fl_df,ndfdData,psurgeData,timeCurrent,timePredicted,mapFunc,dieCoeff,totalDemand,fulfillRate = 0.2):
    # combine weather data and mapping to obtain predicted diesel demand
    # input ndfdData: NDFD dictionary
    #       psurgeData: pSurge dictionary
    #       timePredicted: the datetime variable for the prediction time
    #       mapping: the logistic regression obtained from mapping process
    # output predictedData: a dictionary that contains the demand for each county
    timeNDFDUsed = max([i for i in ndfdData.keys() if i <= timeCurrent])
    predictedData = {}
    for tp in timePredicted:
        predictedData[tp] = {}
    windList = {}
    for c in countyList:
        windList[c] = []
        # for each time period predicted
        for tp in timePredicted:
            if not(c in ndfdData[timeNDFDUsed][tp].keys()):
                findTp = False
                timeReverse = sorted(ndfdData.keys(),reverse = True)
                for tpMade in timeReverse:
                    for tpt in ndfdData[tpMade].keys():
                        if (not(findTp))and(c in ndfdData[tpMade][tpt].keys()):
                            findTp = True
                            windPre = ndfdData[tpMade][tpt][c]
                            windList[c].append(windPre)
            else:
                windPre = ndfdData[timeNDFDUsed][tp][c]
                windList[c].append(windPre)
    # prediction follows the wind speed mapping before the peak and follows a linear
    # recovery rule after the peak
    demandElectricity = {}
    for c in countyList:
        demandElectricity[c] = fl_df.Population[c]/sum(fl_df.Population)*totalDemand
        peakIndex = windList[c].index(max(windList[c]))
        peakOut = 1/(1 + np.exp(-(mapFunc[0] + mapFunc[1]*windList[c][peakIndex])))
        for tpInd in range(peakIndex + 1):
            pPerc = 1/(1 + np.exp(-(mapFunc[0] + mapFunc[1]*windList[c][tpInd])))*fulfillRate
            predictedData[timePredicted[tpInd]][c] = mapFuel(pPerc*demandElectricity[c])
        for tpInd in range(peakIndex + 1,len(timePredicted)):
            pPerc = max(peakOut - dieCoeff/100*(tpInd - peakIndex),0.0)*fulfillRate
            predictedData[timePredicted[tpInd]][c] = mapFuel(pPerc*demandElectricity[c])
    return predictedData

def realDemandt(countyList,totalData,fl_df,timeCurrent,totalDemand,fulfillRate = 1):
    realDemand = {}
    demandElectricity = {}
    for c in countyList:
        demandElectricity[c] = fl_df.Population[c]/sum(fl_df.Population)*totalDemand
    if timeCurrent in totalData.keys():
        for c in countyList:
            pPerc = totalData[timeCurrent][c][0]/totalData[timeCurrent][c][1]*fulfillRate
            realDemand[c] = mapFuel(pPerc*demandElectricity[c])
    else:
        # search for the closest demand if not listed in the outage data
        afterSet = [i for i in totalData if i > timeCurrent]
        beforeSet = [i for i in totalData if i < timeCurrent]
        if (afterSet != [])and(beforeSet != []):
            firstAfter = min(afterSet)
            lastBefore = max(beforeSet)
            tAfter = firstAfter - timeCurrent
            tBefore = timeCurrent - lastBefore
            ratio = tAfter.total_seconds()/(tAfter.total_seconds() + tBefore.total_seconds())
            for c in countyList:
                pPercA = totalData[firstAfter][c][0]/totalData[firstAfter][c][1]
                pPercB = totalData[lastBefore][c][0]/totalData[lastBefore][c][1]
                pPerc = (pPercB*ratio + pPercA*(1 - ratio))*fulfillRate
                realDemand[c] = mapFuel(pPerc*demandElectricity[c])
        else:
            # if there is no closest demand list, assume zero loss
            for c in countyList:
                realDemand[c] = 0.0
    return realDemand