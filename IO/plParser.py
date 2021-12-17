#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 16:01:34 2019

@author: haoxiangyang
"""
# this is the power loss parser for diesel supply chain analysis

import os
import csv
import datetime
import re

class plData:
    def __init__(self,location,time,customer,out,outP):
        self.nc = customer
        self.out = out
        self.op = outP
        self.loc = location
        self.t = time
    
def plParser(fileAdd):
    # parse the power loss data
    fi = open(fileAdd,"r")
    csvReader = csv.reader(fi,dialect = "excel")
    counter = 0
    dataRaw = []
    for item in csvReader:
        if counter == 0:
            counter += 1
        else:
            dataRaw.append(item)
    fi.close()
    
    data = []
    for item in dataRaw:
        entryTraw = item[-1]
        entryTLraw = re.split(" |/|:",entryTraw)
        yearE = 2000+int(entryTLraw[2])
        monthE = int(entryTLraw[0])
        dayE = int(entryTLraw[1])
        hourE = int(entryTLraw[3])
        minE = int(entryTLraw[4])
        entryT = datetime.datetime(yearE,monthE,dayE,hourE,minE)
        
        customer = int(item[-5])
        out = int(item[-4])
        outP = out/customer
        location = item[-6]
        data.append(plData(location,entryT,customer,out,outP))
    return data

def plPDFAnalyzer(countyList,year,fileAdd):
    # categorize the data by county
    # read in each txt file
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
                totalData[reportT][c] = [int(countyDList[-4].replace(',','')),int(countyDList[-3].replace(',',''))]
    return totalData
