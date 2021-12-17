#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 23:00:38 2018

@author: haoxiangyang
"""
import re
import pickle
import urllib
import csv


#%%
def countyListGen(stateAdd):
    countyIds = {}
    ctxtFile = urllib.request.urlopen(stateAdd)
    ctxt = ctxtFile.read()
    ctxt = ctxt.decode()
    
    rawData = re.findall('"CountyId":([0-9]+),"UtilityId":null,"CountyName":"([A-Za-z\.\ \-]+)".+?"CustomerCount":([0-9]+)}',ctxt)
    for item in rawData:
        countyId = item[0]
        countyName = item[1]
        if (not(countyName.isnumeric()))and(countyName != 'Unknown'):
            countyIds[countyId] = countyName
    return countyIds

#%%
countyIds = dict()
stateIds = dict()
# here we aquire the county ids into a dictionary for scraping in get_powerdat()
fi = open("/Users/haoxiangyang/Desktop/Git/DieselSC_DisasterResponse/IO/state_links.csv","r")
csvReader = csv.reader(fi)
stateList = []
stateCountyDict = {}
for item in csvReader:
    stateName = item[0]
    if stateName == "North Carolina":
        stateName = "North_Carolina"
    if stateName == "South Carolina":
        stateName = "South_Carolina"
    stateAdd = item[1]
    stateCountyDict[stateName] = countyListGen(stateAdd)
    stateList.append(stateName)
fi.close()

# identify duplicate county keys, there is none
countyDictTotal = {}
for stateKey in stateList:
    for countyKey in stateCountyDict[stateKey]:
        if countyKey in countyDictTotal.keys():
            print(countyKey)
        else:
            countyDictTotal[countyKey] = (stateCountyDict[stateKey][countyKey],stateKey)
with open('../data/stateCountyID.p', 'wb') as fp:
    pickle.dump([stateCountyDict,countyDictTotal], fp, protocol=pickle.HIGHEST_PROTOCOL)