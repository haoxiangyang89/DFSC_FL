#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 15:14:38 2019

@author: haoxiangyang
"""

# process GEFS data and obtain the probability
from gefsParser import *
#import pdb
#pdb.set_trace()

if __name__ == "__main__":
    scenList = list(range(21))
    # read the GEFS grid data to find the center of hurricane in each scenario
    GEFSdataGrid,titleGrid = pickle.load(open('/Users/haoxiangyang/Desktop/hurricane_pdata/GEFSdata_Grid.p', 'rb'))
    # obtain the list of locations to generate probability profile
    # county locations
    countyFile = open('../data/pntsFL.txt','r')
    csvReader = csv.reader(countyFile,dialect = 'excel')
    locLatLong = {}
    for item in csvReader:
        locLatLong[item[0].strip()] = (float(item[1]),float(item[2]))
    countyFile.close()

    # girdLocations
    gridFile = open('../data/pntsFL_Grid.txt','r')
    csvReader = csv.reader(gridFile,dialect = 'excel')
    locLatLong_grid = {}
    locDict = {}
    xList = list(range(20,32))
    yList = list(range(-88,-70))
    xyList = np.zeros([len(xList),len(yList)])
    for item in csvReader:
        locLatLong_grid[item[0].strip()] = (float(item[1]),float(item[2]))
        xLat = int(item[1])
        yLong = int(item[2])
        xInd = xList.index(xLat)
        yInd = yList.index(yLong)
        locDict[int(item[0])] = (xInd,yInd)
    gridFile.close()

    # filter the grid data that is in the southeast NDFD database
    windProbGrid,windLocGrid,windLocGridInd = pickle.load(open('/Users/haoxiangyang/Desktop/hurricane_pdata/windProbGrid.p', 'rb'))

    # read the GEFS data for scenario construction
    GEFSdataPart,titlePart = pickle.load(open('/Users/haoxiangyang/Desktop/hurricane_pdata/GEFSdata_Part.p', 'rb'))
    GEFSdataCounty,titleCounty = pickle.load(open('/Users/haoxiangyang/Desktop/hurricane_pdata/GEFSdata_County.p', 'rb'))

    gustNDFD,gustLoc = pickle.load(open('../data/gustNDFD.p',"rb"))
    windProb,windLoc,windLocInd = pickle.load(open('../data/windProb.p', 'rb'))
    #pDict,pCondDict = obtainProb(windLocGrid,GEFSdataPart,GEFSdataGrid,windProbGrid,locDict,locLatLong_grid,refTList)
    startT = datetime.datetime(2017,9,6,0,0)
    endT = datetime.datetime(2017,9,24,18,0)
    refTList = []
    currentT = startT
    while currentT <= endT:
        refTList.append(currentT)
        currentT += datetime.timedelta(hours = 6)

    pDict = obtainProb2(windLoc,GEFSdataCounty['Gust'],GEFSdataGrid['Gust'],gustNDFD['WindGust'],locDict,locLatLong_grid,refTList)

    with open('../data/probScen.p', 'wb') as fp:
        pickle.dump([pDict,pCondDict], fp, protocol=pickle.HIGHEST_PROTOCOL)
    
    # print the solution
    for s in scenList:
        print("------------- Scenario {} -------------".format(s))
        for t in refTList:
            print(pDict[s][t])

    # dDict is the predicted outage percentage of each county just based on the gust speed
    # outPdict is the predicted outage percentage of each county based on the gust speed + maintenance schedule
#    outPdict,dDict = obtainPredWind(GEFSdataCounty,titleCounty,refTList,fitParam,recoverParam[1],realDemand,cutoffVal,demandElectricity)
#    with open('/Users/haoxiangyang/Desktop/Git/daniel_Diesel/data/outPredPerc.p', 'wb') as fp:
#        pickle.dump(dDict, fp, protocol=pickle.HIGHEST_PROTOCOL)
#    with open('/Users/haoxiangyang/Desktop/Git/daniel_Diesel/data/predDemand_20.p', 'wb') as fp:
#        pickle.dump(outPdict, fp, protocol=pickle.HIGHEST_PROTOCOL)
    