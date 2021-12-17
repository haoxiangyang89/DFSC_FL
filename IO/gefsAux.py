#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 15:18:21 2019

@author: haoxiangyang
"""
from gefsParser import *
#import pdb
#pdb.set_trace()

if __name__ == "__main__":
    refTime = datetime.datetime(2017,9,7,0,0)
    timeD = datetime.timedelta(hours = 6)
    # path subject to change
    simuPath = "/Users/haoxiangyang/Dropbox/Research_Documents/Hurricane/Data/GEFS_Simu/"
    xyList,locDict = locDictGen('../data/pntsFL_Grid.txt')
    scenNo = 14
    T = 30
    # generate heatmaps of wind speed progress for a certain scenario
    GEFSdataGrid,titleGrid = pickle.load(open("/Users/haoxiangyang/Desktop/hurricane_pData/GEFSdata_Grid.p","rb"))
    
    centerList = heatMapGen(xyList,locDict,refTime,timeD,scenNo,T,simuPath,GEFSdataGrid)
    centerList = obtainCenterList(xyList,locDict,GEFSdataGrid,scenNo,refTime,timeD,T)
    
    # print tracks between certain time points
    endTime = datetime.datetime(2017,9,12,18,0)
    startT = datetime.datetime(2017,9,6,0,0)
    endT = datetime.datetime(2017,9,14,18,0)
    tracksPlot(startT,endT,endTime,GEFSdataGrid,locDict,simuPath)

#            
#
#threeScenDict = {}
#for plotTime in refTList:
#    fig, ax1 = plt.subplots(figsize=(15,10))
#    ax1.set_title(str(plotTime))
#    
#    for item in ([ax1.xaxis.label, ax1.yaxis.label, ax1.title] +\
#             ax1.get_xticklabels() + ax1.get_yticklabels()):
#        item.set_fontsize(20)
#
#    ax1.set_xlabel('Longitude')
#    # Make the y-axis label, ticks and tick labels match the line color.
#    axC.set_ylabel('Latitude')
#
#    eastwestList = []
#    for s in range(21):
#        xList = []
#        yList = []
#        for item in centerDict[plotTime][s]:
#            if (item[1] != [])and(item[0] <= endT):
#                yList.append(item[1][0])
#                xList.append(item[1][1])
#        if xList != []:
#            eastwestList.append(np.mean(xList))
#        else:
#            eastwestList.append(99999)
#        ax1.plot(xList, yList)
#    ewInd = np.argsort(eastwestList)
#    threeScenDict[plotTime] = (ewInd[0],ewInd[10],ewInd[20])
#    
#    fig.tight_layout()
#    ax1.set_xbound(lower = -88,upper = -70)
#    ax1.set_ybound(lower = 20,upper = 32)
#    handles, labels = ax1.get_legend_handles_labels()
#    ax1.legend(handles, labels)
#    plt.show()
#    fig.savefig(os.path.join("/Users/haoxiangyang/Dropbox/NU Documents/Hurricane/Data/GEFS_Simu/",str(plotTime)+".png"))
#
## obtain the left/right/middle path
#fulRateList = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
#for fr in fulRateList:
#    outPdict = pickle.load(open('/Users/haoxiangyang/Desktop/Git/daniel_Diesel/data/predDemand/predDemand_{}.p'.format(int(fr*100)),'rb'))
#    outPdictNew = {}
#    for plotTime in refTList:
#        for sInd in range(len(threeScenDict[plotTime])):
#            outPdictNew[sInd] = outPdict[threeScenDict[plotTime][sInd]]
#    with open('/Users/haoxiangyang/Desktop/Git/daniel_Diesel/data/predDemand/predDemand1_{}.p'.format(int(fr*100)), 'wb') as fp:
#        pickle.dump(outPdictNew, fp, protocol=pickle.HIGHEST_PROTOCOL)
            