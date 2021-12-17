#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 09:38:55 2019

@author: haoxiangyang
"""
from gefsParser import *
from ndfdParser import *
from dataProcess import *
# obtained from fitting the sigmoid function
fitParam = [-6.3876227823537075,0.08887279]
recoverParam = [-7.1017645909936817,171.58803196]
# recoverParam = [-3.1731722926874966, 142.97540857, 29.4192153] concave fit
cutoffVal = -recoverParam[0]/recoverParam[1]

t_step = 6
# from the outage percentage compute the real demand
totalDemand = 2.383e11/(365*24/t_step)

# generate real demand based on power loss data (plData)
def genReal(plDataAdd,networkAdd,startT,endT,totalDemand,outputFolder,recoverParam = [-7.1017645909936817,171.58803196]):
    # read in Florida network data
    fl_df, fl_edges = pickle.load(open(networkAdd, 'rb'))
    fl_df = fl_df.set_index('County')
    
    # import county names
    countyName,countyNameCap = getCountyName()
    
    # import the power loss data
    totalData = pickle.load(open(plDataAdd, 'rb'))

    # construct base demand (100% loss)
    demandElectricity = {}
    for c in countyName:
        demandElectricity[c] = fl_df.Population[c]/sum(fl_df.Population)*totalDemand
        
    # list all the time periods for the real demand loss
    deltaT = datetime.timedelta(hours = t_step)
    timePeriods = []
    currentT = startT
    while currentT <= endT:
        timePeriods.append(currentT)
        currentT += deltaT
        
    # obtain the real demand
    realDemand = {}
    for timeCurrent in timePeriods:
        realDemand[timeCurrent] = realDemandt(countyName,totalData,fl_df,timeCurrent,totalDemand)
    
    # output the realDemand.p to file
    with open(os.path.join(outputFolder + 'realDemand.p'), 'wb') as fp:
        pickle.dump(realDemand, fp, protocol=pickle.HIGHEST_PROTOCOL)

# generate real demand in the format of GEFS/NDFD
def genRealPred(realDemandLoc,outputFolder,startT,endT,
                fulRateList = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],recoverParam = [-7.1017645909936817,171.58803196]):
    # read in the real demand (100% fulfillment rate)
    realDemand = pickle.load(open(realDemandLoc,'rb'))
    
    # output real demands with different fulfillment rates
    refTList = []
    currentT = startT
    while currentT <= endT:
        refTList.append(currentT)
        currentT += datetime.timedelta(hours = t_step)
        
    for fr in fulRateList:
        realDout = {}
        realDout[0] = {}
        for refT in refTList:
            realDout[0][refT] = {}
            for predT in sorted(realDemand.keys()):
                if predT >= refT:
                    realDout[0][refT][predT] = {}
                    for ikey in realDemand[predT].keys():
                        realDout[0][refT][predT][ikey] = realDemand[predT][ikey]*fr
        
        # output the realDemand.p to file
        with open(os.path.join(outputFolder + 'predReal_{}.p'.format(int(fr*100))), 'wb') as fp:
            pickle.dump(realDout, fp, protocol=pickle.HIGHEST_PROTOCOL)
         

# generate GEFS 21 scenario predictions
def genGEFSPred(networkAdd,GEFSpAdd,realDemandLoc,outputFolder,startT,endT,\
                fulRateList = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0], recoverParam = [-7.1017645909936817,171.58803196]):
    # read in Florida network data
    fl_df, fl_edges = pickle.load(open(networkAdd, 'rb'))
    fl_df = fl_df.set_index('County')
    
    # import county names
    countyName,countyNameCap = getCountyName()
    
    # obtain the scenario prediction
    GEFSdataCounty,titleCounty = pickle.load(open(GEFSpAdd, 'rb'))
    titleCounty = [item.strip() for item in titleCounty]
    
    demandElectricity = {}
    for c in countyName:
        demandElectricity[c] = fl_df.Population[c]/sum(fl_df.Population)*totalDemand
    
    # read in the real demand (100% fulfillment rate)
    realDemand = pickle.load(open(realDemandLoc,'rb'))
    # output GEFS forecast demands with different fulfillment rates
    refTList = []
    currentT = startT
    while currentT <= endT:
        refTList.append(currentT)
        currentT += datetime.timedelta(hours = t_step)

    for fr in fulRateList:
        outPdict,dDict = obtainPredWind(GEFSdataCounty,titleCounty,refTList,fitParam,recoverParam,realDemand,cutoffVal,demandElectricity,fr)
        with open(os.path.join(outputFolder,'predDemand_concave_{}.p'.format(int(fr*100))), 'wb') as fp:
            pickle.dump(outPdict, fp, protocol=pickle.HIGHEST_PROTOCOL)
            
# predict the average GEFS demand
def genGEFSAvgPred(networkAdd,GEFSpAdd,realDemandLoc,outputFolder,startT,endT,
                   fulRateList = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],recoverParam = [-7.1017645909936817,171.58803196]):
    # read in Florida network data
    fl_df, fl_edges = pickle.load(open(networkAdd, 'rb'))
    fl_df = fl_df.set_index('County')
    
    # import county names
    countyName,countyNameCap = getCountyName()
    
    # obtain the scenario prediction
    GEFSdataCounty,titleCounty = pickle.load(open(GEFSpAdd, 'rb'))
    titleCounty = [item.strip() for item in titleCounty]
    
    demandElectricity = {}
    for c in countyName:
        demandElectricity[c] = fl_df.Population[c]/sum(fl_df.Population)*totalDemand
    
    # read in the real demand (100% fulfillment rate)
    realDemand = pickle.load(open(realDemandLoc,'rb'))

    GEFSdataCountyAVG = {}
    # weather info
    for l1key in GEFSdataCounty.keys():
        GEFSdataCountyAVG[l1key] = {}
        GEFSdataCountyAVG[l1key][0] = {}
        # reference time
        for l3key in GEFSdataCounty[l1key][0].keys():
            GEFSdataCountyAVG[l1key][0][l3key] = {}
            for l4key in GEFSdataCounty[l1key][0][l3key].keys():
                scenList = np.zeros(len(GEFSdataCounty[l1key][0][l3key][l4key]))
                for scenNo in range(21):
                    scenList += np.array(GEFSdataCounty[l1key][scenNo][l3key][l4key])
                GEFSdataCountyAVG[l1key][0][l3key][l4key] = scenList/21
     
    # output GEFS forecast demands with different fulfillment rates
    refTList = []
    currentT = startT
    while currentT <= endT:
        refTList.append(currentT)
        currentT += datetime.timedelta(hours = t_step)
           
    for fr in fulRateList:
        outPdict,dDict = obtainPredWind(GEFSdataCountyAVG,titleCounty,refTList,fitParam,recoverParam,realDemand,cutoffVal,demandElectricity,fr,[0])
        with open(os.path.join(outputFolder,'predAvg_concave_{}.p'.format(int(fr*100))), 'wb') as fp:
            pickle.dump(outPdict, fp, protocol=pickle.HIGHEST_PROTOCOL)

def genNDFDPred(networkAdd,gustNDFDAdd,realDemandLoc,outputFolder,startT,endT,
                fulRateList = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],recoverParam = [-7.1017645909936817,171.58803196]):
    # read in Florida network data
    fl_df, fl_edges = pickle.load(open(networkAdd, 'rb'))
    fl_df = fl_df.set_index('County')
    
    # read gustNDFD data
    gustNDFD,gustLoc = pickle.load(open(gustNDFDAdd,"rb"))
    
    # import county names
    countyName,countyNameCap = getCountyName()
    
    # read in the real demand (100% fulfillment rate)
    realDemand = pickle.load(open(realDemandLoc,'rb'))
    
    # output NDFD forecast demands with different fulfillment rates
    demandElectricity = {}
    for c in countyName:
        demandElectricity[c] = fl_df.Population[c]/sum(fl_df.Population)*totalDemand
        refTList = []
        currentT = startT
        while currentT <= endT:
            refTList.append(currentT)
            currentT += datetime.timedelta(hours = t_step)
    
    for fr in fulRateList:
        # NDFDdata,locList,refTList,fitParam,recoverParam,realDemand,cutoff = 0.0,demandDict = {},fulfillRate = 0.2
        outPdict,dDict = obtainPredNDFD(gustNDFD,gustLoc,refTList,fitParam,recoverParam,realDemand,cutoffVal,demandElectricity,fr,recoverParam)
        outNDFD = {}
        outNDFD[0] = outPdict
        with open(os.path.join(outputFolder,'predNDFD_concave_{}.p'.format(int(fr*100))), 'wb') as fp:
            pickle.dump(outNDFD, fp, protocol=pickle.HIGHEST_PROTOCOL)

#%%
# generate real demand with 100% fulfillment rate
genReal('/Users/haoxiangyang/Desktop/Git/DFSC_FLORIDA/data/power_outage_data.p',\
        '/Users/haoxiangyang/Desktop/Git/DFSC_FLORIDA/data/FloridaNetObj.p',\
        datetime.datetime(2017, 9, 6, 0, 0),datetime.datetime(2017, 9, 25, 6, 0),totalDemand,\
        '/Users/haoxiangyang/Desktop/Git/DFSC_FLORIDA/data/predDemand/')

# generate real demand in the prediction data format
genRealPred('/Users/haoxiangyang/Desktop/Git/DFSC_FLORIDA/data/predDemand/realDemand.p',\
            '/Users/haoxiangyang/Desktop/Git/DFSC_FLORIDA/data/predDemand/',\
            datetime.datetime(2017, 9, 6, 0, 0),datetime.datetime(2017, 9, 25, 6, 0),[0.2,0.5,1.0])
        
# generate GEFS 21 scenario p data for 20%, 50%, 100%
genGEFSPred('/Users/haoxiangyang/Desktop/Git/DFSC_FLORIDA/data/FloridaNetObj.p',\
            '/Users/haoxiangyang/Desktop/hurricane_pdata/GEFSdata_County.p',\
            '/Users/haoxiangyang/Desktop/Git/DFSC_FLORIDA/data/predDemand/realDemand.p',\
            '/Users/haoxiangyang/Desktop/Git/DFSC_FLORIDA/data/predDemand/',\
            datetime.datetime(2017,9,6,0,0),datetime.datetime(2017,9,24,18,0),[0.2,0.5,1.0])

# generate GEFS average scenario p data for 20%, 50%, 100%
genGEFSAvgPred('/Users/haoxiangyang/Desktop/Git/DFSC_FLORIDA/data/FloridaNetObj.p',\
            '/Users/haoxiangyang/Desktop/hurricane_pdata/GEFSdata_County.p',\
            '/Users/haoxiangyang/Desktop/Git/DFSC_FLORIDA/data/predDemand/realDemand.p',\
            '/Users/haoxiangyang/Desktop/Git/DFSC_FLORIDA/data/predDemand/',\
            datetime.datetime(2017,9,6,0,0),datetime.datetime(2017,9,24,18,0),[0.2,0.5,1.0])

genNDFDPred('/Users/haoxiangyang/Desktop/Git/DFSC_FLORIDA/data/FloridaNetObj.p',\
            '/Users/haoxiangyang/Desktop/Git/DFSC_FLORIDA/data/gustNDFD.p',\
            '/Users/haoxiangyang/Desktop/Git/DFSC_FLORIDA/data/predDemand/realDemand.p',\
            '/Users/haoxiangyang/Desktop/Git/DFSC_FLORIDA/data/predDemand/',\
            datetime.datetime(2017,9,6,0,0),datetime.datetime(2017,9,24,18,0),[0.2,0.5,1.0])