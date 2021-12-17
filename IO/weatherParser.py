#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 21 12:40:11 2018

@author: haoxiangyang
"""
#%%
# define the constants
os.chdir("/Users/haoxiangyang/Desktop/Git/daniel_Diesel/")
fl_df, fl_edges, demand_nodes, supply_nodes, net_nodes, Tau_max, tau_arcs, trucks = load_florida_network(t_step, 0, 100)
countyCode = {"00235":"CITRUS","00325":"GADSDEN","00326":"NASSAU","00360":"OKEECHOBEE",\
              "00415":"BRADFORD","00482":"SARASOTA","00485":"FLAGLER","03818":"JACKSON",\
              "12812":"CHARLOTTE","12815":"ORANGE","12816":"ALACHUA","12818":"HERNANDO",\
              "12819":"LAKE","12832":"FRANKLIN","12833":"DIXIE","12834":"VOLUSIA",\
              "12836":"MONROE","12838":"BREVARD","12839":"MIAMI-DADE","12842":"HILLSBOROUGH",\
              "12843":"INDIAN RIVER","12844":"PALM BEACH","12849":"BROWARD","12854":"SEMINOLE",\
              "12861":"MARION","12871":"MANATEE","12873":"PINELLAS","12883":"POLK","12894":"LEE",\
              "12895":"ST. LUCIE","12897":"COLLIER","13884":"OKALOOSA",\
              #"13889":"DUVAL",\
              "13899":"ESCAMBIA","53848":"SANTA ROSA","53862":"TAYLOR","63871":"HOLMES",\
              "73805":"BAY","92813":"OSCEOLA","92814":"ST. JOHNS","92815":"MARTIN","92817":"SUMTER",\
              "92827":"HIGHLANDS","93805":"LEON","00486":"PASCO"}
for iKey in countyCode.keys():
    countyCode[iKey] = countyCode[iKey].capitalize() + ' County'
    
#%%
xm = np.array([])
ym = np.array([])
zm = np.array([])
for i in range(len(cset)):
    c = cset[i]
    x = np.array([])
    y = np.array([])
    for tp in sorted(totalData.keys()):
        #x = np.append(x,(tp - mint)/(maxt - mint))
        if (tp <= maxt)and(tp >= mint):
            x = np.append(x,tp)
            y = np.append(y,totalData[tp][c][0]/totalData[tp][c][1])
    #plt.plot(x,y)
    
    # plot the county wind plot
    dataW = dataDict[c]
    dataWsorted = sorted(dataW,key = lambda k:k[0])
    xw = np.array([])
    yw = np.array([])
    for item in dataWsorted:
        if (item[0] <= maxt)and(item[0] >= mint):
            if item[1] != None:
                if item[2] != None:
                    windSFinal = item[2]
                else:
                    windSFinal = item[1]
                #xw = np.append(xw,(item[0] - mint)/(maxt - mint))
                xw = np.append(xw,item[0])
                yw = np.append(yw,windSFinal)
    # add NDFD data here
    for tp in sorted(windNDFD.keys()):
        ttp = min(windNDFD[tp].keys())
        if (ttp <= maxt)and(ttp >= mint)and(c in windNDFD[tp][ttp].keys()):
            xw = np.append(xw,ttp)
            yw = np.append(yw,windNDFD[tp][ttp][c])
    if (len(y) != 0)and(len(yw) != 0):
        ym = np.append(ym,max(y))
        xm = np.append(xm,max(yw))
        zm = np.append(zm,hwmData[c])

#%%
t_step = 6

#mapFunc = [-3.0408515734930983,0.05954703187627413]
totalDemand = 2.383e11/(365*24/t_step)

startT = datetime.datetime(2017, 9, 6, 0, 0)
endT = datetime.datetime(2017, 9, 25, 6, 0)
deltaT = datetime.timedelta(hours = t_step)
timePeriods = []
currentT = startT
while currentT <= endT:
    timePeriods.append(currentT)
    currentT += deltaT

dieCoeff = t_step*3600/(reg.coef_[0]/100)
ndfdReal = ndfdWindReader("/Users/haoxiangyang/Desktop/degrib/CollectedData/NDFDIrma/grbout/",demand_nodes)
predictedDemand = {}
for timeCurrent in timePeriods:
    timeNDFDUsed = max([i for i in ndfdReal.keys() if i <= timeCurrent])
    predictedDemand[timeCurrent] = forecastDemand(demand_nodes,fl_df,ndfdReal,[],timeCurrent,list(ndfdReal[timeNDFDUsed].keys()),mapFunc,dieCoeff,totalDemand)

realDemand = {}
for timeCurrent in timePeriods:
    realDemand[timeCurrent] = realDemandt(demand_nodes,totalData,fl_df,timeCurrent,totalDemand)

with open('../data/predDemand/realDemand.p', 'wb') as fp:
    pickle.dump(realDemand, fp, protocol=pickle.HIGHEST_PROTOCOL)
    
#%%
# read high watermark data
hwmData = usgsReader('/Users/haoxiangyang/Dropbox/NU Documents/Hurricane/Data/FilteredHWMs.csv',countyNameCap)
# read in the pSurge data to estimate the surge height
# use the expected surge height
surgeData,surgeLoc = pickle.load(open('/Users/haoxiangyang/Desktop/Git/daniel_Diesel/data/pSurge.p', 'rb'))
surgePred = {}
surgePdf = {}
for tp in surgeData['ProbSurge01c'].keys():
    surgePdf[tp] = {}
    surgePred[tp] = {}
    for ttp in surgeData['ProbSurge01c'][tp].keys():
        surgePdf[tp][ttp] = []
        surgePred[tp][ttp] = []
        for l in range(len(surgeLoc)):
            probList = [1.0]
            predSurge = 0
            for i in range(1,11):
                level = str(i).zfill(2)
                if surgeData['ProbSurge{}c'.format(level)][tp][ttp][l] != 9999.0:
                    probList.append(surgeData['ProbSurge{}c'.format(level)][tp][ttp][l]/100)
                    predSurge += (i - 1)*(probList[-2] - probList[-1])
                else:
                    probList.append(0.0)
                    predSurge += (i - 1)*(probList[-2] - probList[-1])
            predSurge += 10*probList[-1]
            surgePdf[tp][ttp].append(probList)
            surgePred[tp][ttp].append(predSurge)

#%%
# verify the NDFD data

# compare the tdelta hour data
xn = np.array([])
yn = np.array([])
for tp in windNDFD.keys():
    ttp = min(list(windNDFD[tp].keys()))
    # find the closest record to the test time point
    for c in countyCode.values():
        totaltp = [item[0] for item in lcdData[c]]
        tdList = [i - ttp for i in totaltp]
        tdList = [abs(i.total_seconds()) for i in tdList]
        mintd = min(tdList)
        if mintd <= 600:
            closestT = totaltp[tdList.index(mintd)]
            xn = np.append(xn,windNDFD[tp][ttp][c])
            windItem = lcdData[c][totaltp.index(closestT)]
            if windItem[1] != None:
                if windItem[2] != None:
                    windSFinal = windItem[2]
                else:
                    windSFinal = windItem[1]
            yn = np.append(yn,windSFinal)

reg = linear_model.LinearRegression()
reg.fit(np.transpose(np.matrix(xn)),yn)
xt = np.array(range(0,100,2))
#yt = np.array(range(0,100,2))
yt = reg.predict(np.transpose(np.matrix(xt)))
plt.scatter(xn,yn,color = "black")
plt.plot(xt,yt)

#%%
# find the how long it takes to recover
stageTDict = {}
recoveryTDict = {}
peakDict = {}
xTemporal = np.array([])
yTemporal = np.array([])
zTemporal = np.array([])

for i in range(len(cset)):
    c = cset[i]
    x = np.array([])
    y = np.array([])
    for tp in sorted(totalData.keys()):
        #x = np.append(x,(tp - mint)/(maxt - mint))
        if (tp <= maxt)and(tp >= mint):
            x = np.append(x,tp)
            y = np.append(y,totalData[tp][c][0]/totalData[tp][c][1])
    # find the peak time
    peakT = y.argmax()
    if y[peakT] >= 0.2:
        lastGoodB = 0
        firstGoodA = len(y) - 1
        for loc in range(peakT):
            if y[loc] < 0.01:
                lastGoodB = loc
        for loc in range(peakT,len(y)):
            if y[loc] > 0.01:
                firstGoodA = loc
        if firstGoodA < len(y) - 1:
            firstGoodA += 1
        stageTDict[c] = x[peakT] - x[lastGoodB]
        zTemporal = np.append(zTemporal,stageTDict[c].total_seconds())
        recoveryTDict[c] = x[firstGoodA] - x[peakT]
        yTemporal = np.append(yTemporal,recoveryTDict[c].total_seconds())
        peakDict[c] = y[peakT]
        xTemporal = np.append(xTemporal,peakDict[c])
        
plt.scatter(xTemporal,zTemporal)
plt.scatter(xTemporal,yTemporal)
reg = linear_model.LinearRegression()
reg.fit(np.transpose(np.matrix(xTemporal)),yTemporal)

#%%
# check the accuracy of the NDFD against LCD
startTime = datetime.datetime(2017,9,6,6,0)
endTime = datetime.datetime(2017,9,25,0,0)
tDelta = datetime.timedelta(hours = 3)
currentT = startTime
predTList = sorted(list(windNDFD['WindSpd'].keys()))
errorList = []
errorDict = {}
for c in countyList:
    errorDict[c] = []
windData = {}
while currentT <= endTime:
    # find the closest prediction of NDFD to compare it with the closest LCD
    ndfdTime = max([i for i in predTList if (i <= currentT)and(currentT in windNDFD['WindSpd'][i].keys())])
    windData[currentT] = {}
    for c in countyName:
        cInd = windLoc.index(c)
        td = currentT - ndfdTime
        if td.total_seconds() <= 21600:
            windData[currentT][c] = windNDFD['WindSpd'][ndfdTime][currentT][cInd]
    for c in countyName:
        if c in countyCode.values():
            lcdTime = max([i for i in range(len(lcdData[c])) if lcdData[c][i][0] <= currentT])
            if lcdData[c][lcdTime][1] != None:
                lcdTD = currentT - lcdData[c][lcdTime][0] 
                ndfdTD = currentT - ndfdTime
                errorList.append(lcdData[c][lcdTime][1] - windNDFD['WindSpd'][ndfdTime][currentT][countyName.index(c)])
                errorDict[c].append((lcdData[c][lcdTime][1], windNDFD['WindSpd'][ndfdTime][currentT][countyName.index(c)],lcdTD.total_seconds(),ndfdTD.total_seconds()))
    currentT += tDelta

predTList = sorted(list(gustNDFD['WindGust'].keys()))
gustData = {}
currentT = startTime
while currentT <= endTime:
    # find the closest prediction of NDFD to compare it with the closest LCD
    ndfdTime = max([i for i in predTList if (i < currentT)and(currentT in gustNDFD['WindGust'][i].keys())])
    gustData[currentT] = {}
    for c in countyName:
        cInd = gustLoc.index(c)
        td = currentT - ndfdTime
        if td.total_seconds() <= 21600:
            gustData[currentT][c] = gustNDFD['WindGust'][ndfdTime][currentT][cInd]
    currentT += tDelta

with open('/Users/haoxiangyang/Desktop/Git/daniel_Diesel/data/windData.p', 'wb') as fp:
    pickle.dump([windData,gustData,errorList,errorDict,lcdData,windNDFD], fp, protocol=pickle.HIGHEST_PROTOCOL)
    