#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 23:44:30 2019

@author: haoxiangyang
"""

# scripts to plot figures
import os
os.chdir("/Users/haoxiangyang/Desktop/Git/DFSC_FLORIDA/IO")
import csv
import datetime
import re
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from sklearn import linear_model
import pickle
from dataProcess import *
from gefsParser import *

from mpl_toolkits.basemap import Basemap
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.pyplot import plot


countyList = ["ALACHUA","BAKER","BAY","BRADFORD","BREVARD","BROWARD","CALHOUN",\
                  "CHARLOTTE","CITRUS","CLAY","COLLIER","COLUMBIA","DESOTO","DIXIE",\
                  "DUVAL","ESCAMBIA","FLAGLER","FRANKLIN","GADSDEN","GILCHRIST",\
                  "GLADES","GULF","HAMILTON","HARDEE","HENDRY","HERNANDO","HIGHLANDS",\
                  "HILLSBOROUGH","HOLMES","INDIAN RIVER","JACKSON","JEFFERSON",\
                  "LAFAYETTE","LAKE","LEE","LEON","LEVY","LIBERTY","MADISON","MANATEE",\
                  "MARION","MARTIN","MIAMI-DADE","MONROE","NASSAU","OKALOOSA",\
                  "OKEECHOBEE","ORANGE","OSCEOLA","PALM BEACH","PASCO","PINELLAS",\
                  "POLK","PUTNAM","SANTA ROSA","SARASOTA","SEMINOLE","ST. JOHNS",\
                  "ST. LUCIE","SUMTER","SUWANNEE","TAYLOR","UNION","VOLUSIA",\
                  "WAKULLA","WALTON","WASHINGTON"]
countyCode = {"00235":"Citrus County","00325":"Gadsden County","00326":"Nassau County",\
              "00360":"Okeechobee County","00415":"Bradford County","00482":"Sarasota County",\
              "00485":"Flagler County","03818":"Jackson County","12812":"Charlotte County",\
              "12815":"Orange County","12816":"Alachua County","12818":"Hernando County",\
              "12819":"Lake County","12832":"Franklin County","12833":"Dixie County",\
              "12834":"Volusia County","12836":"Monroe County","12838":"Brevard County",\
              "12839":"Miami-Dade County","12842":"Hillsborough County","12843":"Indian River County",\
              "12844":"Palm Beach County","12849":"Broward County","12854":"Seminole County",\
              "12861":"Marion County","12871":"Manatee County","12873":"Pinellas County",\
              "12883":"Polk County","12894":"Lee County","12895":"St. Lucie County",\
              "12897":"Collier County","13884":"Okaloosa County","13899":"Escambia County",\
              "53848":"Santa Rosa County","53862":"Taylor County","63871":"Holmes County",\
              "73805":"Bay County","92813":"Osceola County","92814":"St. Johns County",\
              "92815":"Martin County","92817":"Sumter County","92827":"Highlands County",\
              "93805":"Leon County","00486":"Pasco County"}
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
              "Santa Rosa County","Sarasota County","Seminole County",\
              "St. Johns County","St. Lucie County","Sumter County","Suwannee County","Taylor County","Union County",\
              "Volusia County","Wakulla County","Walton County","Washington County"]
countyNameCap = [i[:-7].upper() for i in countyName]

# obtain the county-based wind data
lcdData = lcdTotalParser("/Users/haoxiangyang/Dropbox/Research_Documents/Hurricane/Data/Irma_LCD/LCD_FL.csv",countyCode)
totalData = pickle.load(open('/Users/haoxiangyang/Desktop/Git/DFSC_FLORIDA/data/power_outage_data.p', 'rb'))
windNDFD,windLoc = pickle.load(open('/Users/haoxiangyang/Desktop/Git/DFSC_FLORIDA/data/windNDFD.p', 'rb'))
gustNDFD,gustLoc = pickle.load(open('/Users/haoxiangyang/Desktop/Git/DFSC_FLORIDA/data/gustNDFD.p', 'rb'))

maxt = datetime.datetime(2017,9,25,0,0)
mint = min(totalData.keys())
cset = [i for i in countyCode.values()]

#%%
# plot the county power loss plot
for i in range(len(cset)):
    c = cset[i]
    cInd = windLoc.index(c)
    x = np.array([])
    y = np.array([])
    for tp in sorted(totalData.keys()):
        #x = np.append(x,(tp - mint)/(maxt - mint))
        if (tp <= maxt)and(tp >= mint):
            x = np.append(x,tp)
            y = np.append(y,totalData[tp][c][0]/totalData[tp][c][1])
    #plt.plot(x,y)
    
    # plot the county wind plot
    dataW = lcdData[c]
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
    # fill in the NDFD data and then plot
    xnd = np.array([])
    ynd = np.array([])
    for tp in windNDFD['WindSpd'].keys():
        ttp = min(windNDFD['WindSpd'][tp].keys())
        xnd = np.append(xnd,ttp)
        ynd = np.append(ynd,windNDFD['WindSpd'][tp][ttp][cInd])
        
    xndg = np.array([])
    yndg = np.array([])
    for tp in gustNDFD['WindGust'].keys():
        ttp = min(gustNDFD['WindGust'][tp].keys())
        xndg = np.append(xndg,ttp)
        yndg = np.append(yndg,gustNDFD['WindGust'][tp][ttp][cInd])
    #plt.plot(xw,yw)
    
    fig, ax1 = plt.subplots(figsize=(15,10))
    plt.title(c,fontsize = 24)
    line1 = ax1.plot(x, y, 'b-')
    ax1.set_xlabel('time (s)',fontsize = 24)
    daysL = mdates.DayLocator()   # every day
    daysFmt = mdates.DateFormatter('%m-%d')
    ax1.xaxis.set_major_locator(daysL)
    ax1.xaxis.set_major_formatter(daysFmt)
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('Power Loss',fontsize = 24)
    ax1.set_ybound(0,1)
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(20)
    
    fig.autofmt_xdate()
    
    ax2 = ax1.twinx()
    ax2.plot(xw, yw, 'r.')
    ax2.plot(xnd, ynd, 'g.')
    ax2.plot(xndg, yndg, 'b.')
    ax2.set_ylabel('Wind/Gust Speed (mph)',fontsize = 24)
    ax2.set_ybound(0,120)
    ax2.tick_params(axis = "y",labelsize = 20)
    ax1.legend(("Power_Loss",),fontsize = 'xx-large')
    ax2.legend(("LCD_Wind","NDFD_Wind","NDFD_Gust"),bbox_to_anchor=(1, 0.95),fontsize = 'xx-large')

    fig.tight_layout()
    fig.patch.set_facecolor('white')
    plt.show()
    fig.savefig(os.path.join("/Users/haoxiangyang/Dropbox/Research_Documents/Hurricane/Data/Irma_LCD/",c+".png"), dpi=300)

#%%
# plot all recovering paths    
fig, ax1 = plt.subplots(figsize=(15,10))
ax1.set_xlabel('time (s)')
daysL = mdates.DayLocator()   # every day
daysFmt = mdates.DateFormatter('%m-%d')
ax1.xaxis.set_major_locator(daysL)
ax1.xaxis.set_major_formatter(daysFmt)
# Make the y-axis label, ticks and tick labels match the line color.
ax1.set_ylabel('Power Loss')
ax1.set_ybound(0,1)
fig.autofmt_xdate()
for item in ([ax1.xaxis.label, ax1.yaxis.label, ax1.title] +\
             [mt.label for mt in ax1.xaxis.majorTicks] + ax1.get_yticklabels()):
    item.set_fontsize(20)


fig2, ax2 = plt.subplots(figsize=(15,10))
ax2.set_xlabel('time (days)')
# Make the y-axis label, ticks and tick labels match the line color.
ax2.set_ylabel('Power Loss')
ax2.set_ybound(0,1)
fig.autofmt_xdate()
for item in ([ax2.xaxis.label, ax2.yaxis.label, ax2.title] +\
             ax2.get_xticklabels() + ax2.get_yticklabels()):
    item.set_fontsize(20)


plMax = []
for i in range(len(cset)):
    c = cset[i]
    cInd = windLoc.index(c)
    x = np.array([])
    y = np.array([])
    yMax = -1
    tMax = list(sorted(totalData.keys()))[0]
    for tp in sorted(totalData.keys()):
        #x = np.append(x,(tp - mint)/(maxt - mint))
        if (tp <= maxt)and(tp >= mint):
            x = np.append(x,tp)
            y = np.append(y,totalData[tp][c][0]/totalData[tp][c][1])
            if totalData[tp][c][0]/totalData[tp][c][1] > yMax:
                yMax = totalData[tp][c][0]/totalData[tp][c][1]
                tMax = tp
    recBool = True
    recordTime = list(sorted(totalData.keys()))
    recInd = recordTime.index(tMax)
    while recBool:
        remainOut = [totalData[ttp][c][0]/totalData[ttp][c][1] for ttp in totalData.keys() if ttp >= recordTime[recInd]]
        if max(remainOut) <= 0.1:
            recBool = False
        else:
            recInd += 1
    if yMax > 0:
        plMax.append((c,yMax,recordTime[recInd] - tMax))
    xPart = np.array([])
    yPart = np.array([])
    for tp in sorted(totalData.keys()):
        #x = np.append(x,(tp - mint)/(maxt - mint))
        if (tp <= maxt)and(tp >= mint)and(tp >= tMax):
            tDiff = tp - tMax
            xPart = np.append(xPart,tDiff.total_seconds()/86400)
            yPart = np.append(yPart,totalData[tp][c][0]/totalData[tp][c][1])
    
    ax1.plot(x, y, 'b-')
    ax2.plot(xPart, yPart, 'b-')
        
fig.tight_layout()

fig2.tight_layout()
plt.show()

#%%
# extract the max wind before the power loss, plot the histogram
for perc in np.linspace(0.2,0.6,5):
    xs = []
    ys = np.array([])
    for i in range(len(cset)):
        c = cset[i]
        cInd = windLoc.index(c)
        cCap = countyNameCap[countyName.index(c)]
        powerLossP = [tp for tp in totalData.keys() if totalData[tp][c][0]/totalData[tp][c][1] >= perc]
        if len(powerLossP) > 0:
            plStart = min(powerLossP)
            #dataW = lcdData[c]
            yw = []
            for tp in gustNDFD['WindGust'].keys():
                ttp = min(gustNDFD['WindGust'][tp].keys())
                if (ttp <= plStart):
                    windSFinal = gustNDFD['WindGust'][tp][ttp][cInd]
                yw.append(windSFinal)
#            for item in dataW:
#                if (item[0] <= plStart):
#                    if item[1] != None:
#                        if item[2] != None:
#                            windSFinal = item[2]
#                        else:
#                            windSFinal = item[1]
#                    yw.append(windSFinal)
            xs.append(c)
            ys = np.append(ys,max(yw))
    fig = plt.figure()
    plt.hist(ys, bins = 20,range = (0,100))

#%%
# plot GEFS vs. NDFD vs. realDemand
t_step = 6
totalDemand = 2.383e11/(365*24/t_step)
realDemand = pickle.load(open('../data/predDemand/realDemand.p','rb'))
gefsDemand = pickle.load(open('../data/predDemand/predDemand_100.p','rb'))
gefsAvgDemand = pickle.load(open('../data/predDemand/predAvg_100.p','rb'))
ndfdDemand = pickle.load(open('../data/predDemand/predNDFD_100.p','rb'))
demandElectricity = {}
for c in countyName:
    demandElectricity[c] = mapFuel(fl_df.Population[c]/sum(fl_df.Population)*totalDemand)

selectedPath1 = 10
selectedPath2 = 20
predTime = datetime.datetime(2017, 9, 8, 0, 0)
countySelected = "Miami-Dade County"
countyInd = countyName.index(countySelected)

# obtain the real demand
realDList = []
realtY = []
for t in sorted(realDemand.keys()):
    if t >= predTime:
        realDList.append(realDemand[t][countySelected])
        td = t - predTime
        realtY.append(td.total_seconds()/3600)

# obtain 2 GEFS scenario demand
GEFSDList1 = []
GEFSDList2 = []
avgGEFSList = []
gefstY = []
timeList = sorted(gefsDemand[selectedPath1][predTime].keys())
for ikey in timeList:
    GEFSDList1.append(gefsDemand[selectedPath1][predTime][ikey][countySelected])
    GEFSDList2.append(gefsDemand[selectedPath2][predTime][ikey][countySelected])
    avgGEFSList.append(gefsAvgDemand[0][predTime][ikey][countySelected])
    td = ikey - predTime
    gefstY.append(td.total_seconds()/3600)

# obtain NDFD demand
NDFDList = []
ndfdtY = []
NDFDtimeList = sorted(ndfdDemand[0][predTime].keys())
NDFDcountyInd = gustLoc.index(countySelected)
for ikey in NDFDtimeList:
    NDFDList.append(ndfdDemand[0][predTime][ikey][countySelected])
    td = ikey - predTime
    ndfdtY.append(td.total_seconds()/3600)

plt.style.use('classic')
fig, ax1 = plt.subplots(figsize=(15,10))
plt.title("Demand Comparison at " + str(predTime),fontsize = 24)
line1 = ax1.plot(realtY,realDList, color = '#377EB8',linewidth = 4)
line2 = ax1.plot(gefstY,GEFSDList1, color = '#E41A1C', linewidth = 4)
line3 = ax1.plot(gefstY,GEFSDList2, color = '#4DAF4A', linewidth = 4,linestyle='dashed')
line4 = ax1.plot(ndfdtY[:-2],NDFDList[:-2], color = '#984EA3', linewidth = 4,linestyle='dashed')

ax1.set_xlabel('time since prediction (hr)',fontsize = 24)
# Make the y-axis label, ticks and tick labels match the line color.
ax1.set_ylabel('fuel demand (barrels)',fontsize = 24)
ax1.set_ybound(0,40000)
for tick in ax1.xaxis.get_major_ticks():
    tick.label.set_fontsize(20)
for tick in ax1.yaxis.get_major_ticks():
    tick.label.set_fontsize(20)


ax1.legend(("Estimated Demand","GEFS Scen 1","GEFS Scen 2","NDFD"),fontsize = 20)

fig.tight_layout()
plt.show()
fig.savefig("/Users/haoxiangyang/Dropbox/Research_Documents/Hurricane/Writeup/demandComp.png", dpi=300)

#%%
# plot FL map
fig,ax = plt.subplots(figsize=(12,10))
m = Basemap(projection="mill", #miller est une projection connu
    llcrnrlat =24,#24.5,#lower left corner latitude 25.276103, -88.272764
    llcrnrlon =-88.3,
    urcrnrlat =31.5, #upper right lat 30.475935, -79.742215
    urcrnrlon =-79,#-79.7,
    resolution = 'h', ax=ax) #c croud par defaut, l low , h high , f full 
m.drawcoastlines() #dessiner les lignes
m.drawcountries()
m.drawstates()

fi = open('/Users/haoxiangyang/Dropbox/Research_Documents/Hurricane/Data/Irma_LCD/LCD_lat_long.csv',"r")
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
for ckey in countyCode.keys():
    rawDataC = []
    latItem = 0
    longItem = 0
    for item in rawData:
        if item[0] == "WBAN:" + ckey:
            latItem = float(item[title.index("LATITUDE")])
            longItem = float(item[title.index("LONGITUDE")])
            x,y = m(longItem, latItem)
            circle1 = plt.Circle((x,y), 5000, color='blue',fill=True)
            ax.add_patch(circle1)

fig.tight_layout()           
fig.savefig("/Users/haoxiangyang/Dropbox/Research_Documents/Hurricane/Writeup/LCD_loc.png", dpi=300)

#%%
# read in the center information
endTime = datetime.datetime(2017,9,14,18,0)
startT = datetime.datetime(2017,9,7,0,0)
endT = datetime.datetime(2017,9,14,18,0)

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

# plot the Florida map
fig,ax = plt.subplots(figsize=(15,10))
m = Basemap(projection="mill", #miller est une projection connu
    llcrnrlat =20,#24.5,#lower left corner latitude 25.276103, -88.272764
    llcrnrlon =-88.3,
    urcrnrlat =31.5, #upper right lat 30.475935, -79.742215
    urcrnrlon =-70,#-79.7,
    resolution = 'h', ax=ax) #c croud par defaut, l low , h high , f full 
m.drawcoastlines() #dessiner les lignes
m.drawcountries()
m.drawstates()

predT = datetime.datetime(2017,9,8,0,0)
# plot the tracks
for s in range(21):
    xList = []
    yList = []
    for item in centerDict[predT][s]:
        if (item[1] != [])and(item[0] <= endTime):
            yList.append(item[1][0])
            xList.append(item[1][1])
    xm,ym = m(xList, yList)
    m.plot(xm,ym,linewidth = 2)

fig.tight_layout()
fig.savefig("/Users/haoxiangyang/Dropbox/Research_Documents/Hurricane/Writeup/tracks_visualization.png", dpi=300)

#%%
# plot the inventory of the ports and hurricane hit areas
outData = pickle.load(open("/Users/haoxiangyang/Desktop/Git/DFSC_FLORIDA/output/Test_FR50_H96_F72_R24_N24_GEFS.p",'rb'))
outData_ndfd = pickle.load(open("/Users/haoxiangyang/Desktop/Git/DFSC_FLORIDA/output/Test_FR50_H96_F72_R24_N24_NDFD.p",'rb'))
outData_avg = pickle.load(open("/Users/haoxiangyang/Desktop/Git/DFSC_FLORIDA/output/Test_FR50_H96_F72_R24_N24_GAVG.p",'rb'))

dList = ['Lee County','Hillsborough County','Orange County','Alachua County','Pinellas County',
         'Leon County','Miami-Dade County','Broward County','Duval County','Palm Beach County']
sList = ['Hillsborough County_S','Escambia County_S','Duval County_S','Brevard County_S',
         'Broward County_S','Bay County_S']
InvList = {}
InvList_ndfd = {}
InvList_avg = {}
for dloc in dList:
    InvList[dloc] = []
    InvList_ndfd[dloc] = []
    InvList_avg[dloc] = []
    for i,t in outData[3].keys():
        for t1 in range(t,t+24):
            InvList[dloc].append(outData[3][i,t][0][t1,dloc])
            InvList_ndfd[dloc].append(outData_ndfd[3][i,t][0][t1,dloc])
            InvList_avg[dloc].append(outData_avg[3][i,t][0][t1,dloc])
          
    plt.style.use('classic')
    fig, ax1 = plt.subplots(figsize=(15,10))
    plt.title("Inventory Comparison {}".format(dloc),fontsize = 24)
    line1 = ax1.plot(np.array(range(len(InvList[dloc]))),InvList[dloc], color = '#377EB8',linewidth = 4)
    line2 = ax1.plot(np.array(range(len(InvList[dloc]))),InvList_avg[dloc], color = '#E41A1C', linewidth = 4)
    line3 = ax1.plot(np.array(range(len(InvList[dloc]))),InvList_ndfd[dloc], color = '#984EA3', linewidth = 4)
    ax1.legend(("GEFS","GAVG","NDFD"),fontsize = 20)
    fig.savefig("/Users/haoxiangyang/Dropbox/Research_Documents/Hurricane/Data/Inv_plots/inv_{}.png".format(dloc), dpi=300)

#%%
InvList = {}
InvList_ndfd = {}
InvList_avg = {}
for sloc in sList:
    InvList[sloc] = []
    InvList_ndfd[sloc] = []
    InvList_avg[sloc] = []
    for i,t in outData[3].keys():
        for t1 in range(t,t+24):
            InvList[sloc].append(outData[3][i,t][1][t1,sloc])
            InvList_ndfd[sloc].append(outData_ndfd[3][i,t][1][t1,sloc])
            InvList_avg[sloc].append(outData_avg[3][i,t][1][t1,sloc])
          
    plt.style.use('classic')
    fig, ax1 = plt.subplots(figsize=(15,10))
    plt.title("Inventory Comparison {}".format(sloc),fontsize = 24)
    line1 = ax1.plot(np.array(range(len(InvList[sloc]))),InvList[sloc], color = '#377EB8',linewidth = 4)
    line2 = ax1.plot(np.array(range(len(InvList[sloc]))),InvList_avg[sloc], color = '#E41A1C', linewidth = 4)
    line3 = ax1.plot(np.array(range(len(InvList[sloc]))),InvList_ndfd[sloc], color = '#984EA3', linewidth = 4)
    ax1.legend(("GEFS","GAVG","NDFD"),fontsize = 20)
    fig.savefig("/Users/haoxiangyang/Dropbox/Research_Documents/Hurricane/Data/Inv_plots/inv_{}.png".format(sloc), dpi=300)

#%%
rList = {}
rList_ndfd = {}
rList_avg = {}
for dloc in dList:
    rList[dloc] = []
    rList_ndfd[dloc] = []
    rList_avg[dloc] = []
    for i,t in outData[3].keys():
        for t1 in range(t,t+24):
            rList[dloc].append(outData[3][i,t][2][t1,dloc,1])
            rList_ndfd[dloc].append(outData_ndfd[3][i,t][2][t1,dloc,1])
            rList_avg[dloc].append(outData_avg[3][i,t][2][t1,dloc,1])
          
    plt.style.use('classic')
    fig, ax1 = plt.subplots(figsize=(15,10))
    plt.title("r Comparison {}".format(dloc),fontsize = 24)
    line1 = ax1.plot(np.array(range(len(rList[dloc]))),rList[dloc], color = '#377EB8',linewidth = 4)
    line2 = ax1.plot(np.array(range(len(rList[dloc]))),rList_avg[dloc], color = '#E41A1C', linewidth = 4)
    line3 = ax1.plot(np.array(range(len(rList[dloc]))),rList_ndfd[dloc], color = '#984EA3', linewidth = 4)
    ax1.legend(("GEFS","GAVG","NDFD"),fontsize = 20)
    fig.savefig("/Users/haoxiangyang/Dropbox/Research_Documents/Hurricane/Data/Inv_plots/r_{}.png".format(dloc), dpi=300)
    
#%%
gList = {}
gList_ndfd = {}
gList_avg = {}
for sloc in sList:
    gList[sloc] = []
    gList_ndfd[sloc] = []
    gList_avg[sloc] = []
    for i,t in outData[3].keys():
        for t1 in range(t,t+24):
            gList[sloc].append(outData[3][i,t][3][t1,sloc,1])
            gList_ndfd[sloc].append(outData_ndfd[3][i,t][3][t1,sloc,1])
            gList_avg[sloc].append(outData_avg[3][i,t][3][t1,sloc,1])
          
    plt.style.use('classic')
    fig, ax1 = plt.subplots(figsize=(15,10))
    plt.title("g Comparison {}".format(sloc),fontsize = 24)
    line1 = ax1.plot(np.array(range(len(gList[sloc]))),gList[sloc], color = '#377EB8',linewidth = 4)
    line2 = ax1.plot(np.array(range(len(gList[sloc]))),gList_avg[sloc], color = '#E41A1C', linewidth = 4)
    line3 = ax1.plot(np.array(range(len(gList[sloc]))),gList_ndfd[sloc], color = '#984EA3', linewidth = 4)
    ax1.legend(("GEFS","GAVG","NDFD"),fontsize = 20)
    fig.savefig("/Users/haoxiangyang/Dropbox/Research_Documents/Hurricane/Data/Inv_plots/g_{}.png".format(sloc), dpi=300)
