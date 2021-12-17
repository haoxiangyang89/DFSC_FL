#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 15:06:21 2018

@author: haoxiangyang
"""
import os
import pickle
import csv
import datetime
import re
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from sklearn import linear_model

import urllib.request
from bs4 import BeautifulSoup

monthTrans = {'JAN':1, 'FEB':2, 'MAR':3, 'APR':4, 'MAY':5, 'JUN':6,\
              'JUL':7, 'AUG':8, 'SEP':9, 'OCT':10, 'NOV':11, 'DEC':12}
#%%
# parse the information from the advisory

class advDataFormat:
    def __init__(self,timeIn,centerInfo,forecastInfo):
        self.time = timeIn
        self.center = centerInfo
        self.forecast = forecastInfo
        

def parseAdv(storm_year,storm_num,adv_num):
    # read in the year, the hurricane number and the advisory number
    # request the url
    storm_num_str = str(storm_num).zfill(2)
    adv_str = str(adv_num).zfill(3)
    target_url='https://www.nhc.noaa.gov/archive/'+str(storm_year)+'/al'+  \
        storm_num_str+'/al'+storm_num_str+str(storm_year)+'.fstadv.'+adv_str+'.shtml?'
    html = urllib.request.urlopen(target_url).read()
    htmlRaw = html.decode('utf-8')
    soup = BeautifulSoup(htmlRaw, 'html.parser')
    mainObj = soup.find_all(name = 'div',attrs={'class':'textproduct'})[0]
    mainTxt = mainObj.text
    
    # scrape the information from the main text
    timeInfo = re.findall('([0-9]+) UTC [A-Z][A-Z][A-Z] ([A-Z]+) ([0-9]+) ([0-9]+)',mainTxt)[0]
    storm_month = monthTrans[timeInfo[1]]
    storm_day = int(timeInfo[2])
    storm_hour = int(timeInfo[0][0:2])
    storm_minute = int(timeInfo[0][2:4])
    storm_time = datetime.datetime(storm_year,storm_month,storm_day,storm_hour,storm_minute)
    
    centerInfo = re.findall('CENTER LOCATED NEAR ([ 0-9.]+)N ([ 0-9.]+)W',mainTxt)[0]
    centerLat = float(centerInfo[0])
    centerLong = -float(centerInfo[1])
    moveDir,moveSpd = re.findall('PRESENT MOVEMENT TOWARD THE [A-Z\-]+ OR ([0-9]+) DEGREES AT ([0-9 ]+) KT',mainTxt)[0]
    moveDir = int(moveDir)
    moveSpd = float(moveSpd)
    
    centralPress = float(re.findall('ESTIMATED MINIMUM CENTRAL PRESSURE ([0-9 ]+) MB',mainTxt)[0])
    maxWind,maxGust = re.findall('MAX SUSTAINED WINDS ([0-9 ]+) KT WITH GUSTS TO ([0-9 ]+) KT.',mainTxt)[0]
    maxWind = float(maxWind)
    maxGust = float(maxGust)
    
    forecastInfo = re.findall('FORECAST VALID ([0-9/]+)Z ([0-9.]+)N ([0-9. ]+)W\nMAX WIND ([0-9 ]+) KT...GUSTS ([0-9 ]+) KT.\n([0-9A-Z .\n]+)\n',mainTxt)
    forecastList = []
    for i in range(len(forecastInfo)):
        forecast_day,forecast_t = forecastInfo[i][0].split('/')
        forecast_day = int(forecast_day)
        forecast_hour = int(forecast_t[0:2])
        forecast_minute = int(forecast_t[2:4])
        if forecast_day < storm_day:
            forecast_month = storm_month + 1
        else:
            forecast_month = storm_month
        if forecast_month == 13:
            forecast_month = 1
            forecast_year = storm_year + 1
        else:
            forecast_year = storm_year
        forecastTime = datetime.datetime(forecast_year,forecast_month,forecast_day,forecast_hour,forecast_minute)
        forecastLat = float(forecastInfo[i][1])
        forecastLong = -float(forecastInfo[i][2])
        forecastWind = float(forecastInfo[i][3])
        forecastGust = float(forecastInfo[i][4])
        forecastItem = re.findall('([0-9]+) KT...([0-9 ]+)NE ([0-9 ]+)SE ([0-9 ]+)SW ([0-9 ]+)NW.\n',forecastInfo[i][5])
        forecastDict = {}
        for j in range(len(forecastItem)):
            forecastDict[int(forecastItem[j][0])] = [float(forecastItem[j][k]) for k in range(1,len(forecastItem[j]))]
        forecastList.append((forecastTime,forecastLat,forecastLong,forecastWind,forecastGust,forecastDict))
    
    return storm_time,(centerLat,centerLong,moveDir,moveSpd,centralPress,maxWind,maxGust),forecastList

#%%
# collect the data for Irma
storm_year = 2017
storm_num = 11
advData = {}
for i in range(1,53):
    storm_time,centerInfo,forecastInfo = parseAdv(storm_year,storm_num,i)
    advData[i] = (storm_time,centerInfo,forecastInfo)
    
with open('/Users/haoxiangyang/Desktop/Git/daniel_Diesel/data/advData.p', 'wb') as fp:
    pickle.dump(advData, fp, protocol=pickle.HIGHEST_PROTOCOL)