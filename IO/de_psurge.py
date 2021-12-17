# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 16:22:58 2018

@author: Thomas Massion
"""
from datetime import timedelta
import datetime
# put all of the batch created txt files into csv's

# use length/ points from southeast points into 
counter = 0 
psurge = []
lat = []
lon = []

PTSfilepath = input('Full path for the "floridaPTS.txt" file (example: "C:/ndfd/floridaPTS.txt"): ') 
DIRfilepath = input('Full directory path for the all txt ouput files (example: "C:/ndfd/degrib/data/fltdat/out/"): ') 

with open(PTSfilepath,'r',newline='\n') as f2: # in format: #label,lat,lon\n
    pnts = [line.split(',') for line in f2]
    n = len(pnts)
    lat.append('element')
    lat.append('unit')
    lat.append('refTime')
    lat.append('validTime')
    lon.append('element')
    lon.append('unit')
    lon.append('refTime')
    lon.append('validTime')
    
    # making the lat and lon the headers of the "columns" they correspond to
    for i in range(n):
        lat.append(float(pnts[i][1]))
        lon.append(float(pnts[i][2]))
     
    psurge.append(lat)
    psurge.append(lon)
    
    dt = datetime.datetime(2017,9,7,12) # starting date
    for z in range(34,52): # the official: range(34,52): # advisories available from psurge (adv 34 to adv 51)
        sdate = dt.strftime("%Y%m%d%H")
        for j in range(1,21): # the official: range(1,21): from prob surge above 1 feet to above 20
            fname = DIRfilepath + sdate + '_Irma_Adv' + str(z) + '_gt' + str(j) + '_cum_agl.txt' 
            with open(fname,'r') as f:
                for i, line in enumerate(f):
                    if i == 1:
                        psurge.append(line.split(', '))
                        break

        dt = dt + timedelta(hours=6)

    