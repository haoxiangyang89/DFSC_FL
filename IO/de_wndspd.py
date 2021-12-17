# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 11:23:52 2018

@author: Thomas Massion
"""

from datetime import timedelta
import datetime
import os
# put all of the batch created txt files into csv's

# use length/ points from southeast points into 
counter = 0 
wndspd = []
lat = []
lon = []

PTSfilepath = input('Full path for the "floridaPTS.txt" file (example: "C:/ndfd/floridaPTS.txt"): ') 
DIRpath = input('Full directory path for the all txt output files (example: "C:/ndfd/degrib/data/outWndSpd/"): ') 

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
     
    wndspd.append(lat)
    wndspd.append(lon)
    
    #DIRpath = 'C:/ndfd/degrib/data/outWndSpd/'

    for filename in os.listdir(DIRpath):
        print(filename[-8:-4])
        with open(DIRpath+filename,'r') as f:
            # take the third line (the 7-hour out projection since 447 is for 5 am as opposed to 6am)
            if filename[-8:-4] == '0447':  
                for i, line in enumerate(f):
                    if i == 3:
                        wndspd.append(line.split(', '))
                        break
            # otherwise take the 6-hours-out forecast this is the 2nd line since the windspeed has forecasts for every three hours        
            else:
                for i, line in enumerate(f):
                    if i == 2:
                        wndspd.append(line.split(', '))
                        break
        
            
            
            
    # do your stuff

    
    
#    dt = datetime.datetime(2017,9,7,12) # starting date
#    for z in range(34,52): # the official: range(34,52): # advisories available from psurge (adv 34 to adv 51)
#        sdate = dt.strftime("%Y%m%d%H")
#        for j in range(1,21): # the official: range(1,21): from prob surge above 1 feet to above 20'
#            # format:  YCDZ98_KWBN_201709071146
#            fname = "C:/ndfd/degrib/data/outWndSpd/" + sdate + '_Irma_Adv' + str(z) + '_gt' + str(j) + '_cum_agl.txt' 
#            with open(fname,'r') as f:
#                for i, line in enumerate(f):
#                    if i == 1:
#                        psurge.append(line.split(', '))
#                        break
#
#        dt = dt + timedelta(hours=6)
