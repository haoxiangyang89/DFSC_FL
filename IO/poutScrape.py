#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 21:45:38 2018

@author: haoxiangyang

Referenced from Thomas Massion's code in scraping the power outage data for US.
"""

from datetime import datetime

import pickle
from bs4 import BeautifulSoup
import os.path
import sys
import urllib
import time

# INPUT: 1) county keys database for each state you wish to scrape outage data from (collected into countylists below) 
#        2) state name such as 'Georgia','South Carolina','North Carolina'
# OUTPUT: US_power.p -> pickle file containing a dictionary of the outage data just as it is organized online
#                       format: state: county: provider: lastupdated: (cust_tracked,cust_out)

def get_powerdat(stateCountyID,stateList,fileName):
    #helpful source: https://srome.github.io/Parsing-HTML-Tables-in-Python-with-BeautifulSoup-and-pandas/
    
    # only create new dictionary if it is not existent in current directory
    if os.path.isfile(fileName):
        US_states = pickle.load( open( fileName, "rb" ) )
        existentBool = True
    else:    
        US_states = {}
        existentBool = False

    # loops through each state/county to scrape
    for stateKey in stateList:
        if (not(existentBool))or(not(stateKey in US_states.keys())):
            US_states[stateKey] = {}
        for countyKey in stateCountyID[stateKey].keys():
            countyName = stateCountyID[stateKey][countyKey]
            target_url = 'https://poweroutage.us/area/county/' + countyKey
            print(target_url)
            urlFile = urllib.request.urlopen(target_url)
            urltxt = urlFile.read()
            urltxt = urltxt.decode()
            soup = BeautifulSoup(urltxt) # soup = BeautifulSoup(html)
            table = soup.find_all('table')[0] # Grab the first table
            if (not(existentBool))or(not(countyName in US_states[stateKey].keys())):
                US_states[stateKey][countyName] = {}
            
            #the loops below fill the county dictionary
            for j,row in enumerate(table.find_all('tr')):
                #print(row.get_text())
                
                columns = row.find_all('td')
                # print('a row : *******************')
                # skip the row of column entry names
                if j > 0:
                    # extract table row entries
                    provider_name = columns[0].get_text()
                    cust_tracked = float(columns[1].get_text().replace(',',''))
                    cust_out = float(columns[3].get_text().replace(',',''))
                    lastupdated = columns[5].get_text()
                    lastupdated = datetime.strptime(lastupdated, '%m/%d/%Y %I:%M:%S %p %Z')
                    timeStamp = datetime.now()
                    timeStamp = datetime(timeStamp.year,timeStamp.month,timeStamp.day,timeStamp.hour)
                    
                    if existentBool:
                        # account for if the state, county or provider is not already in the dictionary
                        if provider_name not in US_states[stateKey][countyName]:
                            US_states[stateKey][countyName][provider_name] = {}
                        US_states[stateKey][countyName][provider_name][timeStamp] = (cust_tracked,cust_out,lastupdated)
                    else:
                        US_states[stateKey][countyName][provider_name] = {}
                        US_states[stateKey][countyName][provider_name][timeStamp] = (cust_tracked,cust_out,lastupdated)
    
    # save dictionary with format:
    # state: county: provider: lastupdated: (cust_tracked,cust_out)
    pickle.dump(US_states, open( fileName, "wb" ))

#%%
# this will repeat every 3 hours
if __name__=="__main__":
    # the first argument is the time limit (days), second argument is the frequency of running the script
    # third argument is the data file name
    # the rest are the list of states
    timeLimit = int(sys.argv[1])
    samplingfreq = int(sys.argv[2])
    fileName = sys.argv[3]
    stateList = sys.argv[4:]
    stateCountyIDraw = pickle.load(open('../data/stateCountyID.p', 'rb'))
    stateCountyID = stateCountyIDraw[0]
    startT = time.time()
    while time.time() - startT <= timeLimit:
        get_powerdat(stateCountyID,stateList,fileName)
        time.sleep(samplingfreq)