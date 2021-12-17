#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 16:03:06 2019

@author: haoxiangyang
"""

import pickle
from plParser import *

# obtain the power loss data and store it as .p file
fileAdd = "/Users/haoxiangyang/Desktop/Outage/Irma"
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
#==============================================================================
# import county names
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
              "St. Johns County","St. Lucie County","Santa Rosa County","Sarasota County",\
              "Seminole County","Sumter County","Suwannee County","Taylor County","Union County",\
              "Volusia County","Wakulla County","Walton County","Washington County"]
countyNameCap = [i[:-7].upper() for i in countyName]

#%%
totalData = plPDFAnalyzer(countyList,2017,fileAdd)
# fix data error
for tp in sorted(totalData.keys()):
    totalData[tp]["St. Johns County"][1] = 130769
    
with open('/Users/haoxiangyang/Desktop/Git/daniel_Diesel/data/power_outage_data.p', 'wb') as fp:
    pickle.dump(totalData, fp, protocol=pickle.HIGHEST_PROTOCOL)
