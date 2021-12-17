#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 23:03:35 2019

@author: haoxiangyang
"""
from gefsParser import *
# output GEFS data into pickle files

def gefsGridOutput(gridDir,outputAdd):
    # input the directory that contains the GEFS data
    # save the p file to outputAdd
    GEFSdataGrid = {}
    titleGrid,GEFSdataGrid = readGEFS(gridFolder,GEFSdataGrid)
    with open(outputAdd, 'wb') as fp:
        pickle.dump([GEFSdataGrid,titleGrid], fp, protocol=pickle.HIGHEST_PROTOCOL)

def gefsPartOutput(gridDir,outputAdd,windLocGrid):
    # input the directory that contains the GEFS data, 
    # save the p file to outputAdd
    GEFSdata_Part = {}
    titlePart,GEFSdataPart = readGEFS('/Users/haoxiangyang/Desktop/GEFS/output_Grid',GEFSdata_Part,windLocGrid)
    with open('/Users/haoxiangyang/Desktop/hurricane_pData/GEFSdata_Part.p', 'wb') as fp:
        pickle.dump([GEFSdataPart,titlePart], fp, protocol=pickle.HIGHEST_PROTOCOL)

def gefsCountyOutput(countyDir,outputAdd):
    # input the directory that contains the GEFS data, 
    # save the p file to outputAdd
    GEFSdata_County = {}
    titleCounty,GEFSdataCounty = readGEFS(countyDir,GEFSdata_County)
    with open(outputAdd, 'wb') as fp:
        pickle.dump([GEFSdataCounty,titleCounty], fp, protocol=pickle.HIGHEST_PROTOCOL)
    
if __name__ == "__main__":
    windProbGrid,windLocGrid,windLocGridInd = pickle.load(open('../data/windProbGrid.p', 'rb'))
    # obtain the GEFS data for a grid of points in Florida
    gefsGridOutput('/Users/haoxiangyang/Desktop/GEFS/output_Grid',\
                   '/Users/haoxiangyang/Desktop/hurricane_pData/GEFSdata_Grid.p')
    # obtain the GEFS data for a selection of points for Gita
    gefsPartOutput('/Users/haoxiangyang/Desktop/GEFS/output_Grid',\
                   '/Users/haoxiangyang/Desktop/hurricane_pData/GEFSdata_Part.p',windLocGrid)
    gefsCountyOutput('/Users/haoxiangyang/Desktop/GEFS/output_County',\
                     '/Users/haoxiangyang/Desktop/hurricane_pData/GEFSdata_County.p')