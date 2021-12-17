#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 13:56:33 2018

@author: haoxiangyang
"""

from os import path
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.pyplot import plot
matplotlib.use('agg') #To plot on linux
import sys
import pickle
import numpy as np
import pandas as pd
from scipy.stats import norm
import webbrowser

def calL(z):
    L = np.exp(-1/2*z**2)/np.sqrt(2*np.pi) - z*(1 - norm.cdf(z))
    return L

def QRCal(fl_df,c,scenSet,h,K,p,epsilon):
    demandScen = [fl_df.demand[c]*i for i in scenSet]
    mu = np.mean(demandScen)
    sigma = np.std(demandScen)
    QCurr = mu
    RCurr = 0
    stopBool = True
    while stopBool:
        # record the previous one
        QPrev = QCurr
        RPrev = RCurr
        # calculate the new R and Q
        RCurr = norm.ppf(1 - QCurr*h/(p*mu),loc = mu,scale = sigma)
        QCurr = np.sqrt(2*mu*(K + p*sigma*calL((RCurr - mu)/sigma))/h)
        if (QCurr - QPrev <= epsilon*QCurr)and(RCurr - RPrev <= epsilon*RCurr):
            stopBool = False
    return QCurr,RCurr

#%%
T=11
t_lenght = 6# Time of the length of one time period.

N=4

supply_factor = 1.5
Totaltrucks = 1000*supply_factor
truck_cap = 230#Barrels/Truck
dem_pen = 1
truck_speed = 80 #km/h

# create port sections
sectionDict = {}
sectionDict['Bay County_S'] = ['Walton County','Holmes County','Jackson County','Washington County','Bay County','Calhoun County','Gulf County']
sectionDict['Brevard County_S'] = ['Putnam County','Flagler County','Volusia County','Brevard County','Orange County','Seminole County','Osceola County','Indian River County']
sectionDict['Broward County_S'] = ['Lee County','DeSoto County','Hardee County','Highlands County','Glades County','Hendry County','Palm Beach County','St. Lucie County','Martin County','Collier County','Broward County','Monroe County','Miami-Dade County','Okeechobee County','Charlotte County']
sectionDict['Duval County_S'] = ['Bradford County','Clay County','Duval County','St. Johns County','Liberty County','Gadsden County','Franklin County','Wakulla County','Leon County','Jefferson County','Madison County','Taylor County','Lafayette County','Suwannee County','Hamilton County','Columbia County','Baker County','Nassau County','Union County']
sectionDict['Escambia County_S'] = ['Escambia County','Santa Rosa County','Okaloosa County']
sectionDict['Hillsborough County_S'] = ['Pinellas County','Dixie County','Gilchrist County','Alachua County','Levy County','Marion County','Citrus County','Hernando County','Sumter County','Lake County','Pasco County','Polk County','Hillsborough County','Manatee County','Sarasota County']

'''
===========================================================================
Data preparation
'''
netwrok_file='../data/floridaNetObj.p'
fl_df, fl_edges = pickle.load(open(netwrok_file, 'rb'))
fl_df = fl_df.set_index('County')
total_population = sum(fl_df.Population)
fl_df['demand'] = fl_df['Population']/total_population
fl_df['supply'] = 0

#===========================================================================
#Set ports supply
# Tampa  = Hillsborough County
# 42.5%
fl_df.loc['Hillsborough County', 'supply'] = 0.425# 0.425
# Port Everglades = Broward County
# 40.5%
fl_df.loc['Broward County', 'supply'] = 0.405
# Jacksonville - Duval County
# 9.4%
fl_df.loc['Duval County', 'supply'] = 0.094
# entry point - Brevard County (port canaveral)
# 4.4%
fl_df.loc['Brevard County', 'supply'] = 0.044
# Pensacola = Escambia County
# 1.8%
fl_df.loc['Escambia County', 'supply'] = 0.018
# Panama City =Bay County
# 1.3% (1.4 so that they add up to 1)
fl_df.loc['Bay County', 'supply'] = 0.014
#===========================================================================
supply_nodes  = fl_df.loc[fl_df['supply']>0].index.tolist()
supply_df = fl_df.loc[supply_nodes].copy()
supply_df['demand'] = 0
supply_df.index = [nn+'_S' for nn in supply_df.index]
supply_nodes  = supply_df.index.tolist()
trucks = {sn:Totaltrucks*fl_df['supply'][sn[:-2]] for sn in supply_nodes}#Per Port}
print(trucks)
fl_df['supply'] = 0 #Old dataframe only has demands
fl_df = fl_df.append(supply_df)

fl_df['supply'] = fl_df['supply']*100000*supply_factor
fl_df['demand'] = fl_df['demand']*100000

fractions_i = [1,1.25, 1.5, 1.75]
for i in range(N):
    demand_sce = 'demand_%i' %(i)
    fl_df[demand_sce] = fractions_i[i]*fl_df['demand']

demand_nodes = fl_df.loc[fl_df['demand']>0].index.tolist()


# build the SS policy for Hurricane Irma
def build_irma_sample_path(filename):
    irma_peacks = pickle.load(open(filename, 'rb'))
    time_stamps = list(irma_peacks.keys())
    time_stamps.sort()
    sample_path = []
    for (i,t) in enumerate(time_stamps):
        data_t = irma_peacks[t]
        realization_t = {'demand[%i,%s]' %(i+1,c):fl_df['demand'][c]*(data_t[c] if c in data_t else 1)  for c in demand_nodes}
        sample_path.append(realization_t)
    realization_0 = {'demand[%i,%s]' %(0,c):0  for c in demand_nodes}
    sample_path.insert(0, realization_0)
    time_stamps.insert(0, None)
    return sample_path, time_stamps

irma_sample_path, samplepath_time_stamps = build_irma_sample_path('../data/factor_data.p')

#%%
# for each county, calculate Q,R
# assume all shipment can be done within a period of time
Q = {}
R = {}
S = {}
h = 0.1
K = 0
p = 1
for c in demand_nodes:
    # needs the function to calculate Q and R!!!!!!
    #Q[c],R[c] = QRCal(fl_df,c,fractions_i,h,K,p,1e-3)
    demandScen = [fl_df.demand[c]*i for i in fractions_i]
    # 95% fulfillment rate
    S[c] = np.mean(demandScen)+np.std(demandScen)*1.64

# initialization of the supply chain
Inflow = {}
InitialTruckInv = {}
InitialInv = {}
for c in supply_nodes:
    Inflow[c] = fl_df.supply[c]
    InitialTruckInv[c] = trucks[c]
    InitialInv[c] = 10*Inflow[c]
for c in demand_nodes:
    InitialTruckInv[c] = 0
    InitialInv[c] = fl_df.demand[c]*3

# simulate for each time period, the order, the inventory and the demand
InvList = {}
rawOrder = {}
orderList = {}
rawDemand = {}
unsatDemand = {}
TruckInv = {}
for tp in range(len(samplepath_time_stamps)):
    InvList[tp] = {}
    rawOrder[tp] = {}
    orderList[tp] = {}
    rawDemand[tp] = {}
    unsatDemand[tp] = {}
    TruckInv[tp] = {}
    if tp == 0:
        # to initialize the initial inventory
        for c in demand_nodes:
            InvList[tp][c] = InitialInv[c]
            TruckInv[tp][c] = InitialTruckInv[c]
            orderList[tp][c] = 0
            rawDemand[tp][c] = 0
        for c in supply_nodes:
            InvList[tp][c] = InitialInv[c]
            TruckInv[tp][c] = InitialTruckInv[c]
    else:
        # to calculate the inventory and actual order amount
        for c in demand_nodes:
            rawDemand[tp][c] = irma_sample_path[tp]['demand[{},{}]'.format(tp,c)]
            TruckInv[tp][c] = orderList[tp - 1][c]/truck_cap
            InvList[tp][c] = max(InvList[tp - 1][c] - rawDemand[tp - 1][c],0) + orderList[tp - 1][c]
            if InvList[tp][c] < S[c]:
                rawOrder[tp][c] = S[c] - InvList[tp][c]
            else:
                rawOrder[tp][c] = 0
            if rawDemand[tp][c] >= InvList[tp][c]:
                tempusD = rawDemand[tp][c] - InvList[tp][c]
                if tempusD >= 0.001:
                    unsatDemand[tp][c] = tempusD
                else:
                    unsatDemand[tp][c] = 0
            else:
                unsatDemand[tp][c] = 0
        for c in supply_nodes:
            TruckInv[tp][c] = TruckInv[tp - 1][c] - sum(orderList[tp - 1][cd]/truck_cap - TruckInv[tp - 1][cd] for cd in sectionDict[c])
            InvList[tp][c] = InvList[tp - 1][c] - sum(orderList[tp - 1][cd] for cd in sectionDict[c]) + Inflow[c]
            totalOrderCurrent = sum(rawOrder[tp][cd] for cd in sectionDict[c])
            totalCapacity = min(InvList[tp][c],TruckInv[tp][c]*truck_cap)
            if totalOrderCurrent > totalCapacity:
                for cd in sectionDict[c]:
                    orderList[tp][cd] = rawOrder[tp][cd]/totalOrderCurrent*totalCapacity
            else:
                for cd in sectionDict[c]:
                    orderList[tp][cd] = rawOrder[tp][cd]

uDList = []                    
for tp in range(len(samplepath_time_stamps)):
    uDList.append(unsatDemand[tp])
policy_on_irma = {'sample_path': irma_sample_path, 'unmet_demand':uDList}
with open('../data/irma_policy.p', 'wb') as fp:
    pickle.dump(policy_on_irma, fp, protocol=pickle.HIGHEST_PROTOCOL)