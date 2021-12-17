#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 17:48:44 2019

@author: dduque
"""


import os 
import sys
data_path = os.path.expanduser("~/Dropbox/WORKSPACE/FuelDistModel/data/")
import  pickle
import datetime
import numpy as np 
import matplotlib.pyplot as plt

# path_to_forecast = data_path + 'predDemand/predDemand_30.p'
path_to_forecast = data_path + 'predDemand/predDemand_60.p'
'''
Contains a dictionary with each replication of the enamble
data[ensamble_number][issue_time][prediction_time] = array of predictions per county
'''
data  = pickle.load(open(path_to_forecast, 'rb'))
scenarios = list(data.keys())
issue_times = list(data[0].keys())
issue_times.sort()
pred_times = list(set((pt for it in issue_times for pt in data[0][it].keys())))
pred_times.sort()
for s in data.keys():
    print(data[s][datetime.datetime(2017, 9, 7, 18, 0)][datetime.datetime(2017, 9, 10, 0, 0)]["Hillsborough County"])

counties = list(data[0][issue_times[0]][pred_times[0]].keys())
counties.sort()

# Fix a scenario
for s in range(22):
    counties_data = [[data[s][it][it][c]  for c in counties] for it in issue_times]
    plt.plot(counties_data)

# Fix county
c = 'Hillsborough County'#'Palm Beach County'# 'Hillsborough County' # 'Monroe County' # 
it  = datetime.datetime(2017, 9, 9, 6, 0)
counties_data = [[data[s][it][pt][c] for s in [3, 18]] for pt in pred_times if pt in data[0][it]]
plt.plot(counties_data, color='green')
np.set_printoptions(edgeitems=10)
cd_mat = np.array(counties_data).transpose()



# Fix county
c = 'Hillsborough County'#'Palm Beach County'# 'Hillsborough County' # 'Monroe County' # 
counties_data = [[data[s][it][it][c] for s in data] for it in issue_times]
plt.plot(counties_data)


from gurobipy import *

m = Model('dif_scenarios')


x = m.addVars(scenarios, lb=0, ub=1, vtype=GRB.BINARY ,  name='x')
m.addConstr(x.sum()==2, 'choose2')

of = QuadExpr()
xiT = {}
for i in scenarios:
    xiT[i] = np.array([data[i][it][pt][c] for c in counties for it in issue_times for pt in pred_times if pt in data[i][it]])
for i in scenarios:
    for j in scenarios:
        if i<j:
            dij = np.linalg.norm(xiT[i]-xiT[j])**2
            of.addTerms(dij, x[i], x[j])
m.setObjective(of, GRB.MAXIMIZE)
m.optimize()
print([i for i in scenarios if x[i].X > 0])
            
        
