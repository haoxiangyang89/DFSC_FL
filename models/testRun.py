#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 22:31:51 2019

@author: haoxiangyang
"""

import os
import sys
from os import path
import time
from gurobipy import GRB
sys.path.append(path.abspath('/Users/haoxiangyang/Desktop/Git/DFSC_FLORIDA'))  # Crunch
sys.path.append(path.abspath('/home/haoxiang/daniel_Diesel'))  # Crunch
import pickle
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.ticker as ticker
#import matplotlib.dates as mdates
from datetime import datetime, timedelta
from gurobipy import Model, quicksum, GRB, tupledict

from models import project_path, models_path
import argparse
import pandas as pd
from math import radians, sin, cos, asin,sqrt

import pdb
pdb.set_trace()

DELTA_HORIZON = None  # Hours of look ahead (time horizon of the two-stage model)
DELTA_T_MODEL = 1  # Temporal model resolution in hours
DELTA_T_STAGE = None  # Temporal resolution of the first stage
DELTA_T_SECOND_STAGE = None  # number of DELTA_T_MODEL periods in the last stage
DELTA_ROLLING = None  # Periods to roll forward in the rolling horizon
DELTA_NOTIFICATION = None  # Periods in advance that a model is solved

R = 6371 #Earth radius in Km
def haversineEuclidean(lat1, lon1, lat2, lon2):
    '''
    Latitudes and longitudes in radians
    '''
    dLat = lat2 - lat1;
    dLon = lon2 - lon1;
    a = sin(dLat/2)**2 + (sin(dLon / 2)** 2)*cos(lat1)*cos(lat2);
    c = 2*asin(sqrt(a));
    return R * c;

def load_florida_network(t_lenght, t_ini, T_max ,partition_network=True, zone = 2):
    '''
    Data preparation of Florida network
    
    Return:
        fl_df (Pandas.DataFrame): DF with information of each county
        fl_edges (dic of str-list): Adjacent counties data structure
        demand_nodes (list of str): List of county names that represent demands
        supply_nodeS (list of str): List of county names (with and extra _S) that
                                    represent supply ports.
    '''
    
    truck_speed = 80 #km/h
    totaltrucks = 1000 # number of trucs in the network
    truck_cap = 230#Barrels/Truck
    network_file = project_path + '/data/FloridaNetObj.p'
    fl_df, fl_edges = pickle.load(open(network_file, 'rb'))
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
    if partition_network == False: 
        supply_nodes  = fl_df.loc[fl_df['supply']>0].index.tolist()
        demand_nodes = fl_df.loc[fl_df['demand']>0].index.tolist()
    else:
        # zone  = #1:North 2: Center, 3 South
        lat_23 = 27.391234 #27.391234, -81.338838
        lat_12 = 29.703668#29.703668, -82.329412
        '''Zone definition ''' 
        fl_df['zone'] = 1
        fl_df.loc[fl_df['latitude']<=lat_12 , 'zone'] = 2
        fl_df.loc[fl_df['latitude']<=lat_23 , 'zone'] = 3
        
        supply_nodes  = fl_df.loc[(fl_df['supply']>0) & (fl_df['zone']==zone)].index.tolist()
        demand_nodes = fl_df.loc[(fl_df['demand']>0) & (fl_df['zone']==zone)].index.tolist()
    
    supply_df = fl_df.loc[supply_nodes].copy()
    supply_df['demand'] = 0
    supply_df.index = [nn+'_S' for nn in supply_df.index]
    supply_nodes  = supply_df.index.tolist()
    trucks = {sn:totaltrucks*fl_df['supply'][sn[:-2]] for sn in supply_nodes}#Per Port}
    
    fl_df['supply'] = 0 #Old dataframe only has demands
    fl_df = fl_df.append(supply_df)
    
    avg_fl_daily_consumption = 153000
    fl_df['supply'] = fl_df['supply']*avg_fl_daily_consumption
    fl_df['demand'] = fl_df['demand']*avg_fl_daily_consumption
    #Transform demand and supply to t_steps intervals
    fl_df['demand'] = t_lenght*fl_df['demand']/24
    fl_df['supply'] = t_lenght*fl_df['supply']/24
        
        
    net_nodes = set()
    net_nodes.update(demand_nodes)
    net_nodes.update(supply_nodes)
    net_nodes = list(net_nodes)
    
    tau_arcs = {}
    elim = 0
    for ci in fl_edges:
        j = 0
        while j<len(fl_edges[ci]):
            cj = fl_edges[ci][j]
            if cj in fl_df.index:
                j+=1
                
            else:
                fl_edges[ci].pop(j)
                elim +=1
    max_t_time =1
    for ci in net_nodes:
        for cj in net_nodes:
            dist_ij = haversineEuclidean(radians(fl_df.latitude[ci]), radians(fl_df.longitude[ci]),
                                                    radians(fl_df.latitude[cj]), radians(fl_df.longitude[cj]))
            time_ij = dist_ij/truck_speed
            periods_ij = np.maximum(np.ceil(time_ij/t_lenght),1)
            tau_arcs[(ci,cj)]=int(periods_ij)
            if periods_ij>max_t_time:
                max_t_time = periods_ij
    
    Tau_max = [tt for tt in np.arange(1,max_t_time+1)]
    
    nominal_demand = {(t,i):fl_df['demand'][i] for t in range(t_ini,T_max,t_lenght) for i in demand_nodes}
    print('Demand: ', sum(fl_df['demand'][demand_nodes]) , '  Supply ' , sum(fl_df['supply'][supply_nodes]))
    return fl_df, fl_edges, demand_nodes, supply_nodes, net_nodes, Tau_max, tau_arcs, trucks, truck_cap, nominal_demand


def DFSC_TwoStage_extensive(t_FS , t_SS, t_max, delta_t_model, t_roll, t_notification,  DFSC_instance, scenarios, tcoeff = 1.001, xcoeff = 1e-4,
                            nombreak = 0.90, nomcoeff = [1, 10, 10], surgebreak = 0.50, surgecoeff = [2, 5]):
    '''
    Builds a extensive formulation of the two stage stochastic program.
    Args:
        t_FS     (int): Time at which the first stage starts
        t_SS     (int): Time at which the second stage starts
        t_max    (int): Time at which the second stage ends
        delta_t  (int): Model resolution in time
        t_roll   (int): Number of time periods to move forward
        DFSC_instance (tuple): Tuple with objects that define an instance
        scenarios (dict) Dictionary with the demand scenarios
    Output:
        m (GRB Model): An extensive formulation model
    '''
    #scenarios = {0:scenarios[1], 1:scenarios[3], 2:scenarios[8]}
    #print("USING 2 SCENARIOS ONLY!!!!!!!!!!!!!!!!!!!!1")
    assert t_FS<=t_SS<=t_max, "Times defining the model are not consistent"
    
    T_set_FS = range(t_FS, t_SS, delta_t_model)
    T_set_SS = range(t_SS, t_max, delta_t_model)
    W = list(scenarios.keys()) #Set of scenarios
    W.sort()
    #Load network data
    fl_df, fl_edges, demand_nodes, supply_nodes, net_nodes, Tau_max, tau_arcs, trucks, truck_cap, nominal_demand = DFSC_instance 
    last_t_FS = T_set_FS[-1]
    out_t_FS = np.minimum(t_FS+t_roll-delta_t_model, last_t_FS)
    #Compute a valid forecast time
    vft = 0 if t_FS == 0 else np.max(np.array([i*(i<=t_FS-t_notification)*(i in scenarios[0]) for i in range(t_FS+1)]))
    print('solving ' , t_FS, ' predicting', vft)
    DTM = delta_t_model
    '''
    Time-space model of Florida modeling 
    '''
    m = Model('DFSC_EXT_%i_' %(t_FS))
    m.setParam('Method',1)   
    m.setParam('OptimalityTol',1e-9)
    m.setParam('FeasibilityTol',1e-9)
    
    '''
    State variables:
        - Inventory at every node (I)
        - Inventory of loaded in-transit trucks (r)
        - Inventory of empty in-transit trucks  (g)
    '''
    pop_weight_d = list(fl_df.Population/sum(fl_df.Population[:len(demand_nodes)]))[:len(demand_nodes)]
    if tcoeff != 0:
        pop_wd = [-i*1e-3 for i in pop_weight_d]
    else:
        pop_wd = [0 for i in pop_weight_d]
    
    I_out = m.addVars(demand_nodes, lb=0, ub=GRB.INFINITY, obj=pop_wd, vtype =GRB.CONTINUOUS, name='I')
    Is_out = m.addVars(supply_nodes, lb=0, ub=GRB.INFINITY, obj=0, vtype =GRB.CONTINUOUS, name='Is')
    r_out = m.addVars(demand_nodes,Tau_max, lb=0, ub=GRB.INFINITY, obj=0, vtype =GRB.CONTINUOUS, name='r')
    g_out = m.addVars(supply_nodes,Tau_max, lb=0, ub=GRB.INFINITY, obj=0, vtype =GRB.CONTINUOUS, name='g')
    

    I0 = m.addVars(demand_nodes, lb=0, ub=0, obj=pop_wd, vtype =GRB.CONTINUOUS, name='I0')
    Is0 = m.addVars(supply_nodes, lb=0, ub=0, obj=0, vtype =GRB.CONTINUOUS, name='Is0')
    r0 = m.addVars(demand_nodes,Tau_max, lb=0, ub=0, obj=0, vtype =GRB.CONTINUOUS, name='r0')
    g0 = m.addVars(supply_nodes,Tau_max, lb=0, ub=0, obj=0, vtype =GRB.CONTINUOUS, name='g0')
    
    #Intra-stage variables FS
    Iiter = []
    Iobj = []
    for tfs in T_set_FS:
        for dn in range(len(demand_nodes)):
            Iiter.append((tfs,demand_nodes[dn]))
            if tcoeff != 0:
                Iobj.append(-tcoeff**tfs*pop_wd[dn]/len(T_set_FS))
            else:
                Iobj.append(0.0)
    I = m.addVars(Iiter, lb=0, ub=GRB.INFINITY, obj=Iobj, vtype =GRB.CONTINUOUS, name='I_intra')
    Is = m.addVars(T_set_FS, supply_nodes, lb=0, ub=GRB.INFINITY, obj=0, vtype =GRB.CONTINUOUS, name='Is_intra')
    r = m.addVars(T_set_FS, demand_nodes,Tau_max, lb=0, ub=GRB.INFINITY, obj=0, vtype =GRB.CONTINUOUS, name='r_intra')
    g = m.addVars(T_set_FS, supply_nodes,Tau_max, lb=0, ub=GRB.INFINITY, obj=0, vtype =GRB.CONTINUOUS, name='g_intra')
    
    #Controls in FS
    x = tupledict()
    y = tupledict()
    
    #Recourse in  the first stage
    z_nominal = m.addVars(T_set_FS, demand_nodes, lb=0, ub=nominal_demand, obj=0, vtype =GRB.CONTINUOUS, name='zN')    #Shortage in nominal demand
    z_surge = m.addVars(T_set_FS, demand_nodes, lb=0, ub=GRB.INFINITY, obj=0, vtype =GRB.CONTINUOUS, name='zS')        #Shortage in surge demand
    nominal_pen = m.addVars(T_set_FS, lb=0, ub=GRB.INFINITY, obj=1, vtype =GRB.CONTINUOUS, name='zNPen')        #shortage penalty for nominal demand
    surge_pen = m.addVars(T_set_FS,demand_nodes, lb=0, ub=GRB.INFINITY, obj=1, vtype =GRB.CONTINUOUS, name='zSPen')          #shortage penalty for surge demand
    
    #truck counter 
    truck_counter = m.addVars(T_set_FS,lb=0,ub=GRB.INFINITY,obj=0,vtype=GRB.CONTINUOUS, name='trucks')
    
    
    #Intra-stage variables SS
    I2 = m.addVars(T_set_SS,demand_nodes, W, lb=0, ub=GRB.INFINITY, obj=0, vtype =GRB.CONTINUOUS, name='I_intra2')
    Is2 = m.addVars(T_set_SS,supply_nodes, W,  lb=0, ub=GRB.INFINITY, obj=0, vtype =GRB.CONTINUOUS, name='Is_intra2')
    r2 = m.addVars(T_set_SS,demand_nodes,Tau_max, W, lb=0, ub=GRB.INFINITY, obj=0, vtype =GRB.CONTINUOUS, name='r_intra2')
    g2 = m.addVars(T_set_SS,supply_nodes,Tau_max, W, lb=0, ub=GRB.INFINITY, obj=0, vtype =GRB.CONTINUOUS, name='g_intra2')
    
    #Controls in SS
    x2 = tupledict()
    y2 = tupledict()
    
    #Recourse in  the first stage
    z_nominal2 = m.addVars(T_set_SS, demand_nodes, W, lb=0, ub=nominal_demand, obj=0, vtype =GRB.CONTINUOUS, name='zN2')    #Shortage in nominal demand
    z_surge2 = m.addVars(T_set_SS, demand_nodes, W, lb=0, ub=GRB.INFINITY, obj=0, vtype =GRB.CONTINUOUS, name='zS2')        #Shortage in surge demand
    nominal_pen2 = m.addVars(T_set_SS, W, lb=0, ub=GRB.INFINITY, obj=1.0/len(W), vtype =GRB.CONTINUOUS, name='zNPen2')        #shortage penalty for nominal demand
    surge_pen2 = m.addVars(T_set_SS,demand_nodes, W, lb=0, ub=GRB.INFINITY, obj=1.0/len(W), vtype =GRB.CONTINUOUS, name='zSPen2')          #shortage penalty for surge demand
    
    
    
    '''
    FIRST STAGE CONSTRAINTS
    Shipping decision at time t:
        - Shipping from supply nodes to demand nodes x: from i to j arriving at t'
        - Empty trucks shipping y: same as x
        - Amount that stays at a demand node (of what arrived): w
        - recourse: z
    '''
    for t in T_set_FS:
        for ci in net_nodes:
            l_f = t+tau_arcs[(ci,ci)]
            vname = ci+','+str(t)+','+ci+','+str(l_f)
            if ci in supply_nodes:
                x[ci,t,ci,l_f] = m.addVar( lb=0, ub=GRB.INFINITY, obj=0, vtype =GRB.CONTINUOUS, name='x[%s]' %(vname)) #loaded_flow
                y[ci,t,ci,l_f] = m.addVar( lb=0, ub=GRB.INFINITY, obj=0, vtype =GRB.CONTINUOUS, name='y[%s]' %(vname)) #emplty_flow
            vname = ci+','+ci+','+str(l_f-t)
            #delta[ci,ci,l_f-t] = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY,  obj=0, vtype=GRB.CONTINUOUS, name='travel_ind[%s]' %(vname))
            for cj in net_nodes:
                if ci != cj:
                    l_f = t+tau_arcs[(ci,cj)]
                    vname = ci+','+str(t)+','+cj+','+str(l_f)
                    if ci in supply_nodes:
                        x[ci,t,cj,l_f] = m.addVar(lb=0, ub=GRB.INFINITY, obj=xcoeff*tau_arcs[ci,cj], vtype =GRB.CONTINUOUS, name='x[%s]' %(vname)) #loaded_flow
                    if ci in demand_nodes and cj in supply_nodes:
                        y[ci,t,cj,l_f] = m.addVar(lb=0, ub=GRB.INFINITY, obj=xcoeff*tau_arcs[ci,cj], vtype =GRB.CONTINUOUS, name='y[%s]' %(vname)) #emplty_flow
    
    
    #RHS noise for the first stage (Place holder to easily change in the RH scheme)
    demand_b = {(t,i):(sum(scenarios[w][vft][t][i] for w in W)/len(W)) if t in scenarios[0][vft] else 0  for t in T_set_FS for i in demand_nodes}
    
    demand = m.addVars(T_set_FS, demand_nodes,lb=demand_b, ub=demand_b,  obj=0, vtype=GRB.CONTINUOUS, name='demand')
    m.update()
    
    #Demand nodes inventory
    for t in T_set_FS:
        if t == t_FS:
            m.addConstrs((I[t,i] == I0[i] + truck_cap * r0[i, 1] - nominal_demand[t,i] - demand[t,i] + z_nominal[t,i] + z_surge[t,i]  for i in demand_nodes), 'inv_demand[%i]' %(t))
            #Supply nodes inventory
            m.addConstrs((Is[t,i] == Is0[i] + fl_df['supply'][i] - truck_cap * x.sum(i,t, "*", "*")  for i in supply_nodes), 'inv_supply[%i]' %(t))
            #Delivered trucks go out
            m.addConstrs((r0[i, 1]  == y.sum(i, t,"*", "*") for i in demand_nodes), 'delivered_fuel_out[%i]' %(t))
            #Trucks availability
            m.addConstrs((x.sum(i,t,"*","*") + y.sum(i,t,"*","*") == g0[i,1]   for i in supply_nodes), 'truck_avail[%i]' %(t))
            #Update for state r and g 
            max_travel = max(Tau_max)
            m.addConstrs((r[t,i,l] == (r0[i,l+1] if l<max_travel else 0) + quicksum(x[j,t,i,t+l] for j in net_nodes if (j,t,i,t+l) in x) for i in demand_nodes for l in Tau_max), 'r_update[%i]' %(t))
            m.addConstrs((g[t,i,l] ==  (g0[i,l+1] if l<max_travel else 0) + quicksum(y[j,t,i,t+l] for j in net_nodes if (j,t,i,t+l) in y) for i in supply_nodes for l in Tau_max), 'g_update[%i]' %(t))
            m.addConstr(lhs=truck_counter[t], sense=GRB.EQUAL, rhs=r.sum(t,"*","*")+g.sum(t,"*","*"), name='trucks_checker[%i]' %(t))
        else:
            m.addConstrs((I[t,i] == I[t-DTM,i] + truck_cap * r[t-DTM,i, 1] - nominal_demand[t,i] - demand[t,i] + z_nominal[t,i] + z_surge[t,i]  for i in demand_nodes), 'inv_demand[%i]' %(t))
            #Supply nodes inventory
            m.addConstrs((Is[t,i] == Is[t-DTM,i] + fl_df['supply'][i] - truck_cap * x.sum(i,t, "*", "*")  for i in supply_nodes), 'inv_supply[%i]' %(t))
            #Delivered trucks go out
            m.addConstrs((r[t-DTM,i, 1]  == y.sum(i, t,"*", "*") for i in demand_nodes), 'delivered_fuel_out[%i]' %(t))
            #Trucks availability
            m.addConstrs((x.sum(i,t,"*","*") + y.sum(i,t,"*","*") == g[t-DTM,i,1]   for i in supply_nodes), 'truck_avail[%i]' %(t))
     
            #Update for state r and g 
            max_travel = max(Tau_max)
            m.addConstrs((r[t,i,l] == (r[t-DTM,i,l+1] if l<max_travel else 0) + quicksum(x[j,t,i,t+l] for j in net_nodes if (j,t,i,t+l) in x) for i in demand_nodes for l in Tau_max), 'r_update[%i]' %(t))
            m.addConstrs((g[t,i,l] ==  (g[t-DTM,i,l+1] if l<max_travel else 0) + quicksum(y[j,t,i,t+l] for j in net_nodes if (j,t,i,t+l) in y) for i in supply_nodes for l in Tau_max), 'g_update[%i]' %(t))
            m.addConstr(lhs=truck_counter[t], sense=GRB.EQUAL, rhs=r.sum(t,"*","*")+g.sum(t,"*","*"), name='trucks_checker[%i]' %(t))
        
        #objective function and related constraints
        #Nominal demand penalty
        total_nominal_demand = sum(nominal_demand[t,i] for i in demand_nodes)
        tnd90 = nombreak * total_nominal_demand
        #tnd75 = 0.90 * total_nominal_demand
        m1, m2, m3 = nomcoeff
        b1 = 0 #First piece intercept
        b2 = m1*tnd90 - m2*tnd90 #Second piece intercept
        #b3 = (m2*tnd75 + b2) - m3*tnd75
        nom_shortage_t = z_nominal.sum(t,"*")
        m.addConstr((nominal_pen[t]>= m1*nom_shortage_t + b1 ) , 'nominal_penalty[%i][1]' %(t))
        m.addConstr((nominal_pen[t]>= m2*nom_shortage_t + b2 ) , 'nominal_penalty[%i][2]' %(t))
        #m.addConstr((nominal_pen[t]>= m3*nom_shortage_t + b3 ) , 'nominal_penalty[%i][3]' %(t))
       
        #Surge demand penalty
        m1, m2 = surgecoeff    #Slopes of the piece-wise linear function
        surge_frac = surgebreak #Fraction of the demand at which the breakpoint is set
        m.addConstrs((surge_pen[t,i]>= m1*z_surge[t,i]   for i in demand_nodes), 'surge_penalty_piece_1[%i]' %(t))
        m.addConstrs((surge_pen[t,i]>= m2*z_surge[t,i] + (m1-m2)*surge_frac*demand[t,i]  for i in demand_nodes), 'surge_penalty_piece_2[%i]' %(t))
        
        
        #Constraint to define the relationship between demand and shortage
        m.addConstrs((z_surge[t,i] <= demand[t,i] for i in demand_nodes), 'shortage_demand_rel[%i]' %(t))
    
    '''
    SECOND STAGE CONSTRAINTS
    Shipping decision at time t:
        - Shipping from supply nodes to demand nodes x: from i to j arriving at t'
        - Empty trucks shipping y: same as x
        - Amount that stays at a demand node (of what arrived): w
        - recourse: z
    '''
    #RHS noise
    demand2 = {(t,i,w):(scenarios[w][vft][t][i]) if t in scenarios[w][vft] else 0 for w in W for i in demand_nodes for t in T_set_SS }
    for w in W:
        #print('Building scenario ' , w)
        for t in T_set_SS:
            for ci in net_nodes:
                l_f = t+tau_arcs[(ci,ci)]
                vname = ci+','+str(t)+','+ci+','+str(l_f)+','+str(w)
                if ci in supply_nodes:
                    x2[ci,t,ci,l_f,w] = m.addVar( lb=0, ub=GRB.INFINITY, obj=0, vtype =GRB.CONTINUOUS, name='x2[%s]' %(vname)) #loaded_flow
                    y2[ci,t,ci,l_f,w] = m.addVar( lb=0, ub=GRB.INFINITY, obj=0, vtype =GRB.CONTINUOUS, name='y2[%s]' %(vname)) #emplty_flow
                vname = ci+','+ci+','+str(l_f-t)
                #delta[ci,ci,l_f-t] = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY,  obj=0, vtype=GRB.CONTINUOUS, name='travel_ind[%s]' %(vname))
                for cj in net_nodes:
                    if ci != cj:
                        l_f = t+tau_arcs[(ci,cj)]
                        vname = ci+','+str(t)+','+cj+','+str(l_f)+','+str(w)
                        if ci in supply_nodes:
                            x2[ci,t,cj,l_f,w] = m.addVar(lb=0, ub=GRB.INFINITY, obj=0, vtype =GRB.CONTINUOUS, name='x2[%s]' %(vname)) #loaded_flow
                        if ci in demand_nodes and cj in supply_nodes:
                            y2[ci,t,cj,l_f,w] = m.addVar(lb=0, ub=GRB.INFINITY, obj=0, vtype =GRB.CONTINUOUS, name='y2[%s]' %(vname)) #emplty_flow
        
        
        
        #delta = m.addVars(net_nodes,net_nodes,Tau_max,lb=-GRB.INFINITY, ub=GRB.INFINITY,  obj=0, vtype=GRB.CONTINUOUS, name='travel_ind')
        m.update()
        
        #Demand nodes inventory
        for t in T_set_SS:
            if t == t_SS:
                m.addConstrs((I2[t,i,w] == I[last_t_FS,i] + truck_cap * r[last_t_FS, i, 1] - nominal_demand[t,i] - demand2[t,i,w] + z_nominal2[t,i,w] + z_surge2[t,i,w]  for i in demand_nodes), 'inv_demand[%i,%i]' %(t,w))
                #Supply nodes inventory
                m.addConstrs((Is2[t,i,w] == Is[last_t_FS,i] + fl_df['supply'][i] - truck_cap * x2.sum(i,t, "*", "*", w)  for i in supply_nodes), 'inv_supply[%i,%i]' %(t,w))
                #Delivered trucks go out
                m.addConstrs((r[last_t_FS, i, 1]  == y2.sum(i, t,"*", "*", w) for i in demand_nodes), 'delivered_fuel_out[%i,%i]' %(t,w))
                #Trucks availability
                m.addConstrs((x2.sum(i,t,"*","*", w) + y2.sum(i,t,"*","*" ,w) == g[last_t_FS, i, 1]   for i in supply_nodes), 'truck_avail[%i,%i]' %(t,w))
                #Update for state r and g 
                max_travel = max(Tau_max)
                m.addConstrs((r2[t,i,l,w] == (r[last_t_FS, i,l+1] if l<max_travel else 0) + quicksum(x2[j,t,i,t+l,w] for j in net_nodes if (j,t,i,t+l,w) in x2) for i in demand_nodes for l in Tau_max), 'r_update[%i,%i]' %(t,w))
                m.addConstrs((g2[t,i,l,w] ==  (g[last_t_FS, i,l+1] if l<max_travel else 0) + quicksum(y2[j,t,i,t+l,w] for j in net_nodes if (j,t,i,t+l,w) in y2) for i in supply_nodes for l in Tau_max), 'g_update[%i,%i]' %(t,w))
                #m.addConstr(lhs=truck_counter[t], sense=GRB.EQUAL, rhs=r.sum(t,"*","*")+g.sum(t,"*","*"), name='trucks_checker[%i,%i]' %(t,w))
            else:
                m.addConstrs((I2[t,i,w] == I2[t-DTM,i,w] + truck_cap * r2[t-DTM, i, 1,w] - nominal_demand[t,i] - demand2[t,i,w] + z_nominal2[t,i,w] + z_surge2[t,i,w]  for i in demand_nodes), 'inv_demand[%i,%i]' %(t,w))
                #Supply nodes inventory
                m.addConstrs((Is2[t,i,w] == Is2[t-DTM,i,w] + fl_df['supply'][i] - truck_cap * x2.sum(i,t, "*", "*", w)  for i in supply_nodes), 'inv_supply[%i,%i]' %(t,w))
                #Delivered trucks go out
                m.addConstrs((r2[t-DTM,i, 1,w]  == y2.sum(i, t,"*", "*",w) for i in demand_nodes), 'delivered_fuel_out[%i,%i]' %(t,w))
                #Trucks availability
                m.addConstrs((x2.sum(i,t,"*","*",w) + y2.sum(i,t,"*","*",w) == g2[t-DTM,i,1,w]   for i in supply_nodes), 'truck_avail[%i,%i]' %(t,w))
         
                #Update for state r and g 
                max_travel = max(Tau_max)
                m.addConstrs((r2[t,i,l,w] == (r2[t-DTM,i,l+1,w] if l<max_travel else 0) + quicksum(x2[j,t,i,t+l,w] for j in net_nodes if (j,t,i,t+l,w) in x2) for i in demand_nodes for l in Tau_max), 'r_update[%i,%i]' %(t,w))
                m.addConstrs((g2[t,i,l,w] ==  (g2[t-DTM,i,l+1,w] if l<max_travel else 0) + quicksum(y2[j,t,i,t+l,w] for j in net_nodes if (j,t,i,t+l,w) in y2) for i in supply_nodes for l in Tau_max), 'g_update[%i,%i]' %(t,w))
                #m.addConstr(lhs=truck_counter[t], sense=GRB.EQUAL, rhs=r.sum(t,"*","*")+g.sum(t,"*","*"), name='trucks_checker[%i]' %(t))
            
            #objective function and related constraints
            #Nominal demand penalty
            total_nominal_demand = sum(nominal_demand[t,i] for i in demand_nodes)
            tnd90 = nombreak * total_nominal_demand
            #tnd75 = 0.90 * total_nominal_demand
            m1, m2, m3 = nomcoeff
            b1 = 0 #First piece intercept
            b2 = m1*tnd90 - m2*tnd90 #Second piece intercept
            #b3 = (m2*tnd75 + b2) - m3*tnd75
            nom_shortage_t_w = z_nominal2.sum(t,"*",w)
            m.addConstr((nominal_pen2[t,w]>= m1*nom_shortage_t_w + b1 ) , 'nominal_penalty[%i,%i][1]' %(t,w))
            m.addConstr((nominal_pen2[t,w]>= m2*nom_shortage_t_w + b2 ) , 'nominal_penalty[%i,%i][2]' %(t,w))
            #m.addConstr((nominal_pen[t]>= m3*nom_shortage_t + b3 ) , 'nominal_penalty[%i][3]' %(t))
           
            #Surge demand penalty
            m1, m2 = surgecoeff    #Slopes of the piece-wise linear function
            surge_frac = surgebreak #Fraction of the demand at which the breakpoint is set
            m.addConstrs((surge_pen2[t,i,w]>= m1*z_surge2[t,i,w]   for i in demand_nodes), 'surge_penalty_piece_1[%i,%i]' %(t,w))
            m.addConstrs((surge_pen2[t,i,w]>= m2*z_surge2[t,i,w] + (m1-m2)*surge_frac*demand2[t,i,w]  for i in demand_nodes), 'surge_penalty_piece_2[%i,%i]' %(t,w))
            
            
            #Constraint to define the relationship between demand and shortage
            m.addConstrs((z_surge2[t,i,w] <= demand2[t,i,w] for i in demand_nodes), 'shortage_demand_rel[%i,%i]' %(t,w))
    
    
    
    #Travel time 
    #m.addConstrs((x[i,j,tj]<= trucks*delta[i,j,tj-t] for (i,j,tj) in x), 'travel_time')
   
    #Intra-stgge variables relation
    m.addConstrs((I_out[i] == I[out_t_FS,i] for i in demand_nodes), 'I_intra_rel')
    m.addConstrs((Is_out[i] == Is[out_t_FS,i] for i in supply_nodes), 'Is_intra_rel')
    m.addConstrs((r_out[i,l] == r[out_t_FS,i,l] for i in demand_nodes for l in Tau_max), 'r_intra_rel')
    m.addConstrs((g_out[i,l] == g[out_t_FS,i,l] for i in supply_nodes for l in Tau_max), 'g_intra_rel')
   
    m.update()
    
    
    
    
    for v in I0:
        I0[v].lb = nominal_demand[t_FS,v]*3
        I0[v].ub = nominal_demand[t_FS,v]*3
    for v in Is0:
        i = v
        Is0[v].lb = fl_df['supply'][v]*5
        Is0[v].ub = fl_df['supply'][v]*5
    for v in r0:
        r0[v].lb = 0
        r0[v].ub = 0
    for v in g0:
        g0[v].lb = 0
        g0[v].ub = 0
        
    for sn in supply_nodes:
        g0[sn,1].lb = trucks[sn] 
        g0[sn,1].ub = trucks[sn]
   
    
    #===========================================================================
    # for d in demand:
    #     t,i = d  # Set it infeasible so that can check that the real values are inputed
    #     demand[d].lb = 1
    #     demand[d].ub = -1
    #===========================================================================
        
    m.update()

    in_state = [v.VarName for v in I0.values()]
    in_state.extend((v.VarName for v in Is0.values()))
    in_state.extend((v.VarName for v in r0.values()))
    in_state.extend((v.VarName for v in g0.values()))
    
    out_state = [v.VarName for v in I_out.values()]
    out_state.extend((v.VarName for v in Is_out.values()))
    out_state.extend((v.VarName for v in r_out.values()))
    out_state.extend((v.VarName for v in g_out.values()))
    
    rhs_vars = [demand[ti].VarName for ti in demand]
    #Specifies a mapping of names between the out_state variables and the in_state of the next stage
    #Note that this should anticipate how in_state variables are going to be named in the next stage
    out_in_map = {out_state[in_i]:in_name for (in_i, in_name) in enumerate(in_state)}
    #rhs_vars = [demand[t_ini,i].VarName for i in demand_nodes]
    #rhs_vars.extend(v.VarName for v in delta.values())
    
    '''
    ADD REGULARIZATION TERMS: DISPERSE THE INVENTORY
    '''
    #mObj = m.getObjective()
    #mObjOri = m.getObjective()
    #mObj += sum(1e-7*I[iKey]*I[iKey] for iKey in I.keys())
    #mObj += sum(1e-9*I2[iKey]*I2[iKey]*2/len(W) for iKey in I2.keys())
    
    #m.setObjective(mObj,GRB.MINIMIZE)
    m.update()
    
    return m, in_state, out_state, rhs_vars, out_in_map

def two_stage_opt_extensive(t, T, instance_data, scenarios, prev_stage_state, tcoeff, xcoeff, 
                            nombreak = 0.90, nomcoeff = [1, 10, 10], surgebreak = 0.50, surgecoeff = [2, 5]):
    '''
        Two-stage optimization for a given moment t using a extensive formulation.
        Args:
            t (int): Current time in the RH.
            T (int): Last time period.
            instance_data (tuple): tuple with the output of load_florida_network function.
            scenarios (list of dict): List of scenarios organized as dictionaries.
            prev_stage_state (object): A container with information of the previous
                stage. It has the signature of the first output of this function.
        Returns:
            out_states_val (object): Output information to fit the next model in the RH scheme
            performance_out (list): list of performance metrics (e.g., shortfall)
            performance_base (list): list of performance metrics baseline (e.g., demand)
    '''
    tnow = time.time()
    # Build model for the extensive formulation
    DFSC_instance = instance_data
#    , first_order_obj 
    m, in_state, out_state, rhs_vars, out_in_map = DFSC_TwoStage_extensive(
        t, np.minimum(t + DELTA_T_STAGE, T), np.minimum(t + DELTA_T_STAGE + DELTA_T_SECOND_STAGE, T), DELTA_T_MODEL,
        DELTA_ROLLING, DELTA_NOTIFICATION, DFSC_instance, scenarios, tcoeff, xcoeff, nombreak, nomcoeff, surgebreak, surgecoeff)
    m.params.OutputFlag = 0
    m.params.Method = 1
    m.params.Threads = 1
    t_model_build = time.time() - tnow
    '''
    Modify first stage initial state
    '''
    for out_var_name in prev_stage_state:
        var = m.getVarByName(out_in_map[out_var_name])
        var.lb = prev_stage_state[out_var_name]
        var.ub = prev_stage_state[out_var_name]
    m.update()
    m.optimize()
    print(m.ObjVal)
    # obtain the stage solution
#    varList = m.getVars()
#    dataList = []
#    for iterL in range(len(varList)):
#        dataList.append([varList[iterL].VarName,varList[iterL].X])
    print('------------------------------------------------')
    '''
    Realization update and states re-computation
    Need to look for realizations in past forecast
    with respect to: t + DELTA_T_STAGE.
    '''
    demand_nodes = DFSC_instance[2]
    for intra_t in range(t, t + DELTA_ROLLING, DELTA_T_MODEL):
        if intra_t in scenarios[0]:
            for c in demand_nodes:
                v = m.getVarByName('demand[%i,%s]' % (intra_t, c))
                if np.abs(v.lb - scenarios[0][intra_t][intra_t][c]) > 1E-5:
                    pass
                    print('Cambio', (v.lb - scenarios[0][intra_t][intra_t][c]))
                v.lb = scenarios[0][intra_t][intra_t][c]
                v.ub = scenarios[0][intra_t][intra_t][c]
    
    for v in m.getVars():
        if not ('zS' in v.VarName or 'zN' in v.VarName or 'I[' in v.VarName or '2[' in v.VarName
                or 'demand' in v.VarName or 'I_intra[' in v.VarName):
            v.lb = v.X
            v.ub = v.X
        else:
            pass
    
#    m.setObjective(first_order_obj,GRB.MINIMIZE)
    m.update()
    m.reset()
    m.optimize()
    if m.Status == GRB.INFEASIBLE:
        m.computeIIS()
        m.write("MoreFunWithInfModels.ilp")
    #print((sum(m.getVarByName(vn) for vn in rhs_vars)))
    #print((sum(m.getVarByName(vn) for vn in rhs_vars)).getValue())
    print(m.ObjVal)
    #assert 1 ==2
    t_model_solve = time.time() - tnow - t_model_build
    
    out_states_val = {v_name: m.getVarByName(v_name).X for v_name in out_state}
    
    performance_out = [[], []]
    performance_base = [[], []]
    cost_out = [[], []]
    performance_var_name = ['zS', 'zN']
    demand_nodes = DFSC_instance[2]
    supply_nodes = DFSC_instance[3]
    Tau_max = DFSC_instance[5]
    nominal_demand = DFSC_instance[-1]
    I_t = {}
    Is_t = {}
    r_t = {}
    g_t = {}
    for intra_t in range(t, np.minimum(T, t + DELTA_ROLLING), DELTA_T_MODEL):
        shortage_intra_t = sum(
            m.getVarByName('%s[%i,%s]' % (performance_var_name[0], intra_t, c)).X for c in demand_nodes)
        surgeCost_t = sum(
            m.getVarByName('zSPen[%i,%s]' % (intra_t, c)).X for c in demand_nodes)
        performance_out[0].append(shortage_intra_t)
        cost_out[0].append(surgeCost_t)
        demand_t = sum(scenarios[0][intra_t][intra_t][c] for c in demand_nodes) if (intra_t in scenarios[0]) else 0
        performance_base[0].append(demand_t)
        
        shortage_intra_t = sum(
            m.getVarByName('%s[%i,%s]' % (performance_var_name[1], intra_t, c)).X for c in demand_nodes)
        nominalCost_t = m.getVarByName('zNPen[%i]' % (intra_t)).X
        performance_out[1].append(shortage_intra_t)
        cost_out[1].append(nominalCost_t)
        demand_t = sum(nominal_demand[intra_t, c] for c in demand_nodes)
        performance_base[1].append(demand_t)
        
        # return the solution
        for c in demand_nodes:
            I_t[intra_t,c] = m.getVarByName('I_intra[%i,%s]' % (intra_t, c)).X
            for tau in Tau_max:
                r_t[intra_t,c,tau] = m.getVarByName('r_intra[%i,%s,%i]' % (intra_t, c, tau)).X
        for c in supply_nodes:
            Is_t[intra_t,c] = m.getVarByName('Is_intra[%i,%s]' % (intra_t, c)).X
            for tau in Tau_max:
                g_t[intra_t,c,tau] = m.getVarByName('g_intra[%i,%s,%i]' % (intra_t, c, tau)).X
    t_model_out = time.time() - tnow - t_model_build - t_model_solve
    t_total = time.time() - tnow
    print('build=%8.2f solve=%8.2f out=%8.2f total=%8.2f' % (t_model_build, t_model_solve, t_model_out, t_total))
    print('roll short Surge  ', np.sum(performance_out[0]))
    print('roll short Nominal', np.sum(performance_out[1]))
    print('============================================================\n')
    
#    return (out_states_val, performance_out, performance_base, cost_out, dataList)
    return (out_states_val, performance_out, performance_base, cost_out, (I_t,Is_t,r_t,g_t))


def two_stage_results_processor(results_by_time, num_merics=2):
    t_roll = list(results_by_time.keys())
    t_roll.sort()
    out_metrics = [0] * num_merics
    for metric_by_time in range(num_merics):
        shortage = []
        for r in t_roll:
            shortage.extend(results_by_time[r][metric_by_time])
        out_metrics[metric_by_time] = np.array(shortage)
    return out_metrics

def transform_to_forecast(perfect_information):
    '''
        Transforms a dictionary holding perfect information about
        the demand and builds a forecast type dictionary.
        Args:
            perfect_information (dict): a dictionary with real demand.
        Returns:
            forecast_pi (dict): dictionary with a forecast syntax. 
    '''
    ts_list = list(perfect_information.keys())
    ts_list.sort()
    forecast_pi = {0: {t: {} for t in ts_list}}
    for ts_issued in ts_list:
        for ts_actual in ts_list:
            if ts_issued <= ts_actual:
                forecast_pi[0][ts_issued][ts_actual] = perfect_information[ts_actual]
    return forecast_pi


def setup_data_test(forecast, DELTA_HORIZON, DELTA_ROLLING, DELTA_T_STAGE, DELTA_T_SECOND_STAGE, DELTA_T_MODEL = 1, Tmax = 432):
    
    f0 = forecast[0]
    
    issued_times = list(f0.keys())
    issued_times.sort()
    prediction_times = set()
    for it in issued_times:
        pts_it = f0[it]
        for pt in pts_it:
            prediction_times.add(pt)
    prediction_times = list(prediction_times)
    prediction_times.sort()
    #global DELTA_T_STAGE
    #global DELTA_T_SECOND_STAGE
    forecast_delta = int((issued_times[1] - issued_times[0]).seconds / 3600)
    DELTA_T_STAGE = np.maximum(DELTA_T_STAGE, forecast_delta)
    
    pred1 = list(f0[issued_times[0]].keys())
    pred1.sort()
    DELTA_T_SECOND_STAGE = np.minimum(
        DELTA_T_SECOND_STAGE,
        int((pred1[-1] - pred1[0]).days) * 24 + int((pred1[-1] - pred1[0]).seconds / 3600))
    total_hours = int((prediction_times[-1] - prediction_times[0]).days) * 24 + int(
        (prediction_times[-1] - prediction_times[0]).seconds / 3600)
    EXTRA_TIME = 0  #1*DELTA_T_STAGE
    
    T_labels = [prediction_times[0] - timedelta(hours=k) for k in range(1, EXTRA_TIME + 1)]
    T_labels.extend(prediction_times)
    for pt in prediction_times:
        if pt != prediction_times[-1]:
            for k in range(1, forecast_delta, DELTA_T_MODEL):
                T_labels.append(pt + timedelta(hours=k))
    T_labels.sort()
    
    T_set = [i for (i, pt) in enumerate(T_labels)]
    
    ts_map = {ts: k for (k, ts) in enumerate(T_labels)}
    nfor = {
        w: {ts_map[it]: {ts_map[pt]: forecast[w][it][pt]
                         for pt in forecast[w][it]}
            for it in forecast[w]}
        for w in forecast
    }
    
    T_roll = [0]
    tr = 0
    max_roll = np.max(list(nfor[0].keys()))
    while tr < T_set[-1] and tr + DELTA_ROLLING <= max_roll and tr + DELTA_ROLLING <= Tmax:
        tr += DELTA_ROLLING
        T_roll.append(tr)
    
    realized_cum_demand = []
    increments = []
    for t in T_set:
        last_demand = 0 if t == 0 else realized_cum_demand[t - 1]
        if t in nfor[0]:
            sce_t = np.array([sum(nfor[k][t][t].values()) for k in nfor])
            realized_cum_demand.append(last_demand + sce_t.mean())
            increments.append(sce_t.mean())
        else:
            realized_cum_demand.append(last_demand)
            increments.append(0)
    realized_cum_demand = np.array(realized_cum_demand)
    
    return T_set, T_roll, T_labels, nfor, realized_cum_demand

def rolling_horizon_test(T_set, T_roll, T_labels, data, sample_path, optAlg, results_process, tcoeff, xcoeff, 
                         nombreak = 0.90, nomcoeff = [1, 10, 10], surgebreak = 0.50, surgecoeff = [2, 5]):
    '''
        Simulates a rolling horizon scheme in which the optimization engine is given
    
    Attributes:
        T_set (list): All time periods of the problem.
        T_roll(list): Time periods at which decision are made.
        T_labels (dict): maps time periods to a label (e.g. to a datetime object)
        sample_path (object): realization of the random variables to simulate the
                              rolling horizon.
        optAlg (func): A function that solve the problem and has the signature
                    Input:
                        t: current time period
                        T: max number of time periods
                        sample: a sample path (format free as it depends on how the user
                                codes this function)
                    Output:
                        model_output: returns the output of the model rean algorithm object that have access to math models
        
        results_process (func): A function to process the results. It receives as input
                    a dictionary with an algorithm object for each time period in the rolling
                    horizon.
                        
    '''
    performance = {}
    cost_out = {}
    performance_base = {}
    prev_roll_output = []
    sol_rec = {}
#    resultsOut = {}
    for (i_rol, t) in enumerate(T_roll):
        print('Solving t=', t, ' = ', T_labels[t])
        prev_t = T_roll[i_rol - 1] if i_rol > 0 else -1
        T_max = T_set[-1]
        alg_output = optAlg(t, T_max, data, sample_path, prev_roll_output, tcoeff, xcoeff, nombreak, nomcoeff, surgebreak, surgecoeff)
        assert type(alg_output) == tuple and len(alg_output) == 5, 'optAlg function must return a 2-dimensional tuple'
#        assert type(alg_output) == tuple and len(alg_output) == 5, 'optAlg function must return a 2-dimensional tuple'
        prev_roll_output = alg_output[0]
        performance[t] = alg_output[1]
        performance_base[t] = alg_output[2]
        cost_out[t] = alg_output[3]
        sol_rec[i_rol, t] = alg_output[4]
#        resultsOut[t] = alg_output[4]
#    return results_process(performance), results_process(performance_base), results_process(cost_out), resultsOut
    return results_process(performance), results_process(performance_base), results_process(cost_out), sol_rec


def run_rh_first_period(FR=50, H=48, F=24, R=12, N=6, modeStr="GEFS", tcoeff=1.001, xcoeff = 1e-4, 
                        nombreak = 0.90, nomcoeff = [1, 10, 10], surgebreak = 0.50, surgecoeff = [2, 5], costMode = 1):
    global DELTA_HORIZON
    global DELTA_T_STAGE
    global DELTA_T_SECOND_STAGE
    global DELTA_ROLLING
    global DELTA_NOTIFICATION
    '''
    =================================================================
    Reading forecast data
    =================================================================
    '''
    data_path = project_path + "/data/"
    path_to_forecast = None
    if "GEFS" in modeStr:
        if modeStr == "GEFS_c":
            path_to_forecast = data_path + 'predDemand/predDemand_concave_%i.p' % (FR)
        else:
            path_to_forecast = data_path + 'predDemand/predDemand_%i.p' % (FR)
    elif "GAVG" in modeStr:
        if modeStr == "GAVG_c":
            path_to_forecast = data_path + 'predDemand/predAvg_concave_%i.p' % (FR)
        else:
            path_to_forecast = data_path + 'predDemand/predAvg_%i.p' % (FR)
    elif "NDFD" in modeStr:
        if modeStr == "NDFD_c":
            path_to_forecast = data_path + 'predDemand/predNDFD_concave_%i.p' % (FR)
        else:
            path_to_forecast = data_path + 'predDemand/predNDFD_%i.p' % (FR)
    elif modeStr == "REAL":
        path_to_forecast = data_path + 'predDemand/predReal_%i.p' % (FR)
#    else:
#        path_to_forecast = data_path + 'predDemand/%s.p' % (FR)
    ''' Contains a dictionary with each replication of the enamble
        data[ensamble_number][issue_time][prediction_time] = array of predictions per county '''
    irma_data = pickle.load(open(path_to_forecast, 'rb'))
#    if FR == 'perfect_information':
#        irma_data = transform_to_forecast(irma_data)
    '''
    =================================================================
    Set up experiment
    =================================================================
    '''
    if H > F and R <= F:
        DELTA_HORIZON = H
        DELTA_T_STAGE = F
        DELTA_ROLLING = R
        DELTA_T_SECOND_STAGE = DELTA_HORIZON - DELTA_T_STAGE
        DELTA_T_MODEL = 1
        DELTA_NOTIFICATION = N
        T_set, T_roll, T_labels, scenarios, _ = setup_data_test(irma_data, DELTA_HORIZON, DELTA_ROLLING, DELTA_T_STAGE, DELTA_T_SECOND_STAGE)
        instance_data = load_florida_network(DELTA_T_MODEL, 0, T_set[-1], partition_network=False, zone=-1)
        if costMode == 1:
            instance_name = 'Test_FR%i_H%i_F%i_R%i_N%i_%s' % (FR, H, F, R, N, modeStr)
        else:
            instance_name = 'Test_FR%i_H%i_F%i_R%i_N%i_%s_%i' % (FR, H, F, R, N, modeStr, costMode)
        print(instance_name, '\n', T_roll, len(scenarios))
#        shortage_profiles, demand_profiles, cost_profiles, resultsOut = rolling_horizon_test(T_set, T_roll, T_labels, instance_data, scenarios,
#                                                             two_stage_opt_extensive, two_stage_results_processor)
#        save_obj = (shortage_profiles, demand_profiles, cost_profiles, resultsOut, T_set, T_labels)
        shortage_profiles, demand_profiles, cost_profiles, sol_profiles = rolling_horizon_test(T_set, T_roll, T_labels, instance_data, scenarios,
                                                     two_stage_opt_extensive, two_stage_results_processor, tcoeff, xcoeff, nombreak, nomcoeff, surgebreak, surgecoeff)
        save_obj = (shortage_profiles, demand_profiles, cost_profiles, sol_profiles, T_set, T_labels)
        with open('%s/output/%s.p' % (project_path, instance_name), 'wb') as fp:
            pickle.dump(save_obj, fp, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    #parser = argparse.ArgumentParser(description='Process some integers.')
    #from Utils.argv_parser import sys, parse_args
    #argv = sys.argv
    parser = argparse.ArgumentParser()
    parser.add_argument("FR")
    parser.add_argument("H")
    parser.add_argument("F")
    parser.add_argument("R")
    parser.add_argument("N")
    parser.add_argument("modeStr")
    parser.add_argument("tcoeff")
    parser.add_argument("xcoeff")
    parser.add_argument("costMode")
    
    args = parser.parse_args()
    #_, kwargs = parse_args(argv[1:])
    FR = int(args.FR)
    H = int(args.H)
    R = int(args.R)
    F = int(args.F)
    N = int(args.N)
    tcoeff = float(args.tcoeff)
    xcoeff = float(args.xcoeff)
    modeStr = args.modeStr
    costMode = int(args.costMode)
    if costMode == 1:
        nombreak = 0.90
        nomcoeff = [1, 10, 10]
        surgebreak = 0.50
        surgecoeff = [2, 5]
    else:
        nombreak = 0.90
        nomcoeff = [1, 100, 100]
        surgebreak = 0.50
        surgecoeff = [4, 25]
    
    run_rh_first_period(FR, H, F, R, N, modeStr,tcoeff,xcoeff, nombreak, nomcoeff, surgebreak, surgecoeff, costMode)
#    if 'FR' in kwargs:
#        FR = kwargs['FR']
#    if 'H' in kwargs:
#        H = kwargs['H']
#    if 'F' in kwargs:
#        F = kwargs['F']
#    if 'R' in kwargs:
#        R = kwargs['R']
#    if 'N' in kwargs:
#        N = kwargs['N']
#    if 'M' in kwargs:
#        modeStr = kwargs['M']
        
    #print(FR,H,F,R,N,modeStr)