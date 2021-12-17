'''
Created on May 28, 2018

@author: dduque
'''
from os import path
from mpl_toolkits.basemap import Basemap
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.pyplot import plot
matplotlib.use('agg') #To plot on linux
import sys
import pickle
import numpy as np
import pandas as pd
from IO.NetworkBuilder import haversineEuclidean
import gmplot
import webbrowser

#Import SDDP library
sys.path.append(path.abspath('/Users/dduque/Dropbox/WORKSPACE/SDDP'))
import CutSharing
from CutSharing.SDDP_Alg import SDDP  # @UnresolvedImport
from math import radians
from CutSharing.RandomnessHandler import RandomContainer, StageRandomVector   # @UnresolvedImport
from gurobipy import *
T=11
t_lenght = 6# Time of the length of one time period.

N=2

supply_factor = 1.0
Totaltrucks = 1000*supply_factor
truck_cap = 230#Barrels/Truck
dem_pen = 1
truck_speed = 80 #km/h

#%%
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

fl_df['supply'] = fl_df['supply']*10000*supply_factor
fl_df['demand'] = fl_df['demand']*100000

demand_nodes = fl_df.loc[fl_df['demand']>0].index.tolist()

south_point = {'lat':25.46793, 'lon': -80.4691}
north_point = {'lat':30.27842, 'lon': -82.6402}
div_slope = (north_point['lat'] - south_point['lat'])/(north_point['lon'] - south_point['lon'])
div_inter = north_point['lat'] - div_slope*north_point['lon']
print(div_slope, ' ', div_inter)
for s in range(N):
    demand_sce = 'demand_%i' %(s)
    fl_df[demand_sce] = 0
    if s == 0: #Scenario in which there is a demand surge in the west
        for dn in demand_nodes:
            lat_dn = fl_df['latitude'][dn]
            lon_dn = fl_df['longitude'][dn]
            if lat_dn <= div_slope*lon_dn +div_inter:
                fl_df.loc[dn,demand_sce] = fl_df.loc[dn,'demand']*1.5
            else:
                fl_df.loc[dn,demand_sce] =  fl_df.loc[dn,'demand']
    elif s == 1:#Scenario in which there is a demand surge in the east
        for dn in demand_nodes:
            lat_dn = fl_df['latitude'][dn]
            lon_dn = fl_df['longitude'][dn]
            if lat_dn <= div_slope*lon_dn +div_inter:               
                fl_df.loc[dn,demand_sce] =  fl_df.loc[dn,'demand']
            else:
                fl_df.loc[dn,demand_sce] =  fl_df.loc[dn,'demand']*1.5

outcomes_prob = np.array([0.99, 0.01])


#fractions_i = [1,1.25]
#for i in range(N):
#    demand_sce = 'demand_%i' %(i)
#    fl_df[demand_sce] = fractions_i[i]*fl_df['demand']





net_nodes = set()
#===============================================================================
# for _ in range(45):
#     demand_nodes.pop(0)
#===============================================================================
net_nodes.update(demand_nodes)
net_nodes.update(supply_nodes)
net_nodes = list(net_nodes)
print(len(demand_nodes), len(supply_nodes))

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
print('max travel time ' , max_t_time, Tau_max)

def DFSC_random_builder():
    '''
    Function to construct the random of the problem
    ''' 
    rc = RandomContainer()
    rndVectors = []
    for t in range(0,T):
        rv_t = StageRandomVector(t)
        rc.append(rv_t)
        for (i,r) in enumerate(demand_nodes):
            if t>0:
                re = rv_t.addRandomElememnt('demand[%i,%s]' %(t,r), [fl_df['demand_%i' %(outcome)][r] for outcome in range(N)], outcomes_prob)
            else:
                re = rv_t.addRandomElememnt('demand[%i,%s]' %(t,r), [0.0])
            rndVectors.append(rv_t)
        
        #=======================================================================
        # for ci in net_nodes:
        #     for cj in net_nodes:
        #         #===================================================================
        #         # for cj in fl_edges[ci]:
        #         #     if cj in fl_df.index:
        #         #===================================================================
        #         ij = ci+','+cj
        #         periods_ij = tau_arcs[(ci,cj)]
        #         if t>0:
        #             rv_t.addRandomElememnt('travel_ind[%s,%i]' %(ij,periods_ij), [1 for outcome in range(N)])
        #         else:
        #             rv_t.addRandomElememnt('travel_ind[%s,%i]' %(ij,periods_ij), [1])
        #=======================================================================
                    
                    
        #=======================================================================
        # for l in travel_ind:
        #     if t>0:
        #         inds = travel_ind[l]
        #         for ij in inds:
        #             rv_t.addRandomElememnt('travel_ind[%s,%i]' %(ij,l), inds[ij])
        #         pass
        #     else:
        #         t0_inds= {1:1,2:0}
        #         for ij in travel_ind[l]:
        #             rv_t.addRandomElememnt('travel_ind[%s,%i]' %(ij,l), [t0_inds[l]])
        #=======================================================================
            
    rc.preprocess_randomness()
    return rc



def DFSC_model_builder(t):
    '''
    Builds a gurobi model for the stage given as a parameter
    Args:
        t(int): stage id 
    '''
    
    '''
    Time-space model of Florida modeling 
    '''
    m = Model('DFSC')
    
    
    '''
    State variables:
        - Inventory at every node (I)
        - Inventory of loaded in-transit trucks (r)
        - Inventory of empty in-transit trucks  (g)
    '''
    I = m.addVars(demand_nodes, lb=0, ub=GRB.INFINITY, obj=0, vtype =GRB.CONTINUOUS, name='I')
    Is = m.addVars(supply_nodes, lb=0, ub=GRB.INFINITY, obj=0, vtype =GRB.CONTINUOUS, name='Is')
    arrival_schedules = [t+l for l in Tau_max]
    r = m.addVars(demand_nodes,Tau_max, lb=0, ub=GRB.INFINITY, obj=0, vtype =GRB.CONTINUOUS, name='r')
    g = m.addVars(supply_nodes,Tau_max, lb=0, ub=GRB.INFINITY, obj=0, vtype =GRB.CONTINUOUS, name='g')
    
    I0 = m.addVars(demand_nodes, lb=0, ub=0, obj=0, vtype =GRB.CONTINUOUS, name='I0')
    Is0 = m.addVars(supply_nodes, lb=0, ub=0, obj=0, vtype =GRB.CONTINUOUS, name='Is0')
    r0 = m.addVars(demand_nodes,Tau_max, lb=0, ub=0, obj=0, vtype =GRB.CONTINUOUS, name='r0')
    g0 = m.addVars(supply_nodes,Tau_max, lb=0, ub=0, obj=0, vtype =GRB.CONTINUOUS, name='g0')
    
    #truck counter 
    truck_counter = m.addVar(lb=0,ub=GRB.INFINITY,obj=0,vtype=GRB.CONTINUOUS, name='trucks')
    '''
    Shipping decision at time t:
        - Shipping from supply nodes to demand nodes x: from i to j arriving at t'
        - Empty trucks shipping y: same as x
        - Amount that stays at a demand node (of what arrived): w
        - recourse: z
    '''
    taus = [t+tau_arcs[(ci,cj)] for ci in net_nodes for cj in net_nodes]
    x = tupledict()
    y = tupledict()
    #delta = tupledict()
    #for l_f in [t+l for l in Tau_max]:
    for ci in net_nodes:
        l_f = t+tau_arcs[(ci,ci)]
        vname = ci+','+ci+','+str(l_f)
        if ci in supply_nodes:
            x[ci,ci,l_f] = m.addVar( lb=0, ub=GRB.INFINITY, obj=0, vtype =GRB.CONTINUOUS, name='x[%s]' %(vname)) #loaded_flow
            y[ci,ci,l_f] = m.addVar( lb=0, ub=GRB.INFINITY, obj=0, vtype =GRB.CONTINUOUS, name='y[%s]' %(vname)) #emplty_flow
        vname = ci+','+ci+','+str(l_f-t)
        #delta[ci,ci,l_f-t] = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY,  obj=0, vtype=GRB.CONTINUOUS, name='travel_ind[%s]' %(vname))
        for cj in net_nodes:
            if ci != cj:
                l_f = t+tau_arcs[(ci,cj)]
                vname = ci+','+cj+','+str(l_f)
                if ci in supply_nodes:
                    x[ci,cj,l_f] = m.addVar(lb=0, ub=GRB.INFINITY, obj=0, vtype =GRB.CONTINUOUS, name='x[%s]' %(vname)) #loaded_flow
                if ci in demand_nodes and cj in supply_nodes:
                    y[ci,cj,l_f] = m.addVar(lb=0, ub=GRB.INFINITY, obj=0, vtype =GRB.CONTINUOUS, name='y[%s]' %(vname)) #emplty_flow
                         
                #RHS noise
                #vname = ci+','+cj+','+str(l_f-t)
                #delta[ci,cj,l_f-t] = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY,  obj=0, vtype=GRB.CONTINUOUS, name='travel_ind[%s]' %(vname))

    
    #[t+l for l in Tau_max]
    #x = m.addVars(net_nodes,net_nodes,taus, lb=0, ub=GRB.INFINITY, obj=0, vtype =GRB.CONTINUOUS, name='x') #loaded_flow
    #y = m.addVars(net_nodes,net_nodes,taus, lb=0, ub=GRB.INFINITY, obj=0, vtype =GRB.CONTINUOUS, name='y') #emplty_flow
    
    z = m.addVars(demand_nodes, lb=0, ub=GRB.INFINITY, obj=0, vtype =GRB.CONTINUOUS, name='z') #emplty_flow
    pen = m.addVars(demand_nodes, lb=0, ub=GRB.INFINITY, obj=1, vtype =GRB.CONTINUOUS, name='z') #shortage penalty
    #RHS noise
    demand = m.addVars([t],demand_nodes,lb=-GRB.INFINITY, ub=GRB.INFINITY,  obj=0, vtype=GRB.CONTINUOUS, name='demand')
    #delta = m.addVars(net_nodes,net_nodes,Tau_max,lb=-GRB.INFINITY, ub=GRB.INFINITY,  obj=0, vtype=GRB.CONTINUOUS, name='travel_ind')
    m.update()
    
    #Demand nodes inventory
    m.addConstrs((I[i] == I0[i] + truck_cap * r0[i, 1] - demand[t,i] + z[i]  for i in demand_nodes), 'inv_demand')
    #Supply nodes inventory
    m.addConstrs((Is[i] == Is0[i] + fl_df['supply'][i] - truck_cap * x.sum(i, "*", "*")  for i in supply_nodes), 'inv_supply')
    #Delivered trucks go out
    m.addConstrs((r0[i, 1]  == y.sum(i, "*", "*") for i in demand_nodes), 'delivered_fuel_out')
    #Trucks availability
    m.addConstrs((x.sum(i,"*","*") + y.sum(i,"*","*") == g0[i,1]   for i in supply_nodes), 'truck_avail')
    #m.addConstrs((x.sum(i,"*","*") + y.sum(i,"*","*") <= g0[i,1]   for i in supply_nodes), 'truck_avail')
    
    #m.addConstrs((x.sum(i,"*","*") >= r0[i,1] for i in supply_nodes), 'full_truck_avail')
    #m.addConstrs((y.sum(i,"*","*")<=g0[i,1] for i in supply_nodes), 'truck_avail')  + y.sum(i,"*","*")
    
    #Update for state r and g 
    max_travel = max(Tau_max)
    m.addConstrs((r[i,l] == (r0[i,l+1] if l<max_travel else 0) + quicksum(x[j,i,t+l] for j in net_nodes if (j,i,t+l) in x) for i in demand_nodes for l in Tau_max), 'r_update')
    m.addConstrs((g[i,l] ==  (g0[i,l+1] if l<max_travel else 0) + quicksum(y[j,i,t+l] for j in net_nodes if (j,i,t+l) in y) for i in supply_nodes for l in Tau_max), 'g_update')
    m.addConstr(lhs=truck_counter, sense=GRB.EQUAL, rhs=r.sum()+g.sum(), name='trucks_checker')
    #Travel time 
    #m.addConstrs((x[i,j,tj]<= trucks*delta[i,j,tj-t] for (i,j,tj) in x), 'travel_time')
    m.update()
    
    #objective function and related constraints
    #piece 1: pen >= 2 - 3 (d_bar-z)/d_bar
    m.addConstrs((pen[i]>= 2 - 3*(fl_df['demand'][i]- z[i])/fl_df['demand'][i] for i in demand_nodes), 'penalty_piece_1')
    #piece 2: pen >= 1 - (d_bar-z)/d_bar
    m.addConstrs((pen[i]>= 1 - 1*(fl_df['demand'][i]- z[i])/fl_df['demand'][i] for i in demand_nodes), 'penalty_piece_1')
    
    
    if t == 0:
        for v in I0:
            I0[v].lb = fl_df['demand'][v]*3
            I0[v].ub = fl_df['demand'][v]*3
        for v in Is0:
            Is0[v].lb = fl_df['supply'][v]*10
            Is0[v].ub = fl_df['supply'][v]*10
        for v in r0:
            r0[v].lb = 0
            r0[v].ub = 0
        for v in g0:
            g0[v].lb = 0
            g0[v].ub = 0
        
        for sn in supply_nodes:
            g0[sn,1].lb = trucks[sn] 
            g0[sn,1].ub = trucks[sn]
        
        
        for d in demand:
            demand[d].lb = 0
            demand[d].ub = 0
        

            
    m.update()
            
    in_state = [v.VarName for v in I0.values()]
    in_state.extend((v.VarName for v in Is0.values()))
    in_state.extend((v.VarName for v in r0.values()))
    in_state.extend((v.VarName for v in g0.values()))
    out_state = [v.VarName for v in I.values()]
    out_state.extend((v.VarName for v in Is.values()))
    out_state.extend((v.VarName for v in r.values()))
    out_state.extend((v.VarName for v in g.values()))
    rhs_vars = [v.VarName for v in demand.values()]
    #rhs_vars.extend(v.VarName for v in delta.values())
    
    return m, in_state, out_state, rhs_vars
    
    
    
def DFSC_oracle_model(t_ini , maxT):
    '''
    Builds an oracle 
    '''
    
    T_set = range(t_ini,maxT)
    '''
    Time-space model of Florida modeling 
    '''
    m = Model('DFSC_otracle_%i' %(t_ini))
    
    
    '''
    State variables:
        - Inventory at every node (I)
        - Inventory of loaded in-transit trucks (r)
        - Inventory of empty in-transit trucks  (g)
    '''
    I = m.addVars(T_set,demand_nodes, lb=0, ub=GRB.INFINITY, obj=0, vtype =GRB.CONTINUOUS, name='I')
    Is = m.addVars(T_set,supply_nodes, lb=0, ub=GRB.INFINITY, obj=0, vtype =GRB.CONTINUOUS, name='Is')
    r = m.addVars(T_set,demand_nodes,Tau_max, lb=0, ub=GRB.INFINITY, obj=0, vtype =GRB.CONTINUOUS, name='r')
    g = m.addVars(T_set,supply_nodes,Tau_max, lb=0, ub=GRB.INFINITY, obj=0, vtype =GRB.CONTINUOUS, name='g')
    
    I0 = m.addVars(demand_nodes, lb=0, ub=0, obj=0, vtype =GRB.CONTINUOUS, name='I0')
    Is0 = m.addVars(supply_nodes, lb=0, ub=0, obj=0, vtype =GRB.CONTINUOUS, name='Is0')
    r0 = m.addVars(demand_nodes,Tau_max, lb=0, ub=0, obj=0, vtype =GRB.CONTINUOUS, name='r0')
    g0 = m.addVars(supply_nodes,Tau_max, lb=0, ub=0, obj=0, vtype =GRB.CONTINUOUS, name='g0')
    
    #truck counter 
    truck_counter = m.addVars(T_set,lb=0,ub=GRB.INFINITY,obj=0,vtype=GRB.CONTINUOUS, name='trucks')
    '''
    Shipping decision at time t:
        - Shipping from supply nodes to demand nodes x: from i to j arriving at t'
        - Empty trucks shipping y: same as x
        - Amount that stays at a demand node (of what arrived): w
        - recourse: z
    '''
    x = tupledict()
    y = tupledict()
    
    for t in T_set:
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
                        x[ci,t,cj,l_f] = m.addVar(lb=0, ub=GRB.INFINITY, obj=0, vtype =GRB.CONTINUOUS, name='x[%s]' %(vname)) #loaded_flow
                    if ci in demand_nodes and cj in supply_nodes:
                        y[ci,t,cj,l_f] = m.addVar(lb=0, ub=GRB.INFINITY, obj=0, vtype =GRB.CONTINUOUS, name='y[%s]' %(vname)) #emplty_flow
    
    z = m.addVars(T_set, net_nodes, lb=0, ub=GRB.INFINITY, obj=0, vtype =GRB.CONTINUOUS, name='z') #emplty_flow
    pen = m.addVars(T_set,demand_nodes, lb=0, ub=GRB.INFINITY, obj=1, vtype =GRB.CONTINUOUS, name='z') #shortage penalty
    #RHS noise
    demand = m.addVars(T_set, demand_nodes,lb=-GRB.INFINITY, ub=GRB.INFINITY,  obj=0, vtype=GRB.CONTINUOUS, name='demand')
    #delta = m.addVars(net_nodes,net_nodes,Tau_max,lb=-GRB.INFINITY, ub=GRB.INFINITY,  obj=0, vtype=GRB.CONTINUOUS, name='travel_ind')
    m.update()
    
    #Demand nodes inventory
    for t in T_set:
        if t == t_ini:
            m.addConstrs((I[t,i] == I0[i] + truck_cap * r0[i, 1] - demand[t,i] + z[t,i]  for i in demand_nodes), 'inv_demand[%i]' %(t))
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
            m.addConstrs((I[t,i] == I[t-1,i] + truck_cap * r[t-1,i, 1] - demand[t,i] + z[t,i]  for i in demand_nodes), 'inv_demand[%i]' %(t))
            #Supply nodes inventory
            m.addConstrs((Is[t,i] == Is[t-1,i] + fl_df['supply'][i] - truck_cap * x.sum(i,t, "*", "*")  for i in supply_nodes), 'inv_supply[%i]' %(t))
            #Delivered trucks go out
            m.addConstrs((r[t-1,i, 1]  == y.sum(i, t,"*", "*") for i in demand_nodes), 'delivered_fuel_out[%i]' %(t))
            #Trucks availability
            m.addConstrs((x.sum(i,t,"*","*") + y.sum(i,t,"*","*") == g[t-1,i,1]   for i in supply_nodes), 'truck_avail[%i]' %(t))
     
            #Update for state r and g 
            max_travel = max(Tau_max)
            m.addConstrs((r[t,i,l] == (r[t-1,i,l+1] if l<max_travel else 0) + quicksum(x[j,t,i,t+l] for j in net_nodes if (j,t,i,t+l) in x) for i in demand_nodes for l in Tau_max), 'r_update[%i]' %(t))
            m.addConstrs((g[t,i,l] ==  (g[t-1,i,l+1] if l<max_travel else 0) + quicksum(y[j,t,i,t+l] for j in net_nodes if (j,t,i,t+l) in y) for i in supply_nodes for l in Tau_max), 'g_update[%i]' %(t))
            m.addConstr(lhs=truck_counter[t], sense=GRB.EQUAL, rhs=r.sum(t,"*","*")+g.sum(t,"*","*"), name='trucks_checker[%i]' %(t))
        
        #objective function and related constraints
        #piece 1: pen >= 2 - 3 (d_bar-z)/d_bar
        m.addConstrs((pen[t,i]>= 2 - 3*(fl_df['demand'][i]- z[t,i])/fl_df['demand'][i] for i in demand_nodes), 'penalty_piece_1')
        #piece 2: pen >= 1 - (d_bar-z)/d_bar
        m.addConstrs((pen[t,i]>= 1 - 1*(fl_df['demand'][i]- z[t,i])/fl_df['demand'][i] for i in demand_nodes), 'penalty_piece_1')
    #Travel time 
    #m.addConstrs((x[i,j,tj]<= trucks*delta[i,j,tj-t] for (i,j,tj) in x), 'travel_time')
    m.update()
    
    
    
    
    for v in I0:
        I0[v].lb = fl_df['demand'][v]*3
        I0[v].ub = fl_df['demand'][v]*3
    for v in Is0:
        i = v
        Is0[v].lb = fl_df['supply'][v]*10
        Is0[v].ub = fl_df['supply'][v]*10
    for v in r0:
        r0[v].lb = 0
        r0[v].ub = 0
    for v in g0:
        g0[v].lb = 0
        g0[v].ub = 0
        
    for sn in supply_nodes:
        g0[sn,1].lb = trucks[sn] 
        g0[sn,1].ub = trucks[sn]
   
    
    for d in demand:
        t,i = d
        demand[d].lb = np.mean([fl_df['demand_%i' %(outcome)][i] for outcome in range(N)])
        demand[d].ub = np.mean([fl_df['demand_%i' %(outcome)][i] for outcome in range(N)])
        

    #m.addConstr(lhs=x.sum("*",1,"*","*"), sense='>', rhs=1200, name='asasas')   
    #m.addConstr(lhs=z.sum(2,"*"), sense='<', rhs=5000, name='asasas')   
    m.update()

    #m.computeIIS()
    #m.write('oracle_model.ilp')
    in_state = [v.VarName for v in I0.values()]
    in_state.extend((v.VarName for v in Is0.values()))
    in_state.extend((v.VarName for v in r0.values()))
    in_state.extend((v.VarName for v in g0.values()))
    rhs_vars = [demand[t_ini,i].VarName for i in demand_nodes]
    #rhs_vars.extend(v.VarName for v in delta.values())
    
    return m, in_state, rhs_vars
    

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


mode = 'build_policy11' 
#mode = 'plot'
if __name__ == '__main__':
    irma_sample_path, samplepath_time_stamps = build_irma_sample_path('../data/factor_data.p') 
    if mode =='build_policy':
        CutSharing.options['max_iter'] =2000
        CutSharing.options['in_sample_ub'] = 50
        CutSharing.options['multicut'] = True
        CutSharing.options['max_stage_with_oracle'] = 20
        CutSharing.options['max_iters_oracle_ini'] = 10
        CutSharing.options['cut_selector'] = CutSharing.LAST_CUTS_SELECTOR#CutSharing.SLACK_BASED_CUT_SELECTOR #CutSharing.LAST_CUTS_SELECTOR
        CutSharing.options['max_cuts_last_cuts_selector'] = 5000
        
        T = len(irma_sample_path)
        alg = SDDP(T, DFSC_model_builder, DFSC_random_builder, lower_bound=0)
        #alg.add_oracle_model(DFSC_oracle_model)
        alg.run(instance_name='DieselFuel-Florida - Model 2 - T%i' %(T), dynamic_sampling=False)
        
        sample_path, forward_sol, models = alg.simulate_single_scenario(DFSC_random_builder(), sample_path=irma_sample_path)
        unmet_demand = [[m.getVarByName('z[%s]' %(dn)).X for dn in demand_nodes] for m in models]
        policy_on_irma = {'sample_path': sample_path, 'unmet_demand':unmet_demand}
        with open('../data/irma_policy.p', 'wb') as fp:
            pickle.dump(policy_on_irma, fp, protocol=pickle.HIGHEST_PROTOCOL)
    
    
    
    track  = pickle.load(open('../data/trackdata.p', 'rb'))
    
    forecast  = pickle.load(open('../data/forecastdata.p', 'rb'))
    
      
    plot_type = 'abs' #abs
    with open('../data/irma_policy.p', 'rb') as fp:
        policy_on_irma = pickle.load(fp)
        sample_path = policy_on_irma['sample_path']
        unmet_demand = policy_on_irma['unmet_demand']
        print(len(unmet_demand[2]), demand_nodes)
        T = len(unmet_demand)
        print(T, plot_type)
        #Plot in any mode 
        max_radius = 0.2
        min_radius = 0.01
        delta_radius =  max_radius- min_radius
        max_demand =  np.max([sample_path[t]['demand[%i,%s]' %(t,dn)] for t in range(1,T) for dn in demand_nodes])
        min_demand =  np.min([sample_path[t]['demand[%i,%s]' %(t,dn)] for t in range(1,T) for dn in demand_nodes])
        delta_demand = max_demand - min_demand
        if plot_type == 'rel':
            max_demand =  {dn:np.max([sample_path[t]['demand[%i,%s]' %(t,dn)] for t in range(1,T)])for dn in demand_nodes}
            min_demand =  {dn:np.min([sample_path[t]['demand[%i,%s]' %(t,dn)] for t in range(1,T)])for dn in demand_nodes}
            delta_demand = {dn:(max_demand[dn] - min_demand[dn]) for dn in demand_nodes}
        
        print(delta_demand)               
        for t in range(1,T):
            print('Ploting stage %i' %(t))
            #unmet_dem = [models[t].getVarByName('z[%s]' %(dn)).X for dn in demand_nodes]
            time_stamp_t = samplepath_time_stamps[t]
            fig,ax = plt.subplots()
            m = Basemap(projection="mill", #miller est une projection connu
                llcrnrlat =23,#24.5,#lower left corner latitude 25.276103, -88.272764
                llcrnrlon =-88.3,
                urcrnrlat =31.0, #upper right lat 30.475935, -79.742215
                urcrnrlon =-79,#-79.7,
                resolution = 'l', ax=ax) #c croud par defaut, l low , h high , f full 
            m.drawcoastlines() #dessiner les lignes
            m.drawcountries()
            m.drawstates()
            #m.drawcounties(color='blue')
            
            #Print timestamp
            x,y=m(25,-87)
            plt.annotate(str(time_stamp_t), xy=(0.05, 0.05), xycoords='axes fraction')
            #ts_txt = plt.text(x, y, , fontsize=1000)
            #ax.add_patch(ts_txt)
            
            #Draw circles to represent demand
            for (i,dn) in enumerate(demand_nodes):
                x,y=m(fl_df['longitude'][dn],fl_df['latitude'][dn])
                d_i = sample_path[t]['demand[%i,%s]' %(t,dn)] 
                r = min_radius
                if plot_type == 'rel':
                    if delta_demand[dn] > 1E-8:
                        r = min_radius + delta_radius*(d_i-min_demand[dn])/delta_demand[dn]
                elif plot_type == 'abs':
                    if delta_demand > 1E-8:
                        r = min_radius + delta_radius*np.sqrt((d_i-min_demand)/delta_demand)
                
                x2,y2 = m(fl_df['longitude'][dn],fl_df['latitude'][dn]+r)
                #dn_color = 'red' if r>0.15 else ('orange' if r<=0.15 and r>0.1 else 'green')
                circle1 = plt.Circle((x, y), (y2-y), color='blue',fill=True)
                ax.add_patch(circle1)
                
                #draw shortage
                if unmet_demand[t][dn]  > 0:
                    r_s= min_radius + (r-min_radius)*np.sqrt((unmet_demand[t][dn])/d_i)
                    x3,y3 = m(fl_df['longitude'][dn],fl_df['latitude'][dn]+r_s)
                    circle2 = plt.Circle((x, y), (y3-y), color='red',fill=True)
                    ax.add_patch(circle2)
                
                #unmet_dem[i]/sample_path[t]['demand[%i,%s]' %(t,dn)]
            
            #Draw hurricane track
            x = None
            y = None
            last_stamp = None
            track_time_stamps = list(track.keys())
            track_time_stamps.sort()
            #print(track_time_stamps)
            #print(time_stamp_t)
            for (ts_i, track_ts) in enumerate(track_time_stamps):
                if track_ts <= time_stamp_t:
                    lat_ts = track[track_ts][0]
                    lon_ts = track[track_ts][1]
                    if x!=None: #draw line
                        last_x,last_y=m(track[track_time_stamps[ts_i-1]][1],track[track_time_stamps[ts_i-1]][0])
                        new_x,new_y=m(lon_ts,lat_ts)
                        xs = [last_x, new_x]
                        ys = [last_y, new_y]
                        track_line = plt.Line2D(xs,ys,color='black')
                        ax.add_line(track_line)
                    x,y=m(lon_ts,lat_ts)
                    x2,y2 = m(lon_ts,lat_ts+0.05)
                    circle3 = plt.Circle((x, y), (y2-y), color='black',fill=True)
                    ax.add_patch(circle3)
                
                
            #Draw hurricane forecast cone
            try:
                x = None
                y = None
                last_stamp = None
                forecast_t = forecast[time_stamp_t]
                print(forecast_t)
                forecast_time_stamps = list(forecast_t.keys())
                forecast_time_stamps.sort()
                for (fts_i, fts) in enumerate(forecast_time_stamps):
                    lat_ts = forecast_t[fts][0]
                    lon_ts = forecast_t[fts][1]
                    f_diam = forecast_t[fts][2]/60 # https://en.wikipedia.org/wiki/Nautical_mile
                   
                    x,y=m(lon_ts,lat_ts)
                    x2,y2 = m(lon_ts,lat_ts+f_diam)
                    circle4 = plt.Circle((x, y), (y2-y), color='black',fill=True, alpha=0.3)
                    ax.add_patch(circle4)
            except:
                print('No more forecasts')
                
            
            
            filename = '../output/heat_test_t%i.pdf' %(t)
            complete_file_name = os.path.realpath(filename)
            pp = PdfPages(complete_file_name)
            pp.savefig(fig)
            pp.close()
            #plt.show()
    
    
    
    #alg.simulate_policy(1000, DFSC_random_builder())
