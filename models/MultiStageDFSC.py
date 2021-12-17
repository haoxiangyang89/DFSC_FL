'''
Created on May 28, 2018

@author: dduque
'''
from os import path

import sys
import pickle
import numpy as np
from IO.NetworkBuilder import haversineEuclidean
#Import SDDP library
sys.path.append(path.abspath('/Users/dduque/Dropbox/WORKSPACE/SDDP'))
import CutSharing
from CutSharing.SDDP_Alg import SDDP  # @UnresolvedImport
from math import radians
from CutSharing.RandomnessHandler import RandomContainer, StageRandomVector   # @UnresolvedImport
from gurobipy import *
T=10
t_lenght = 3# Time of the length of one time period.

N=3
net_nodes = ['p%i' %(i) for i in range(4)]
demand_nodes = ['p%i' %(i) for i in range(3)]
supply_nodes = ['p3']
sypply = {'p3':40}
toy_demand = [[10,15,30],
              [5,9,10],
              [9,10,60]]

truck_cap = 230#Barrels/Truck
dem_pen = 1
trucks = 100#Per Port
truck_speed = 80 #km/h

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
#assert np.abs(sum(fl_df['supply'])-1.0)<1E-8, 'Supply mismatch %f' %(sum(fl_df['supply']))
#assert np.abs(sum(fl_df['demand'])-1.0)<1E-8,  'Demand mismatch %f' %(sum(fl_df['demand']))
fl_df['b'] = fl_df['supply'] - fl_df['demand']
fl_df['supply'] = (fl_df['supply']-fl_df['demand'])*100000
fl_df['demand'] = fl_df['demand']*100000
fractions_i = [0.5,1,1.5]
for i in range(N):
    demand_sce = 'demand_%i' %(i)
    fl_df[demand_sce] = fractions_i[i]*fl_df['demand']

demand_nodes = fl_df.loc[fl_df['supply']<0].index.tolist()
supply_nodes  = fl_df.loc[fl_df['supply']>0].index.tolist()
net_nodes = list()
#===============================================================================
# for _ in range(40):
#     demand_nodes.pop(0)
#===============================================================================
net_nodes.extend(demand_nodes)
net_nodes.extend(supply_nodes)
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
                re = rv_t.addRandomElememnt('demand[%s]' %(r), [fl_df['demand_%i' %(outcome)][r] for outcome in range(N)])
            else:
                re = rv_t.addRandomElememnt('demand[%s]' %(r), [0.0])
            rndVectors.append(rv_t)
        
        for ci in net_nodes:
            for cj in net_nodes:
                #===================================================================
                # for cj in fl_edges[ci]:
                #     if cj in fl_df.index:
                #===================================================================
                ij = ci+','+cj
                periods_ij = tau_arcs[(ci,cj)]
                if t>0:
                    rv_t.addRandomElememnt('travel_ind[%s,%i]' %(ij,periods_ij), [1 for outcome in range(N)])
                else:
                    rv_t.addRandomElememnt('travel_ind[%s,%i]' %(ij,periods_ij), [1])
                    
                    
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
    I = m.addVars(net_nodes, lb=0, ub=GRB.INFINITY, obj=0, vtype =GRB.CONTINUOUS, name='I')
    arrival_schedules = [t+l for l in Tau_max]
    r = m.addVars(net_nodes,Tau_max, lb=0, ub=GRB.INFINITY, obj=0, vtype =GRB.CONTINUOUS, name='r')
    g = m.addVars(net_nodes,Tau_max, lb=0, ub=GRB.INFINITY, obj=0, vtype =GRB.CONTINUOUS, name='g')
    
    I0 = m.addVars(net_nodes, lb=0, ub=GRB.INFINITY, obj=0, vtype =GRB.CONTINUOUS, name='I0')
    r0 = m.addVars(net_nodes,Tau_max, lb=0, ub=0, obj=0, vtype =GRB.CONTINUOUS, name='r0')
    g0 = m.addVars(net_nodes,Tau_max, lb=0, ub=0, obj=0, vtype =GRB.CONTINUOUS, name='g0')
    
    
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
    delta = tupledict()
    #for l_f in [t+l for l in Tau_max]:
    for ci in net_nodes:
        l_f = t+tau_arcs[(ci,ci)]
        vname = ci+','+ci+','+str(l_f)
        x[ci,ci,l_f] = m.addVar( lb=0, ub=GRB.INFINITY, obj=0, vtype =GRB.CONTINUOUS, name='x[%s]' %(vname)) #loaded_flow
        y[ci,ci,l_f] = m.addVar( lb=0, ub=GRB.INFINITY, obj=0, vtype =GRB.CONTINUOUS, name='y[%s]' %(vname)) #emplty_flow
        vname = ci+','+ci+','+str(l_f-t)
        delta[ci,ci,l_f-t] = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY,  obj=0, vtype=GRB.CONTINUOUS, name='travel_ind[%s]' %(vname))
        for cj in net_nodes:
            l_f = t+tau_arcs[(ci,cj)]
            vname = ci+','+cj+','+str(l_f)
            x[ci,cj,l_f] = m.addVar(lb=0, ub=GRB.INFINITY, obj=0.001, vtype =GRB.CONTINUOUS, name='x[%s]' %(vname)) #loaded_flow
            y[ci,cj,l_f] = m.addVar(lb=0, ub=GRB.INFINITY, obj=0.001, vtype =GRB.CONTINUOUS, name='y[%s]' %(vname)) #emplty_flow
                     
            #RHS noise
            vname = ci+','+cj+','+str(l_f-t)
            delta[ci,cj,l_f-t] = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY,  obj=0, vtype=GRB.CONTINUOUS, name='travel_ind[%s]' %(vname))

    
    #[t+l for l in Tau_max]
    #x = m.addVars(net_nodes,net_nodes,taus, lb=0, ub=GRB.INFINITY, obj=0, vtype =GRB.CONTINUOUS, name='x') #loaded_flow
    #y = m.addVars(net_nodes,net_nodes,taus, lb=0, ub=GRB.INFINITY, obj=0, vtype =GRB.CONTINUOUS, name='y') #emplty_flow
    
    w = m.addVars(net_nodes, lb=0, ub=GRB.INFINITY, obj=0, vtype =GRB.CONTINUOUS, name='w') #Delivered
    z = m.addVars(net_nodes, lb=0, ub=GRB.INFINITY, obj=dem_pen, vtype =GRB.CONTINUOUS, name='z') #emplty_flow
    
    #RHS noise
    demand = m.addVars(demand_nodes,lb=-GRB.INFINITY, ub=GRB.INFINITY,  obj=0, vtype=GRB.CONTINUOUS, name='demand')
    #delta = m.addVars(net_nodes,net_nodes,Tau_max,lb=-GRB.INFINITY, ub=GRB.INFINITY,  obj=0, vtype=GRB.CONTINUOUS, name='travel_ind')
    m.update()
    
    #Demand nodes inventory
    m.addConstrs((I[i]==I0[i]+truck_cap*w[i]-demand[i] + z[i]  for i in demand_nodes), 'inv_demand')
    #Supply nodes inventory
    m.addConstrs((I[i]==I0[i]+fl_df['supply'][i]-truck_cap*x.sum(i,"*","*")  for i in supply_nodes), 'inv_supply')
    #Delivered fuel
    m.addConstrs((w[i] + x.sum(i,"*","*") == r0[i,1] for i in demand_nodes), 'delivered_fuel')
    #Delivered trucks go out
    m.addConstrs((w[i] + g0[i,1] == y.sum(i,"*","*") for i in demand_nodes), 'delivered_fuel_out')
    #Trucks availability
    m.addConstrs((x.sum(i,"*","*") + y.sum(i,"*","*") == g0[i,1] + r0[i,1]  for i in supply_nodes), 'truck_avail')
    #m.addConstrs((x.sum(i,"*","*") + y.sum(i,"*","*") <= g0[i,1]   for i in supply_nodes), 'truck_avail')
    
    #m.addConstrs((x.sum(i,"*","*") >= r0[i,1] for i in supply_nodes), 'full_truck_avail')
    #m.addConstrs((y.sum(i,"*","*")<=g0[i,1] for i in supply_nodes), 'truck_avail')  + y.sum(i,"*","*")
    
    #Update for state r and g 
    max_travel = max(Tau_max)
    m.addConstrs((r[i,l] == (r0[i,l+1] if l<max_travel else 0) + quicksum(x[j,i,t+l] for j in net_nodes if (j,i,t+l) in x) for i in net_nodes for l in Tau_max), 'r_update')
    m.addConstrs((g[i,l] ==  (g0[i,l+1] if l<max_travel else 0) + quicksum(y[j,i,t+l] for j in net_nodes if (j,i,t+l) in y) for i in net_nodes for l in Tau_max), 'g_update')
    
    #Travel time 
    m.addConstrs((x[i,j,tj]<= trucks*delta[i,j,tj-t] for (i,j,tj) in x), 'travel_time')
    m.update()
    
    if t == 0:
        for v in I0:
            I0[v].lb = 0
            I0[v].ub = 0
        for v in r0:
            r0[v].lb = 0
            r0[v].ub = 0
        for v in g0:
            g0[v].lb = 0
            r0[v].ub = 0
        for sn in supply_nodes:
            g0[sn,1].lb = trucks #Assumes 20 trucks per port
            g0[sn,1].ub = trucks
        for sn in demand_nodes:
            g0[sn,1].lb = 0 #Assumes 20 trucks per port
            g0[sn,1].ub = 0
        
        for d in demand:
            demand[d].lb = 0
            demand[d].ub = 0
        
        for d in delta:
            delta[d].lb = 1
            delta[d].ub = 1
            
    m.update()
            
    in_state = [v.VarName for v in I0.values()]
    in_state.extend((v.VarName for v in r0.values()))
    in_state.extend((v.VarName for v in g0.values()))
    out_state = [v.VarName for v in I.values()]
    out_state.extend((v.VarName for v in r.values()))
    out_state.extend((v.VarName for v in g.values()))
    rhs_vars = [v.VarName for v in demand.values()]
    rhs_vars.extend(v.VarName for v in delta.values())
    
    return m, in_state, out_state, rhs_vars
    
    


if __name__ == '__main__':
    CutSharing.options['max_iter'] = 100000
    CutSharing.options['in_sample_ub'] = 30
    CutSharing.options['multicut'] = False
    alg =  SDDP(T, DFSC_model_builder,DFSC_random_builder, lower_bound=0)
    #alg.stage_problems[0].printModel()
    alg.run(instance_name='DieselFuel-Florida')
    alg.simulate_policy(1000, DFSC_random_builder())
