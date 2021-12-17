'''
Created on Feb 17, 2018

@author: dduque hyang

This module implements a nominal model of a network flow in Florida
representing fuel
'''
import pickle
import gmplot
import numpy as np
import pandas as pd
from gurobipy import *
import webbrowser, os, gmplot
from math import radians, sin, cos, asin,sqrt
from IO.NetworkBuilder import haversineEuclidean
from colour import Color

def solve_nominal_model(netwrok_file, distance_func, visualize = True):
    '''
    Constructs and solve a nominal model given an instance of
    a network.
    Args:
        network_file (str): file name of pickled tuple with a 
        pandas data frame with the nodes information and a dictionary
        with the edges of the network.
        distance_function (func): a function to compute the distance 
            between two locations. The signature requires two lat-lon
            pairs.
    '''
    
    '''
    ===========================================================================
    Data preparation
    '''
    fl_df, fl_edges = pickle.load(open(netwrok_file, 'rb'))
    fl_df = fl_df.set_index('County')
    total_population = sum(fl_df.Population)
    fl_df['demand'] = fl_df['Population']/total_population
    fl_df['supply'] = 0
    
    #===========================================================================
    #Set ports supply
    # Tampa  = Hillsborough County
    # 42.5%
    fl_df.loc['Hillsborough County', 'supply'] = 0.425
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
    assert np.abs(sum(fl_df['supply'])-1.0)<1E-8, 'Supply mismatch %f' %(sum(fl_df['supply']))
    assert np.abs(sum(fl_df['demand'])-1.0)<1E-8,  'Demand mismatch %f' %(sum(fl_df['demand']))
    fl_df['b'] = fl_df['supply'] - fl_df['demand']
    print(fl_df[fl_df.b>0])
    '''
    ===========================================================================
    Model
    '''
    
    m = Model('FDM_nominal')
    #Variables
    flow = tupledict()
    for c_i in fl_edges:
        for c_j in fl_edges[c_i]:
            if c_j != c_i and c_j in fl_df.index:
                dist_ij = distance_func(radians(fl_df.latitude[c_i]), radians(fl_df.longitude[c_i]),
                                        radians(fl_df.latitude[c_j]), radians(fl_df.longitude[c_j]),)
                flow[c_i,c_j] = m.addVar(lb=0, ub=1, obj=dist_ij, vtype=GRB.CONTINUOUS, 
                                         name='flow[%s,%s]' %(c_i,c_j))
                
    m.update()
    #Balance constraint
    m.addConstrs((flow.sum(ci,'*')-flow.sum('*',ci)==fl_df.b[ci]    for ci in fl_df.index), name='balance')            
    m.update()
    
    m.optimize()
    solution = {} #arc = flow
    max_flow = 0
    for ij in flow:
        if flow[ij].X > 0:
            solution[ij]=flow[ij].X
            if max_flow<solution[ij]:
                max_flow = solution[ij]
    print('Demand in barrels')
    print(sum(fl_df.demand*465000/(11600/42)))
    
    if visualize:
        #Draw Solution 
        lat_bar = sum(fl_df['latitude'])/len(fl_df['latitude'])
        lon_bar = sum(fl_df['longitude'])/len(fl_df['longitude'])
        filename = '../output/florida_network_min_cost_flow.html'
        gmap = gmplot.GoogleMapPlotter(lat_bar,lon_bar, 6)
        gmap.scatter(fl_df['latitude'][fl_df.supply>0], fl_df['longitude'][fl_df.supply>0], color = 'red', marker = True , size = 3)
        
        red = Color("green")
        colors = list(red.range_to(Color("red"),len(solution)))
        sol_arcs = [ij for ij in solution]
        sol_arcs.sort(key = lambda x:solution[x])
        for (k,ij) in enumerate(sol_arcs):
            color_ij = colors[k].get_hex()
            lats = [fl_df['latitude'][ij[0]], fl_df['latitude'][ij[1]]]
            lons = [fl_df['longitude'][ij[0]], fl_df['longitude'][ij[1]]]
            gmap.plot(lats, lons, color=color_ij, edge_width=5)
        gmap.draw(filename)
        webbrowser.open('file://' + os.path.realpath(filename))
               
            
            
    
    
    
    
if __name__ == '__main__':
    solve_nominal_model(netwrok_file='../data/floridaNetObj.p', distance_func=haversineEuclidean)
    