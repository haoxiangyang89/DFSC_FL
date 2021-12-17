'''
Created on Jan 15, 2019

@author: dduque

Modified on Mar 10, 2019
 
'''
import sys
from os import path
#Import SDDP library
sys.path.append(path.abspath('/Users/haoxiangyang/Desktop/Git/SDDP'))
from CutSharing.RandomnessHandler import RandomContainer, StageRandomVector  # @UnresolvedImport
from gurobipy import Model, quicksum, GRB, tupledict
import numpy as np






def DFSC_TwoStage_random_builder(T, intra_time_periods, sce_forecast,DFSC_instance):
    '''
    Function to construct the random of the problem
    Args:
        T (int): Number of stages (should be 2 for two-stage SP regardless of the intra time
        intra_time_intervals (list of tuples): A list of intra-time intervals or each stage
        sce_forecast (dict): forecast of the demand
        DFSC_instance (tuple): Tuple with various elements that define the isntnace 
    ''' 
    fl_df, fl_edges, demand_nodes, supply_nodes, net_nodes, Tau_max, tau_arcs, trucks, truck_cap, nominal_demand = DFSC_instance
    n_sce = len(sce_forecast)
    rc = RandomContainer()
    rndVectors = []
    tnow = intra_time_periods[0][0]
    #Need to use a valid forecast (i.e., that is avaiable at tnow). Often this value is tnow itself
    valid_forecast_time = np.max(np.array([i*(i<=tnow)*(i in sce_forecast[0]) for i in range(tnow+1)]))
    assert valid_forecast_time in sce_forecast[0], 'Forecast time is not valid'
    for t in range(T):
        rv_t = StageRandomVector(t)
        rc.append(rv_t)
        for (i,r) in enumerate(demand_nodes):
            for intra_t in intra_time_periods[t]:
                tnow_forecasts = None
                if intra_t in sce_forecast[0][valid_forecast_time]:
                    tnow_forecasts = np.array([sce_forecast[s][valid_forecast_time][intra_t][r] for s in range(n_sce)])
                else: # There is not a forecast at this intra_t
                    tnow_forecasts = np.zeros(n_sce)
                if t==0: #First stage
                    rv_t.addRandomElememnt('demand[%i,%s]' %(intra_t,r), [tnow_forecasts.mean()])
                else:    #Second stage
                    rv_t.addRandomElememnt('demand[%i,%s]' %(intra_t,r), tnow_forecasts)
            rndVectors.append(rv_t)
    return rc


def DFSC_TwoStage(t_ini , maxT, delta_t_model, DFSC_instance):
    '''
    Builds the model to run SDDP on a two stage model
    that comprise multiple time periods in both stages.
    '''
    assert t_ini<maxT
    T_set = range(t_ini,maxT, delta_t_model)
    #Load network data
    fl_df, fl_edges, demand_nodes, supply_nodes, net_nodes, Tau_max, tau_arcs, trucks, truck_cap, nominal_demand = DFSC_instance 
    last_t = T_set[-1]
    DTM = delta_t_model
    '''
    Time-space model of Florida modeling 
    '''
    m = Model('DFSC_otracle_%i_' %(t_ini))
    
    
    '''
    State variables:
        - Inventory at every node (I)
        - Inventory of loaded in-transit trucks (r)
        - Inventory of empty in-transit trucks  (g)
    '''
    I_out = m.addVars(demand_nodes, lb=0, ub=GRB.INFINITY, obj=0, vtype =GRB.CONTINUOUS, name='I')
    Is_out = m.addVars(supply_nodes, lb=0, ub=GRB.INFINITY, obj=0, vtype =GRB.CONTINUOUS, name='Is')
    r_out = m.addVars(demand_nodes,Tau_max, lb=0, ub=GRB.INFINITY, obj=0, vtype =GRB.CONTINUOUS, name='r')
    g_out = m.addVars(supply_nodes,Tau_max, lb=0, ub=GRB.INFINITY, obj=0, vtype =GRB.CONTINUOUS, name='g')
    

    I0 = m.addVars(demand_nodes, lb=0, ub=0, obj=0, vtype =GRB.CONTINUOUS, name='I0')
    Is0 = m.addVars(supply_nodes, lb=0, ub=0, obj=0, vtype =GRB.CONTINUOUS, name='Is0')
    r0 = m.addVars(demand_nodes,Tau_max, lb=0, ub=0, obj=0, vtype =GRB.CONTINUOUS, name='r0')
    g0 = m.addVars(supply_nodes,Tau_max, lb=0, ub=0, obj=0, vtype =GRB.CONTINUOUS, name='g0')
    
    #Intra-stage variables
    I = m.addVars(T_set,demand_nodes, lb=0, ub=GRB.INFINITY, obj=0, vtype =GRB.CONTINUOUS, name='I_intra')
    Is = m.addVars(T_set,supply_nodes, lb=0, ub=GRB.INFINITY, obj=0, vtype =GRB.CONTINUOUS, name='Is_intra')
    r = m.addVars(T_set,demand_nodes,Tau_max, lb=0, ub=GRB.INFINITY, obj=0, vtype =GRB.CONTINUOUS, name='r_Intra')
    g = m.addVars(T_set,supply_nodes,Tau_max, lb=0, ub=GRB.INFINITY, obj=0, vtype =GRB.CONTINUOUS, name='g_Intra')
    
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
    
    z_nominal = m.addVars(T_set, demand_nodes, lb=0, ub=nominal_demand, obj=0, vtype =GRB.CONTINUOUS, name='zN')    #Shortage in nominal demand
    z_surge = m.addVars(T_set, demand_nodes, lb=0, ub=GRB.INFINITY, obj=0, vtype =GRB.CONTINUOUS, name='zS')        #Shortage in surge demand
    nominal_pen = m.addVars(T_set, lb=0, ub=GRB.INFINITY, obj=1, vtype =GRB.CONTINUOUS, name='zNPen')        #shortage penalty for nominal demand
    surge_pen = m.addVars(T_set,demand_nodes, lb=0, ub=GRB.INFINITY, obj=1, vtype =GRB.CONTINUOUS, name='zSPen')          #shortage penalty for surge demand
    #RHS noise
    demand = m.addVars(T_set, demand_nodes,lb=-GRB.INFINITY, ub=GRB.INFINITY,  obj=0, vtype=GRB.CONTINUOUS, name='demand')
    #delta = m.addVars(net_nodes,net_nodes,Tau_max,lb=-GRB.INFINITY, ub=GRB.INFINITY,  obj=0, vtype=GRB.CONTINUOUS, name='travel_ind')
    m.update()
    
    #Demand nodes inventory
    for t in T_set:
        if t == t_ini:
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
        tnd90 = 0.90 * total_nominal_demand
        #tnd75 = 0.90 * total_nominal_demand
        m1, m2, m3 = 1, 10, 10
        b1 = 0 #First piece intercept
        b2 = m1*tnd90 - m2*tnd90 #Second piece intercept
        #b3 = (m2*tnd75 + b2) - m3*tnd75
        nom_shortage_t = z_nominal.sum(t,"*")
        m.addConstr((nominal_pen[t]>= m1*nom_shortage_t + b1 ) , 'nominal_penalty[%i][1]' %(t))
        m.addConstr((nominal_pen[t]>= m2*nom_shortage_t + b2 ) , 'nominal_penalty[%i][2]' %(t))
        #m.addConstr((nominal_pen[t]>= m3*nom_shortage_t + b3 ) , 'nominal_penalty[%i][3]' %(t))
       
        #Surge demand penalty
        m1, m2 = 2, 5    #Slopes of the piece-wise linear function
        surge_frac = 0.5 #Fraction of the demand at which the breakpoint is set
        m.addConstrs((surge_pen[t,i]>= m1*z_surge[t,i]   for i in demand_nodes), 'surge_penalty_piece_1[%i]' %(t))
        m.addConstrs((surge_pen[t,i]>= m2*z_surge[t,i] + (m1-m2)*surge_frac*demand[t,i]  for i in demand_nodes), 'surge_penalty_piece_2[%i]' %(t))
        
        
        #Constraint to define the relationship between demand and shortage
        m.addConstrs((z_surge[t,i] <= demand[t,i] for i in demand_nodes), 'shortage_demand_rel[%i]' %(t))
    #Travel time 
    #m.addConstrs((x[i,j,tj]<= trucks*delta[i,j,tj-t] for (i,j,tj) in x), 'travel_time')
   
    #Intra-stgge variables relation
    m.addConstrs((I_out[i] == I[last_t,i] for i in demand_nodes), 'I_intra_rel')
    m.addConstrs((Is_out[i] == Is[last_t,i] for i in supply_nodes), 'Is_intra_rel')
    m.addConstrs((r_out[i,l] == r[last_t,i,l] for i in demand_nodes for l in Tau_max), 'r_intra_rel')
    m.addConstrs((g_out[i,l] == g[last_t,i,l] for i in supply_nodes for l in Tau_max), 'g_intra_rel')
   
    m.update()
    
    
    
    
    for v in I0:
        I0[v].lb = fl_df['demand'][v]*3
        I0[v].ub = fl_df['demand'][v]*3
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
   
    
    for d in demand:
        t,i = d  # Set it infeasible so that can check that the real values are inputed
        demand[d].lb = 1
        demand[d].ub = -1
        

    #m.addConstr(lhs=x.sum("*",1,"*","*"), sense='>', rhs=1200, name='asasas')   
    #m.addConstr(lhs=z.sum(2,"*"), sense='<', rhs=5000, name='asasas')   
    m.update()

    #m.computeIIS()
    #m.write('oracle_model.ilp')
    in_state = [v.VarName for v in I0.values()]
    in_state.extend((v.VarName for v in Is0.values()))
    in_state.extend((v.VarName for v in r0.values()))
    in_state.extend((v.VarName for v in g0.values()))
    
    #===========================================================================
    # out_state = [v.VarName for v in I.subset(last_t).values()]
    # out_state.extend((v.VarName for v in Is.subset(last_t).values()))
    # out_state.extend((v.VarName for v in r.subset(last_t).values()))
    # out_state.extend((v.VarName for v in g.subset(last_t).values()))
    #===========================================================================
    out_state = [v.VarName for v in I_out.values()]
    out_state.extend((v.VarName for v in Is_out.values()))
    out_state.extend((v.VarName for v in r_out.values()))
    out_state.extend((v.VarName for v in g_out.values()))
    demand
    rhs_vars = [demand[ti].VarName for ti in demand]
    #Specifies a mapping of names between the out_state variables and the in_state of the next stage
    #Note that this should anticipate how in_state variables are going to be named in the next stage
    #out_in_map = {out_state[in_i]:in_name for (in_i, in_name) in enumerate(in_state)}
    #rhs_vars = [demand[t_ini,i].VarName for i in demand_nodes]
    #rhs_vars.extend(v.VarName for v in delta.values())
    
    return m, in_state, out_state, rhs_vars




def DFSC_TwoStage_extensive(t_FS , t_SS, t_max, delta_t_model, t_roll, t_notification,  DFSC_instance, scenarios):
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
    
    '''
    State variables:
        - Inventory at every node (I)
        - Inventory of loaded in-transit trucks (r)
        - Inventory of empty in-transit trucks  (g)
    '''
    I_out = m.addVars(demand_nodes, lb=0, ub=GRB.INFINITY, obj=0, vtype =GRB.CONTINUOUS, name='I')
    Is_out = m.addVars(supply_nodes, lb=0, ub=GRB.INFINITY, obj=0, vtype =GRB.CONTINUOUS, name='Is')
    r_out = m.addVars(demand_nodes,Tau_max, lb=0, ub=GRB.INFINITY, obj=0, vtype =GRB.CONTINUOUS, name='r')
    g_out = m.addVars(supply_nodes,Tau_max, lb=0, ub=GRB.INFINITY, obj=0, vtype =GRB.CONTINUOUS, name='g')
    

    I0 = m.addVars(demand_nodes, lb=0, ub=0, obj=0, vtype =GRB.CONTINUOUS, name='I0')
    Is0 = m.addVars(supply_nodes, lb=0, ub=0, obj=0, vtype =GRB.CONTINUOUS, name='Is0')
    r0 = m.addVars(demand_nodes,Tau_max, lb=0, ub=0, obj=0, vtype =GRB.CONTINUOUS, name='r0')
    g0 = m.addVars(supply_nodes,Tau_max, lb=0, ub=0, obj=0, vtype =GRB.CONTINUOUS, name='g0')
    
    #Intra-stage variables FS
    I = m.addVars(T_set_FS, demand_nodes, lb=0, ub=GRB.INFINITY, obj=0, vtype =GRB.CONTINUOUS, name='I_intra')
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
    np.random.seed(0)
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
                        x[ci,t,cj,l_f] = m.addVar(lb=0, ub=GRB.INFINITY, obj=np.random.uniform(0,1e-4), vtype =GRB.CONTINUOUS, name='x[%s]' %(vname)) #loaded_flow
                    if ci in demand_nodes and cj in supply_nodes:
                        y[ci,t,cj,l_f] = m.addVar(lb=0, ub=GRB.INFINITY, obj=np.random.uniform(0,1e-4), vtype =GRB.CONTINUOUS, name='y[%s]' %(vname)) #emplty_flow
    
    
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
        tnd90 = 0.90 * total_nominal_demand
        #tnd75 = 0.90 * total_nominal_demand
        m1, m2, m3 = 1, 10, 10
        b1 = 0 #First piece intercept
        b2 = m1*tnd90 - m2*tnd90 #Second piece intercept
        #b3 = (m2*tnd75 + b2) - m3*tnd75
        nom_shortage_t = z_nominal.sum(t,"*")
        m.addConstr((nominal_pen[t]>= m1*nom_shortage_t + b1 ) , 'nominal_penalty[%i][1]' %(t))
        m.addConstr((nominal_pen[t]>= m2*nom_shortage_t + b2 ) , 'nominal_penalty[%i][2]' %(t))
        #m.addConstr((nominal_pen[t]>= m3*nom_shortage_t + b3 ) , 'nominal_penalty[%i][3]' %(t))
       
        #Surge demand penalty
        m1, m2 = 2, 5    #Slopes of the piece-wise linear function
        surge_frac = 0.5 #Fraction of the demand at which the breakpoint is set
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
            tnd90 = 0.90 * total_nominal_demand
            #tnd75 = 0.90 * total_nominal_demand
            m1, m2, m3 = 1, 10, 10
            b1 = 0 #First piece intercept
            b2 = m1*tnd90 - m2*tnd90 #Second piece intercept
            #b3 = (m2*tnd75 + b2) - m3*tnd75
            nom_shortage_t_w = z_nominal2.sum(t,"*",w)
            m.addConstr((nominal_pen2[t,w]>= m1*nom_shortage_t_w + b1 ) , 'nominal_penalty[%i,%i][1]' %(t,w))
            m.addConstr((nominal_pen2[t,w]>= m2*nom_shortage_t_w + b2 ) , 'nominal_penalty[%i,%i][2]' %(t,w))
            #m.addConstr((nominal_pen[t]>= m3*nom_shortage_t + b3 ) , 'nominal_penalty[%i][3]' %(t))
           
            #Surge demand penalty
            m1, m2 = 2, 5    #Slopes of the piece-wise linear function
            surge_frac = 0.5 #Fraction of the demand at which the breakpoint is set
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
#, mObjOri

