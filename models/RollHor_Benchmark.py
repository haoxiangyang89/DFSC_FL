'''
Created on Sep 18, 2018

@author: dduque
Implements a benchmark based on rolling horizon scheme based on NOAA advisories. 
'''
import numpy as np
import pickle
from IO.NetworkBuilder import load_florida_network
from gurobipy import Model, GRB, tupledict, quicksum
import datetime
#from mpl_toolkits.basemap import Basemap
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from models.MultiStageDFSC import demand_nodes
import os


def get_current_advisory(adv_date):
    #Get advisory
    pass

def read_pass_decision(file_name):
    pass



def create_full_model( t_max, fl_df, fl_edges, demand_nodes, supply_nodes, net_nodes, Tau_max, tau_arcs, trucks, truck_cap):
    '''
    Creates and solve a optimization model for the distribution of diesel.
    
    Args:
        t_max (int): End of the planning horizon
    '''
    t_now = 1
    T_set = [t for t in np.arange(t_now, t_max + 1)]
    T_set_large = T_set.copy()
    T_set_large.insert(0, t_now-1)
    '''
    Time-space model of Florida modeling 
    '''
    m = Model('DFSC_otracle_%i' %(t_now))
    
    m.params.OutputFlag = 0
    m.params.NumericFocus = 0
    m.params.Method = 0
    
    '''
    State variables:
        - Inventory at every node (I)
        - Inventory of loaded in-transit trucks (r)
        - Inventory of empty in-transit trucks  (g)
    '''
    I = m.addVars(T_set_large,demand_nodes, lb=0, ub=GRB.INFINITY, obj=0, vtype =GRB.CONTINUOUS, name='I')
    Is = m.addVars(T_set_large,supply_nodes, lb=0, ub=GRB.INFINITY, obj=0, vtype =GRB.CONTINUOUS, name='Is')
    r = m.addVars(T_set_large,demand_nodes,Tau_max, lb=0, ub=GRB.INFINITY, obj=0, vtype =GRB.CONTINUOUS, name='r')
    g = m.addVars(T_set_large,supply_nodes,Tau_max, lb=0, ub=GRB.INFINITY, obj=0, vtype =GRB.CONTINUOUS, name='g')
    
    
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
        #Demand nodes inventory
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

    days_of_inv = 3
    for i in demand_nodes:
        #8 comes from 24h/3h times steps
        I[t_now-1,i].ub = np.round(days_of_inv*8*fl_df['demand'][i],0)
        I[t_now-1,i].lb = np.round(days_of_inv*8*fl_df['demand'][i],0)
        for tau in Tau_max:
            r[t_now-1, i, tau].ub =0
    for i in supply_nodes:
        Is[t_now-1,i].ub = np.round(fl_df['supply'][i]*8*10, 0 )
        Is[t_now-1,i].lb = np.round(fl_df['supply'][i]*8*10, 0 )
        for tau in Tau_max:
            g[t_now-1, i, tau].ub =trucks[i]
        
        
    for d in demand:
        t,i = d
        demand[d].lb = fl_df['demand'][i]#demand_forecast[d]
        demand[d].ub = fl_df['demand'][i]#demand_forecast[d]
        
    m.update()
    m.optimize()
    return m, x, y, demand, z, I

def solve_time_period(m, x, y, demand, z, I , t, prev_t, T_max, demand_forecast_t, realized_demand ):
    '''
    Solves the model of a given time period fixing all previous decisions
    and demand realizations and changing the demand forecast for future stages.
    Args:
        m (GRBModel): Gurobi model with all time periods
        t (int): stage of the model to be modeled
        prev_t (int): previous stage that was solved 
        demand_forecast_t(TBD): demand forecast for all future stages. If t=T, is an empty dictionary.
        realized_demand(dict str-float): realized demand of stage t-1
    '''
    
    '''
    ========================================================================
    Fix future forecasts
    '''
    for tf in demand_forecast_t:
        if tf<= T_max:
            demand_tf =demand_forecast_t[tf] #Dictionary with keys for all counties
            for i in demand_tf:
                demand[tf,i].lb = demand_tf[i]
                demand[tf,i].ub = demand_tf[i]
    
    if t > 1:
        '''
        ========================================================================
        Fix all decisions that took place before the current stage
        '''
        for (i,t_ini,j,t_end) in x:
            if prev_t<=t_ini<t:
                x_val = x[i,t_ini,j,t_end].x
                if -1E-8 < x_val < 0:
                    x_val = 0 
                assert x_val>=0, '%s %i %f' %(i,t , x[i,t_ini,j,t_end].x)
                x[i,t_ini,j,t_end].lb = x_val
                x[i,t_ini,j,t_end].ub = x_val
        for (i,t_ini,j,t_end) in y:
            if prev_t<= t_ini<t:
                y_val = y[i,t_ini,j,t_end].x
                if -1E-8 < y_val < 0:
                    y_val = 0 
                assert y_val>=0, '%s %i %15.10f' %(i,t , y[i,t_ini,j,t_end].x)
                y[i,t_ini,j,t_end].lb = y_val
                y[i,t_ini,j,t_end].ub = y_val
        '''
        ========================================================================
        Fix all demands that had been realized
        '''
        for tp in range(prev_t, t): #[prev_t, ... , t-1]
            for i in realized_demand[tp]:
                demand[tp,i].lb = realized_demand[tp][i]
                demand[tp,i].ub = realized_demand[tp][i]
        '''
        Solve updated problem and update states and recourse 
        '''
        m.optimize()
        if m.status != 2:
            m.computeIIS()
            m.write("model%i.ilp" %t)
        assert m.status == GRB.OPTIMAL, 'Model not optimal %i' %(t)
                
        for i in realized_demand[t-1]:  #All demand nodes
            for tp in range(prev_t, t): #[prev_t, ... , t-1]
                try:
                    assert I[tp,i].x>=0 and  z[tp,i].x>=0, '%s %i %f %f' %(i , tp , I[tp,i].x , z[tp,i].x)
                except:
                    print('%s %i %f %f' %(i , tp , I[tp,i].x , z[tp,i].x))
                    print([z[ttt,i].x for ttt in [1,2,3,4,5,6,7]])
                    print([I[ttt,i].x for ttt in [0,1,2,3,4,5,6,7]])
                    print('end')
                z[tp,i].lb = z[tp,i].x
                z[tp,i].ub = z[tp,i].x
                I[tp,i].lb = I[tp,i].x
                I[tp,i].ub = I[tp,i].x
                
    else:     
        m.update()
        m.optimize()
    


def safe_solution(hurricane_name, sol_timestamp, model):
    var_vals = {}
    for v in model.getVars():
        var_vals[v.varname] = v.X
    file_name  ='../data/BenchmarkSolutions/%s_%s' %(hurricane_name,sol_timestamp)
    pickle.dump(var_vals,open(file_name,'wb'),protocol=pickle.HIGHEST_PROTOCOL)
    
    
    
def parse_demands(path_to_forecast):
    data  = pickle.load(open(path_to_forecast, 'rb'))
    forecast  = data['predictedDemand']
    realized  = data['realDemand']
    
    real_stamps = list(realized.keys())
    real_stamps.sort()
    t_step = int((real_stamps[1] - real_stamps[0]).seconds/3600)
    num_intervals = np.minimum(len(real_stamps),150)
    T_set = [t for t in range(1,num_intervals+1)]
    T_max = num_intervals
    time_map = {rts:t+1 for (t,rts) in enumerate(real_stamps)}
    
    forecast_stamps = list(forecast.keys())
    forecast_stamps.sort()
    T_roll = [time_map[fts] for fts in forecast_stamps if time_map[fts]<=T_max]
    assert 1 in T_roll, 'First time period needs to be in the rolling horizon'
    demand_forecast = {}
    for fts in forecast_stamps:
        t = time_map[fts]
        demand_forecast[t] = {}
        for fts_pred in forecast[fts]:
            if fts_pred in time_map:
                tf = time_map[fts_pred]
                demand_forecast[t][tf] = {}
                for i in forecast[fts][fts_pred]:
                    demand_forecast[t][tf][i] = forecast[fts][fts_pred][i]
            else:
                pass#print('at ', fts, '   ' , fts_pred, 'not in map')
                    
          
        
    #demand_forecast = {t:{time_map[tsf]:{i:forecast[forecast_stamps[t-1]][tsf][i] for i in forecast[forecast_stamps[t-1]][tsf]}  for tsf in forecast[forecast_stamps[t-1]]} for t in T_roll}
    demand_realized = {t:{i:realized[real_stamps[t-1]][i] for i in realized[real_stamps[t-1]]}  for t in T_set}
    print('Forecast and realization parsed')
    return t_step, T_max, T_set, T_roll, time_map, real_stamps, forecast_stamps, demand_forecast, demand_realized
    
def simulate_rolling_horizon(T_roll, T_set, model, x, y, demand,  shortage, I, T_max, demand_forecast, demand_realized, demand_nodes):
    for (i_rol,t) in enumerate(T_roll):
        print('Solving t=',t)
        prev_t = T_roll[i_rol-1] if i_rol>0 else -1
        solve_time_period(model, x, y, demand,  shortage, I, t, prev_t,  T_max, demand_forecast_t=demand_forecast[t], realized_demand=demand_realized)
        if model.status != 2:
            model.computeIIS()
            model.write("model%i.ilp" %t)
        if t<=-T_max:
            cc = 'Martin County'
            enviado   = np.round(x.sum('*', '*', cc , t).getValue(),3)
            Inv_t = np.round(model.getVarByName('I[%i,%s]' %(t,cc)).X,3)
            Inv_t0 = np.round(model.getVarByName('I[%i,%s]' %(t-1,cc)).X,3)
            print(t, Inv_t, Inv_t0,   demand_realized[t][cc] ,enviado*230, shortage[t,cc].X)
            if t>1:
                print('I: ' , [ np.round(model.getVarByName('I[%i,%s]' %(tt,cc)).X,3) for tt in range(1,t)])
                print('z: ', [ np.round(model.getVarByName('z[%i,%s]' %(tt,cc)).X,3) for tt in range(1,t)])
                #print('\t',t, Inv_t, Inv_t0, demand_forecast[t][cc],  demand_realized[cc] ,enviado*230, shortage[t-1,cc].X)
    
    shortage = {t:{dn:shortage[t,dn].x for dn in demand_nodes} for t in T_set}
    return shortage

if __name__ == "__main__":
    #===========================================================================
    # import imageio
    # file_name_gif =  '../output/RH_IRMA.gif'
    # complete_file_name = os.path.realpath(file_name_gif)
    # with imageio.get_writer('../output/RH_IRMA.gif', mode='I') as writer:
    #     for t in range(1,128):
    #         image = imageio.imread('../output/RH_IRMA_t%i.png' %(t))
    #         writer.append_data(image)    
    #===========================================================================
    
    np.random.seed(0)
    
    t_step, T_max, T_set, T_roll, time_map, real_stamps, forecast_stamps, demand_forecast, demand_realized = parse_demands('../data/demandData_static.p')
    T_roll.append(T_max+1)
    
    T_roll4 = [1, T_max+1]
    for ti in range(200):
        t_r = T_roll4[-2]+4
        if t_r <= T_max:
            T_roll4.insert(-1, t_r)
            
    T_roll8 = [1, T_max+1]
    for ti in range(200):
        t_r = T_roll8[-2]+7
        if t_r <= T_max:
            T_roll8.insert(-1, t_r)
    
  
    
    #Load network data
    fl_df, fl_edges, demand_nodes, supply_nodes, net_nodes, Tau_max, tau_arcs, trucks, truck_cap = load_florida_network(t_step)
    
    #Add nominal demand to the outage forcast
    demand_forecast ={t:{tf:{i:demand_forecast[t][tf][i]+ fl_df['demand'][i] for i in demand_nodes} for tf in demand_forecast[t]} for t in range(1,T_max+1)}
    demand_realized = {t:{i:demand_realized[t][i] + fl_df['demand'][i]for i in demand_realized[t]} for t in T_set}
    demand_forecast[T_max+1] = {}
    print('Nominal demand added')
    
    model, x, y, demand, shortage, I = create_full_model(T_max, fl_df, fl_edges, demand_nodes, supply_nodes, net_nodes, Tau_max, tau_arcs, trucks, truck_cap )
    print('Main model built')
    shortage_roll_1 = simulate_rolling_horizon(T_roll, T_set, model, x, y, demand,  shortage, I,  T_max, demand_forecast, demand_realized, demand_nodes)
     
    print('Reseting model')
    model, x, y, demand, shortage, I = create_full_model(T_max, fl_df, fl_edges, demand_nodes, supply_nodes, net_nodes, Tau_max, tau_arcs, trucks, truck_cap )
    shortage_roll_4 = simulate_rolling_horizon(T_roll4, T_set, model, x, y, demand,  shortage, I,  T_max, demand_forecast, demand_realized, demand_nodes)
      
    print('Reseting model')
    model, x, y, demand, shortage, I = create_full_model(T_max, fl_df, fl_edges, demand_nodes, supply_nodes, net_nodes, Tau_max, tau_arcs, trucks, truck_cap )
    shortage_roll_8 = simulate_rolling_horizon(T_roll8, T_set, model, x, y, demand,  shortage, I,  T_max, demand_forecast, demand_realized, demand_nodes)
    
    
     
    track  = pickle.load(open('../data/trackdata.p', 'rb'))
    forecast  = pickle.load(open('../data/forecastdata.p', 'rb'))
    plot_type = 'abs' #abs
    print(T_max, plot_type)
    #Plot in any mode 
    max_radius = 0.2
    min_radius = 0.01
    delta_radius =  max_radius- min_radius
    max_demand =  np.max([demand_realized[t][dn] for t in T_set for dn in demand_nodes])
    min_demand =  np.min([demand_realized[t][dn] for t in T_set for dn in demand_nodes])
    delta_demand = max_demand - min_demand
    if plot_type == 'rel':
        max_demand =  {dn:np.max([demand_realized[t][dn] for t in T_set])for dn in demand_nodes}
        min_demand =  {dn:np.min([demand_realized[t][dn] for t in T_set])for dn in demand_nodes}
        delta_demand = {dn:(max_demand[dn] - min_demand[dn]) for dn in demand_nodes}
    
    for t in []:# T_set:
        print('Ploting stage %i' %(t))
        #unmet_dem = [models[t].getVarByName('z[%s]' %(dn)).X for dn in demand_nodes]
        time_stamp_t = real_stamps[t]
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
            d_i = demand_realized[t][dn]
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
            if shortage[t,dn].x  > 0:
                r_s= min_radius + (r-min_radius)*np.sqrt((shortage[t,dn].x)/d_i)
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
             
         
         
        #=======================================================================
        # filename = '../output/RH_IRMA_t%i.pdf' %(t)
        # complete_file_name = os.path.realpath(filename)
        # pp = PdfPages(complete_file_name)
        # pp.savefig(fig)
        # pp.close()
        #=======================================================================
        
        filename = '../output/RH_IRMA_t%i.png' %(t)
        complete_file_name = os.path.realpath(filename)
        fig.savefig(complete_file_name)
    
    
    shortage_profile1 = []
    shortage_profile4 = []
    shortage_profile8 = []
    demand_profile = []
    kwh_factor = 42*14.1/1000000
    for t in T_set:
        z1_t = kwh_factor*sum(shortage_roll_1[t][dn] for dn in demand_nodes)
        z4_t = kwh_factor*sum(shortage_roll_4[t][dn] for dn in demand_nodes)
        z8_t = kwh_factor*sum(shortage_roll_8[t][dn] for dn in demand_nodes)
        d_t = kwh_factor*sum(demand_realized[t][dn] for dn in demand_nodes)
        if t == 1 :
            shortage_profile1.append(z1_t)
            shortage_profile4.append(z4_t)
            shortage_profile8.append(z8_t)
            demand_profile.append(d_t)
        else:
            shortage_profile1.append(shortage_profile1[-1] + z1_t)
            shortage_profile4.append(shortage_profile4[-1] + z4_t)
            shortage_profile8.append(shortage_profile8[-1] + z8_t)
            demand_profile.append(demand_profile[-1]+d_t)
    
    fig,ax = plt.subplots(figsize=(8, 4), dpi=300) 
    import matplotlib.dates as mdates
    days = mdates.DayLocator()  
    
    save_obj = (shortage_profile1,shortage_profile4,shortage_profile8, demand_profile)
    with open('../data/irma_policy_RH.p', 'wb') as fp:
        pickle.dump(save_obj, fp, protocol=pickle.HIGHEST_PROTOCOL)
    
    
    with open('../data/irma_policy_RH.p', 'rb') as fp:
        load_obj = pickle.load(fp)
    shortage_profile1 = load_obj[0]
    shortage_profile4 = load_obj[1]
    shortage_profile8 = load_obj[2]
    demand_profile = load_obj[3]
    
    real_stamps = real_stamps[:len(demand_profile)]                      
    ax.plot(real_stamps, shortage_profile1, color='red', linestyle='--', dashes=(1,1) ,label='Shortage RH-3h')
    ax.plot(real_stamps, shortage_profile4, color='red', linestyle='--', dashes=(3,1), label='Shortage RH-12h')
    ax.plot(real_stamps, shortage_profile8, color='red', label='Shortage RH-24h')
   
    ax.set_ylabel('Shortage (MWh)', color='red', fontsize=10)
    ax.set_ylim(0,300)
    ax.set_yticks(np.linspace(0, ax.get_ybound()[1], 7))
    ax.set_xlabel('Date', color='black', fontsize=10)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(10) 
        # specify integer or one of preset strings, e.g.
        #tick.label.set_fontsize('x-small') 
        tick.label.set_rotation('45')
    axR = ax.twinx()
    axR.plot(real_stamps, demand_profile, color='blue', label='Cumulative demand')
    axR.set_ylabel('Demand (MWh)', color='blue', fontsize=10)
    axR.set_ylim(0,2400)
    axR.set_yticks(np.linspace(0, axR.get_ybound()[1], 7))
    ax.legend(loc='best', shadow=True, fontsize='small')
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_minor_locator(mdates.HourLocator() )
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.grid(True)
    #ax.xaxis.set_major_formatter( mdates.AutoDateFormatter())
    
    
    filename = '../output/RH_IRMA_Shortage.pdf'
    complete_file_name = os.path.realpath(filename)
    fig.savefig(complete_file_name)
    plt.show()
    #===========================================================================
    # for i in demand_nodes:
    #     print('z[%s]: ' %(i), [ np.round(model.getVarByName('z[%i,%s]' %(tt,i)).X,0) for tt in T_set])
    # print(shortage.sum().getValue())
    #===========================================================================
    
    #safe_solution(hurricane, adv_date, model)
    
    
    
    
    
    