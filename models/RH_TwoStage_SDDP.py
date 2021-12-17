'''
Created on Jan 15, 2019

@author: dduque

Implements a rolling horizon scheme in which every decision period a two stage model is
solved by SDDP.

Modified on Oct 6, 2019
'''

import os
import sys
from os import path
import time
from gurobipy import GRB
sys.path.append(path.abspath('/Users/dduque/MacWorkspace/FuelDistModel'))
sys.path.append(path.abspath('/home/dduque/dduque_projects/FuelDistModel'))  # Crunch
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from models.RollingHorizon import rolling_horizon
from models.MultistageModels import DFSC_TwoStage, DFSC_TwoStage_random_builder,\
    DFSC_TwoStage_extensive

from IO.NetworkBuilder import load_florida_network

from models import project_path, models_path
#Import SDDP library
sys.path.append(path.abspath('/Users/dduque/Dropbox/WORKSPACE/SDDP'))
sys.path.append(path.abspath('/home/dduque/dduque_projects/SDDP'))  # Crunch
import CutSharing
from CutSharing.SDDP_Alg import SDDP as SDDP
'''
Model Constants
'''
DELTA_HORIZON = None  # Hours of look ahead (time horizon of the two-stage model)
DELTA_T_MODEL = 1  # Temporal model resolution in hours
DELTA_T_STAGE = None  # Temporal resolution of the first stage
DELTA_T_SECOND_STAGE = None  # number of DELTA_T_MODEL periods in the last stage
DELTA_ROLLING = None  # Periods to roll forward in the rolling horizon
DELTA_NOTIFICATION = None


def random_builder_factory(t_now, T_max, scenarios, DFSC_instance):
    '''
        Build the random build function for SDDP
        Args:
            t_now (int): current time at which the RH is going to be solved
            T_max (int): last time in the global planning horizon
            scenarios (list of dict): A list with scenarios organized as dictionaries
            DFSC_instance (tuple): instance of the problem
        Returns:
            A function that build the randomness in the model
    '''
    T_max_RH = np.minimum(T_max, t_now + DELTA_T_STAGE + DELTA_T_SECOND_STAGE)
    
    def DFSC_random_builder():
        intra_ints = [
            range(t_now, t_now + DELTA_T_STAGE, DELTA_T_MODEL),
            range(t_now + DELTA_T_STAGE, T_max_RH, DELTA_T_MODEL)
        ]
        return DFSC_TwoStage_random_builder(T=2,
                                            intra_time_periods=intra_ints,
                                            sce_forecast=scenarios,
                                            DFSC_instance=DFSC_instance)
    
    return DFSC_random_builder


def model_builder_factory(t_now, T_max, DFSC_instance):
    '''
        Builds the model builder function that depends on
        the stage (first or second stage)
        Args:
            t_now (int): current time at which the RH is going to be solved
            T_max (int): last time in the global planning horizon
            DFSC_instance (tuple): instance of the problem
        Returns:
            A function that build the stage problem
    '''
    
    T_max_RH = np.minimum(T_max, t_now + DELTA_T_STAGE + DELTA_T_SECOND_STAGE)
    
    def DFSC_two_stage_model_builder(stage):
        if stage == 0:  # First stage lasts DELTA_T_STAGE periods
            return DFSC_TwoStage(t_now, t_now + DELTA_T_STAGE, DELTA_T_MODEL, DFSC_instance)
        elif stage == 1:
            return DFSC_TwoStage(t_now + DELTA_T_STAGE, T_max_RH, DELTA_T_MODEL, DFSC_instance)
        else:
            raise "Time period out of bound for a two-stage model"
    
    return DFSC_two_stage_model_builder


def two_stage_opt(t, T, instance_data, scenarios, prev_stage_state):
    '''
        Two-stage optimization for a given moment t
        Args:
            t (int): Current time in the RH.
            T (int): Last time period.
            scenarios (list of dict): List of scenarios organized as dictionaries.
            prev_stage_state (object): A container with information of the previews.
                It has the sintex of the first output of this function.
        Returns:
            out_states_val (object): Output information to fit the next model in the RH scheme
            performance_out (list): list of performance metrics (e.g., shortfall)
            perfromance_base (list): list of performance metrics baseline (e.g., demand)
    '''
    
    DFSC_instance = instance_data
    two_stage_rnd_builder = random_builder_factory(t, T, scenarios, DFSC_instance)
    two_stage_model_builder = model_builder_factory(t, T, DFSC_instance)
    CutSharing.options['multicut'] = False
    alg = SDDP(2, two_stage_model_builder, two_stage_rnd_builder, lower_bound=0)
    '''
    Modify first stage initial state
    '''
    model_t = alg.stage_problems[0]
    for out_state in prev_stage_state:
        var = model_t.states_map[out_state]
        var.lb = prev_stage_state[out_state]
        var.ub = prev_stage_state[out_state]
    CutSharing.options['max_time'] = 30 * 60
    CutSharing.options['lines_freq'] = 10
    CutSharing.options['in_sample_ub'] = 1
    CutSharing.options['max_iter'] = 1000
    CutSharing.options['expected_value_problem'] = True
    alg.run(instance_name='DieselFuel-Florida - RH_TwoStage - (t,T)=(%i,%i)' % (t, T))
    CutSharing.options['max_time'] = CutSharing.options['max_time'] - alg.get_wall_time()
    CutSharing.options['in_sample_ub'] = 20
    CutSharing.options['max_iter'] = 1000
    CutSharing.options['expected_value_problem'] = False
    alg.run(instance_name='DieselFuel-Florida - RH_TwoStage - (t,T)=(%i,%i)' % (t, T))
    '''
    Realization update and states re-computation
    Need to look for realizations in past forecast
    with respect to: t + DELTA_T_STAGE.  
    '''
    demand_nodes = DFSC_instance[2]
    for intra_t in range(t, t + DELTA_T_STAGE, DELTA_T_MODEL):
        if intra_t in scenarios[0]:
            for c in demand_nodes:
                v = model_t.rhs_vars_var['demand[%i,%s]' % (intra_t, c)]
                v.lb = scenarios[0][intra_t][intra_t][c]
                v.ub = scenarios[0][intra_t][intra_t][c]
    
    for v in model_t.model.getVars():
        if not ('zS' in v.VarName or 'zN' in v.VarName or 'I[' in v.VarName):
            v.lb = v.X
            v.ub = v.X
    
    model_t.model.update()
    model_t.model.optimize()
    
    out_states_val = {v_name: model_t.out_state_var[v_name].X for v_name in model_t.out_state_var}
    
    performance_out = [[], []]
    perfromance_base = [[], []]
    pefromance_var_name = ['zS', 'zN']
    demand_nodes = DFSC_instance[2]
    nominal_demand = DFSC_instance[-1]
    for intra_t in range(t, t + DELTA_T_STAGE, DELTA_T_MODEL):
        shortage_intra_t = sum(
            model_t.model.getVarByName('%s[%i,%s]' % (pefromance_var_name[0], intra_t, c)).X for c in demand_nodes)
        performance_out[0].append(shortage_intra_t)
        demand_t = sum(scenarios[0][intra_t][intra_t][c] for c in demand_nodes) if (intra_t in scenarios[0]) else 0
        perfromance_base[0].append(demand_t)
        
        shortage_intra_t = sum(
            model_t.model.getVarByName('%s[%i,%s]' % (pefromance_var_name[1], intra_t, c)).X for c in demand_nodes)
        performance_out[1].append(shortage_intra_t)
        demand_t = sum(nominal_demand[intra_t, c] for c in demand_nodes)
        perfromance_base[1].append(demand_t)
    
    return (out_states_val, performance_out, perfromance_base)


def two_stage_opt_extensive(t, T, instance_data, scenarios, prev_stage_state):
    '''
        Two-stage optimization for a given moment t using the 
        extensive formulation. 
        Args:
            t (int): Current time in the RH.
            T (int): Last time period.
            scenarios (list of dict): List of scenarios organized as dictionaries.
            prev_stage_state (object): A container with information of the previous
                stage. It has the sintax of the first output of this function.
        Returns:
            out_states_val (object): Output information to fit the next model in the RH scheme
            performance_out (list): list of performance metrics (e.g., shortfall)
            perfromance_base (list): list of performance metrics baseline (e.g., demand)
    '''
    tnow = time.time()
    # Build model for the extensive formulation
    DFSC_instance = instance_data
    m, in_state, out_state, rhs_vars, out_in_map = DFSC_TwoStage_extensive(
        t, np.minimum(t + DELTA_T_STAGE, T), np.minimum(t + DELTA_T_STAGE + DELTA_T_SECOND_STAGE, T), DELTA_T_MODEL,
        DELTA_ROLLING, DELTA_NOTIFICATION, DFSC_instance, scenarios)
    m.params.OutputFlag = 0
    # m.params.Method = 3
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
    
    m.update()
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
    perfromance_base = [[], []]
    pefromance_var_name = ['zS', 'zN']
    demand_nodes = DFSC_instance[2]
    nominal_demand = DFSC_instance[-1]
    for intra_t in range(t, np.minimum(T, t + DELTA_ROLLING), DELTA_T_MODEL):
        shortage_intra_t = sum(
            m.getVarByName('%s[%i,%s]' % (pefromance_var_name[0], intra_t, c)).X for c in demand_nodes)
        performance_out[0].append(shortage_intra_t)
        demand_t = sum(scenarios[0][intra_t][intra_t][c] for c in demand_nodes) if (intra_t in scenarios[0]) else 0
        perfromance_base[0].append(demand_t)
        
        shortage_intra_t = sum(
            m.getVarByName('%s[%i,%s]' % (pefromance_var_name[1], intra_t, c)).X for c in demand_nodes)
        performance_out[1].append(shortage_intra_t)
        demand_t = sum(nominal_demand[intra_t, c] for c in demand_nodes)
        perfromance_base[1].append(demand_t)
    t_model_out = time.time() - tnow - t_model_build - t_model_solve
    t_total = time.time() - tnow
    print('build=%8.2f solve=%8.2f out=%8.2f total=%8.2f' % (t_model_build, t_model_solve, t_model_out, t_total))
    print('roll short Surge  ', np.sum(performance_out[0]))
    print('roll short Nominal', np.sum(performance_out[1]))
    print('============================================================\n')
    
    return (out_states_val, performance_out, perfromance_base)


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


def setup_data(forecast):
    global DELTA_HORIZON
    global DELTA_ROLLING
    global DELTA_T_STAGE
    global DELTA_T_SECOND_STAGE
    assert DELTA_T_STAGE % DELTA_T_MODEL == 0, "Time resolution of the model should fit in the stage"
    assert DELTA_T_SECOND_STAGE % DELTA_T_MODEL == 0, "Time resolution of the model should fit in the stage"
    
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
    while tr < T_set[-1] and tr + DELTA_ROLLING <= max_roll:
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


def run_all_sequential():
    global DELTA_HORIZON
    global DELTA_T_STAGE
    global DELTA_T_SECOND_STAGE
    
    fullfilment_rates = [10, 20]  #[10,20,30]
    for FR in fullfilment_rates:
        '''
        =================================================================
        Reading forecast data
        =================================================================
        '''
        data_path = os.path.expanduser("~/Dropbox/WORKSPACE/FuelDistModel/data/")
        path_to_forecast = data_path + 'predDemand/predDemand_%i.p' % (FR)
        ''' Contains a dictionary with each replication of the enamble
            data[ensamble_number][issue_time][prediction_time] = array of predictions per county '''
        irma_data = pickle.load(open(path_to_forecast, 'rb'))
        '''
        =================================================================
        Set up experiments
        =================================================================
        '''
        Look_Aheads = [24 * i for i in [1, 2, 2, 3, 4, 5, 6]]
        Operation_fex = [6, 12, 24, 48, 72]
        for la in Look_Aheads:
            for first_stage_delta in Operation_fex:
                if la > first_stage_delta:
                    DELTA_HORIZON = la
                    DELTA_T_STAGE = first_stage_delta
                    DELTA_T_SECOND_STAGE = DELTA_HORIZON - DELTA_T_STAGE
                    T_set, T_roll, T_labels, scenarios, _ = setup_data(irma_data)
                    instance_data = load_florida_network(DELTA_T_MODEL, 0, T_set[-1], partition_network=False, zone=-1)
                    instance_name = 'RH_TwoStage_IRMA_Shortage_FR%i_H%i_FS%i_SS%i' % (FR, DELTA_HORIZON, DELTA_T_STAGE,
                                                                                      DELTA_T_SECOND_STAGE)
                    print(T_roll, len(scenarios), '\n', instance_name)
                    shortage_profiles, demand_profiles = rolling_horizon(T_set, T_roll, T_labels, instance_data,
                                                                         scenarios, two_stage_opt,
                                                                         two_stage_results_processor)
                    save_obj = (shortage_profiles, demand_profiles, T_set, T_labels)
                    with open('%s/output/%s.p' % (project_path, instance_name), 'wb') as fp:
                        pickle.dump(save_obj, fp, protocol=pickle.HIGHEST_PROTOCOL)


def run_rh_two_stage(R, H, F):
    global DELTA_HORIZON
    global DELTA_T_STAGE
    global DELTA_T_SECOND_STAGE
    '''
    =================================================================
    Reading forecast data
    =================================================================
    '''
    FR = R  #Fullfilment rate
    data_path = project_path + "/data/"
    path_to_forecast = data_path + 'predDemand/predDemand_%i.p' % (FR)
    ''' Contains a dictionary with each replication of the enamble
        data[ensamble_number][issue_time][prediction_time] = array of predictions per county '''
    irma_data = pickle.load(open(path_to_forecast, 'rb'))
    '''
    =================================================================
    Set up experiment
    =================================================================
    '''
    if H > F:
        DELTA_HORIZON = H
        DELTA_T_STAGE = F
        DELTA_T_SECOND_STAGE = DELTA_HORIZON - DELTA_T_STAGE
        T_set, T_roll, T_labels, scenarios, _ = setup_data(irma_data)
        instance_data = load_florida_network(DELTA_T_MODEL, 0, T_set[-1], partition_network=False, zone=-1)
        instance_name = 'RH_TwoStage_IRMA_Shortage_FR%i_H%i_FS%i_SS%i' % (FR, DELTA_HORIZON, DELTA_T_STAGE,
                                                                          DELTA_T_SECOND_STAGE)
        print(T_roll, len(scenarios), '\n', instance_name)
        shortage_profiles, demand_profiles = rolling_horizon(T_set, T_roll, T_labels, instance_data, scenarios,
                                                             two_stage_opt, two_stage_results_processor)
        save_obj = (shortage_profiles, demand_profiles, T_set, T_labels)
        with open('%s/output/%s.p' % (project_path, instance_name), 'wb') as fp:
            pickle.dump(save_obj, fp, protocol=pickle.HIGHEST_PROTOCOL)


def launch_rh_ts_ext():
    '''
    Launches all the experiments for rolling horizon
    two stage extensive formulation.
    '''
    FR = 20
    H_set = [int(24 * i) for i in [0.5, 1, 2, 3, 4, 5, 6]]  #0.5, 1,2,
    F_set = [6, 12, 24, 36, 48, 72]  #
    R_set = [24, 48]
    for H in H_set:
        for F in F_set:
            if F < H:
                print(FR, F, H, F)
                #run_rh_two_stage_extensive(FR,F,H,F)
                for R in R_set:
                    if R < F:
                        print(FR, R, H, F)
                        #run_rh_two_stage_extensive(FR,R,H,F)


def run_rh_two_stage_extensive(FR=50, H=48, F=24, R=12, N=6):
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
    if isinstance(FR, int):
        path_to_forecast = data_path + 'predDemand/predDemand_%i.p' % (FR)
    elif FR == 'perfect_information':
        path_to_forecast = data_path + 'realDemand.p'
    else:
        path_to_forecast = data_path + 'predDemand/%s.p' % (FR)
    ''' Contains a dictionary with each replication of the enamble
        data[ensamble_number][issue_time][prediction_time] = array of predictions per county '''
    irma_data = pickle.load(open(path_to_forecast, 'rb'))
    if FR == 'perfect_information':
        irma_data = transform_to_forecast(irma_data)
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
        DELTA_NOTIFICATION = N
        T_set, T_roll, T_labels, scenarios, _ = setup_data(irma_data)
        instance_data = load_florida_network(DELTA_T_MODEL, 0, T_set[-1], partition_network=False, zone=-1)
        instance_name = 'RH_TwoStageExtensive_IRMA_Shortage_FR%s_H%i_F%i_R%i_N%i' % (str(FR), H, F, R, N)
        print(instance_name, '\n', T_roll, len(scenarios))
        shortage_profiles, demand_profiles = rolling_horizon(T_set, T_roll, T_labels, instance_data, scenarios,
                                                             two_stage_opt_extensive, two_stage_results_processor)
        save_obj = (shortage_profiles, demand_profiles, T_set, T_labels)
        with open('%s/output/%s.p' % (project_path, instance_name), 'wb') as fp:
            pickle.dump(save_obj, fp, protocol=pickle.HIGHEST_PROTOCOL)
        
        #m.computeIIS()
        #m.write("extensive_model.ilp")
        #shortage_profiles, demand_profiles = rolling_horizon(T_set, T_roll, T_labels, instance_data, scenarios, two_stage_opt, two_stage_results_processor)
        #save_obj = (shortage_profiles, demand_profiles,T_set, T_labels)
        #with open('%s/output/%s.p' %(project_path, instance_name), 'wb') as fp:
        #    pickle.dump(save_obj, fp, protocol=pickle.HIGHEST_PROTOCOL)


def decorate_plot(ax, y_title, y_lim, y_ticks=11, f_size=10, leyend=False):
    ax.set_ylim(0, y_lim)
    ax.set_yticks(np.linspace(0, ax.get_ybound()[1], y_ticks))
    ax.set_xlabel('Date', color='black', fontsize=f_size)
    ax.set_ylabel(y_title, color='black', fontsize=f_size)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(f_size)
        # specify integer or one of preset strings, e.g.
        #tick.label.set_fontsize('x-small')
        tick.label.set_rotation('45')
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_minor_locator(mdates.HourLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.grid(True)
    ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    if leyend:
        ax.legend(loc=2, shadow=False, fontsize='x-small')


def plot():
    np.random.seed(19867)
    '''========================================================
                            READ DATA
       ========================================================'''
    results = {}
    
    H_set = [int(24 * i) for i in [1, 2, 3, 4, 5, 6]]  #0.5, 1,2,
    F_set = [6, 12, 24, 36, 48, 60, 72, 96, 120]  #
    FR_set = [50, 'RDSurge_50', 'ndfdDemand_50']
    N_set = [0, 6, 12, 24]
    for FR in FR_set:
        for H in H_set:
            for F in F_set:
                for R in F_set:
                    for N in N_set:
                        #plotted as H96F48plots: FR in [50, 'avgDemand_50','RDSurge_50', 'ndfdDemand_50'] and H==96 and R==48 and F==48  and N in [24]
                        
                        if H in [72] and F in [24] and R in [24] and N in [24]:  #plotted as PI plot
                            #    print('RH_TwoStageExtensive_IRMA_Shortage_FR%s_H%i_F%i_R%i_N%i' %(str(FR), H, F, R, N))
                            #if FR == 'RDSurge_50' and  H in [48,96,144] and F in [int(H/2)] and R in [24]  and N in [0,24]:# and H in [48] and F == 24 and R==24 and N in [0,6,12,24]:
                            try:
                                instance_name = 'RH_TwoStageExtensive_IRMA_Shortage_FR%s_H%i_F%i_R%i_N%i' % (str(FR), H,
                                                                                                             F, R, N)
                                load_obj = None
                                with open('%s/output/%s.p' % (project_path, instance_name), 'rb') as fp:
                                    load_obj = pickle.load(fp)
                                prob_type = FR
                                if (FR == 'RDSurge_50'):
                                    prob_type = 'PI  '
                                elif FR == 'avgDemand_50':
                                    prob_type = 'Avg.'
                                elif FR == 'ndfdDemand_50':
                                    prob_type = 'NDFD'
                                else:
                                    print('Warning: problem type is ', FR)
                                
                                results[(prob_type, H, F, R, N)] = load_obj
                            except Exception as e:
                                print(e)
                            #===================================================
                            # try:
                            #     instance_name = 'RH_TwoStageExtensive_IRMA_Shortage_FR%s_R%i_H%i_FS%i_SS%i' %(str(FR), R, H, F, S)
                            #     load_obj = None
                            #     with open('%s/output/%s.p' %(project_path, instance_name), 'rb') as fp:
                            #         load_obj = pickle.load(fp)
                            #     results[(FR,H,R,F,S)] = load_obj
                            # except Exception as e:
                            #     print(e)
                            # try:
                            #
                            #     instance_name = 'RH_TwoStageExtensive_IRMA_Shortage_FR%s_R%i_H%i_FS%i_SS%i_ori' %(str(FR), R, H, F, S)
                            #     load_obj = None
                            #     with open('%s/output/%s.p' %(project_path, instance_name), 'rb') as fp:
                            #         load_obj = pickle.load(fp)
                            #     results[('50_ori',H,R,F,S)] = load_obj
                            # except Exception as e:
                            #     print(e)
                            # try:
                            #     instance_name = 'RH_TwoStageExtensive_IRMA_Shortage_FR%s_R%i_H%i_FS%i_SS%i_zeroFor' %(str(FR), R, H, F, S)
                            #     load_obj = None
                            #     with open('%s/output/%s.p' %(project_path, instance_name), 'rb') as fp:
                            #         load_obj = pickle.load(fp)
                            #     results[('50_zero',H,R,F,S)] = load_obj
                            # except Exception as e:
                            #     print(e)
                            #===================================================
    '''========================================================'''
    '''
    Plot results
    '''
    nominal_demand_plot = False
    surge_demand_plot = False
    total_demand_plot = False
    
    #Plot controls
    my_colors = np.random.rand(3, 50)
    my_cmap = plt.cm.get_cmap('hsv', len(results) + 1)
    y_lim_total = 3500000
    y_lim_surge = 1000000
    y_lim_nominal = 2500000
    y_lim_total_short = 1000000
    my_font_size = 8
    plot_id = 0
    fig1, ax1 = plt.subplots(figsize=(8, 4), dpi=300)
    axR1 = ax1.twinx()
    fig2, ax2 = plt.subplots(figsize=(8, 4), dpi=300)
    axR2 = ax2.twinx()
    fig3, ax3 = plt.subplots(figsize=(8, 4), dpi=300)
    axR3 = ax3.twinx()
    results_key = list(results.keys())
    
    def sort_fun(x):
        xk = str(x[0])
        if 'PI' in xk:
            return 0 + x[1] + 0.5 * x[2] + 0.05 * x[3]
        elif 'Avg' in xk:
            return 1000 + x[1] + 0.5 * x[2] + 0.05 * x[3]
        elif 'NDFD' in xk:
            return 10000 + x[1] + 0.5 * x[2] + 0.05 * x[3]
        else:
            return 100000 + x[1] + 0.5 * x[2] + 0.05 * x[3]
    
    results_key.sort(key=sort_fun)
    for (FR, H, F, R, N) in results_key:
        instance_name = 'RH_TwoStageExtensive_IRMA_Shortage_FR%s_H%i_F%i_R%i_N%i' % (str(FR), H, F, R, N)
        label_name = 'H%i_F%i_R%i_N%i' % (H, F, R, N) if type(FR) == type(1) else 'H%i_F%i_R%i_N%i-%s' % (H, F, R, N,
                                                                                                          FR)
        load_obj = results[FR, H, F, R, N]
        shortage_profiles, demand_profiles, T_set, T_labels = load_obj
        print(instance_name)
        print(T_labels[0], T_labels[-1], len(load_obj[0][0]), len(load_obj[1][0]))
        plot_color = my_cmap(plot_id)  #np.array([10*R-H, H, F+R+H]) #my_colors[:,plot_id]
        '''============SURGE DEMAND============'''
        shortage_profile = load_obj[0][0]
        demand_profile = load_obj[1][0]
        
        kwh_factor = 1  #42*14.1/1000000
        cummulative_surge_shortage = []
        cummulative_surge_demand = []
        for (i, s) in enumerate(shortage_profile):
            if i > 0:
                cummulative_surge_shortage.append(s * kwh_factor + cummulative_surge_shortage[-1])
                cummulative_surge_demand.append(demand_profile[i] * kwh_factor + cummulative_surge_demand[-1])
            else:
                cummulative_surge_shortage.append(s * kwh_factor)
                cummulative_surge_demand.append(demand_profile[i] * kwh_factor)
        cummulative_surge_shortage = np.array(cummulative_surge_shortage)
        cummulative_surge_demand = np.array(cummulative_surge_demand)
        min_lenth = np.min([380, len(cummulative_surge_shortage), len(cummulative_surge_demand)])
        cummulative_surge_shortage = cummulative_surge_shortage[:min_lenth]
        cummulative_surge_demand = cummulative_surge_demand[:min_lenth]
        
        real_stamps = T_labels[:min_lenth]
        
        ax1.plot(real_stamps,
                 cummulative_surge_shortage,
                 color=plot_color,
                 linestyle='--',
                 dashes=(1, 1),
                 label=label_name)
        decorate_plot(ax1, y_title='Surge shortage (Barrels)', y_lim=200000, f_size=my_font_size, leyend=True)
        axR1.plot(real_stamps, cummulative_surge_demand, color='black', label='Cumulative demand')
        decorate_plot(axR1, y_title='Surge demand (Barrels)', y_lim=y_lim_surge, f_size=my_font_size)
        fig1.tight_layout()
        
        if surge_demand_plot == False:  #Just surge demand plot
            surge_demand_plot = True
            fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
            ax.plot(real_stamps, cummulative_surge_demand, color='black', label='Cumulative demand')
            decorate_plot(ax, y_title='Surge demand (Barrels)', y_lim=y_lim_total, y_ticks=11, f_size=my_font_size)
            fig.tight_layout()
            filename = '%s/output/RH_TwoStage_IRMA_SurgeDemand_FR%s.pdf' % (project_path, str(FR))
            fig.savefig(filename)
        '''============NOMINAL DEMAND============'''
        shortage_profile = load_obj[0][1]
        demand_profile = load_obj[1][1]
        kwh_factor = 1
        cummulative_nominal_shortage = []
        cummulative_nominal_demand = []
        for (i, s) in enumerate(shortage_profile):
            if i > 0:
                cummulative_nominal_shortage.append(s * kwh_factor + cummulative_nominal_shortage[-1])
                cummulative_nominal_demand.append(demand_profile[i] * kwh_factor + cummulative_nominal_demand[-1])
            else:
                cummulative_nominal_shortage.append(s * kwh_factor)
                cummulative_nominal_demand.append(demand_profile[i] * kwh_factor)
        cummulative_nominal_shortage = np.array(cummulative_nominal_shortage)
        cummulative_nominal_demand = np.array(cummulative_nominal_demand)
        #min_lenth = np.minimum(len(cummulative_short), len(cummulative_demand))
        cummulative_nominal_shortage = cummulative_nominal_shortage[:min_lenth]
        cummulative_nominal_demand = cummulative_nominal_demand[:min_lenth]
        
        real_stamps = T_labels[:min_lenth]
        ax2.plot(real_stamps,
                 cummulative_nominal_shortage,
                 color=plot_color,
                 linestyle='--',
                 dashes=(1, 1),
                 label=label_name)
        decorate_plot(ax2, 'Nominal shortage (Barrels)', y_lim=1000000, f_size=my_font_size, leyend=True)
        axR2.plot(real_stamps, cummulative_nominal_demand, color='black', label='Cumulative demand')
        decorate_plot(axR2, y_title='Nominal demand (Barrels)', y_lim=y_lim_nominal, f_size=my_font_size)
        fig2.tight_layout()
        
        total_demand = cummulative_surge_demand + cummulative_nominal_demand
        if nominal_demand_plot == False:  #Just nominal demand plot
            nominal_demand_plot = True
            fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
            ax.plot(real_stamps, cummulative_nominal_demand, color='black', label='Cumulative demand')
            decorate_plot(ax, y_title='Nominal demand (Barrels)', y_lim=y_lim_total, y_ticks=11, f_size=my_font_size)
            fig.tight_layout()
            filename = '%s/output/RH_TwoStage_IRMA_NominalDemand_FR%s.pdf' % (project_path, str(FR))
            fig.savefig(filename)
            #Total demand plot
            fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
            ax.plot(real_stamps, total_demand, color='black', label='Cumulative demand')
            decorate_plot(ax, y_title='Total demand (Barrels)', y_lim=y_lim_total, y_ticks=11, f_size=my_font_size)
            fig.tight_layout()
            filename = '%s/output/RH_TwoStage_IRMA_TotalDemand_FR%s.pdf' % (project_path, str(FR))
            fig.savefig(filename)
        
        #shortage plot
        total_shortage = cummulative_surge_shortage + cummulative_nominal_shortage
        ax3.plot(real_stamps, total_shortage, color=plot_color, linestyle='--', dashes=(1, 1), label=label_name)
        decorate_plot(ax3,
                      'Total shortage (Barrels)',
                      y_lim=y_lim_total_short,
                      y_ticks=11,
                      f_size=my_font_size,
                      leyend=True)
        axR3.plot(real_stamps, total_demand, color='black', label='Cumulative demand')
        decorate_plot(axR3, 'Total demand (Barrels)', y_lim=y_lim_total, y_ticks=11, f_size=my_font_size)
        fig3.tight_layout()
        
        plot_id = plot_id + 1
        
        plot_name = str(FR) + '_' + str(H) + '_' + str(F) + '_' + str(R) + '_' + str(N)
        filename = '%s/output/RH_TwoStage_IRMA_SurgeShortage_FR%s_%s.pdf' % (project_path, str(FR), plot_name)
        fig1.savefig(filename)
        filename = '%s/output/RH_TwoStage_IRMA_NominalShortage_FR%s_%s.pdf' % (project_path, str(FR), plot_name)
        fig2.savefig(filename)
        filename = '%s/output/RH_TwoStage_IRMA_TotalShortage_FR%s_%s.pdf' % (project_path, str(FR), plot_name)
        fig3.savefig(filename)
    plt.show()


if __name__ == '__main__':
    from Utils.argv_parser import sys, parse_args
    argv = sys.argv
    _, kwargs = parse_args(argv[1:])
    FR, R, H, F, N = None, None, None, None, None
    if 'FR' in kwargs:
        FR = kwargs['FR']
    if 'H' in kwargs:
        H = kwargs['H']
    if 'F' in kwargs:
        F = kwargs['F']
    if 'R' in kwargs:
        R = kwargs['R']
    if 'N' in kwargs:
        N = kwargs['N']
    
    #run_rh_two_stage_extensive(FR=FR, H=H, F=F, R=R, N=N)
    #run_rh_two_stage_extensive(FR='perfect_information')
    
    #===========================================================================
    #(FR='RDSurge_50',H=96,F=72,R=72,N=6)
    # run_rh_two_stage_extensive(FR='RDSurge_50',H=96,F=72,R=72,N=12)
    # run_rh_two_stage_extensive(FR='avgDemand_50',H=24,F=12,R=12,N=12)
    #run_rh_two_stage_extensive(FR=50, H=24, F=12, R=12, N=12)
    # run_rh_two_stage_extensive(FR=50,H=24,F=12,R=12,N=0)
    # run_rh_two_stage_extensive(FR=50,H=24,F=12,R=12,N=6)
    # run_rh_two_stage_extensive(FR=50,H=24,F=12,R=12,N=24)
    #===========================================================================
    #run_rh_two_stage_extensive(FR=50,H=24,F=12,R=12,N=12)
    #scp dduque@crunch.osl.northwestern.edu:/home/dduque/dduque_projects/FuelDistModel/output/RH_TwoStageExtensive_IRMA_Shortage* output/
    plot()
