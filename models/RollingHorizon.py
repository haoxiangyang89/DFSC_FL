'''
Created on Jan 15, 2019

@author: dduque
'''
import numpy as np
import datetime

#import pdb
#pdb.set_trace()


def rolling_horizon(T_set, T_roll, T_labels, data, sample_path, optAlg, results_process):
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
    resultsOut = {}
    for (i_rol, t) in enumerate(T_roll):
        print('Solving t=', t, ' = ', T_labels[t])
        prev_t = T_roll[i_rol - 1] if i_rol > 0 else -1
        T_max = T_set[-1]
        alg_output = optAlg(t, T_max, data, sample_path, prev_roll_output)
        assert type(alg_output) == tuple and len(alg_output) == 5, 'optAlg function must return a 2-dimensional tuple'
        prev_roll_output = alg_output[0]
        performance[t] = alg_output[1]
        performance_base[t] = alg_output[2]
        cost_out[t] = alg_output[3]
        #resultsOut[t] = alg_output[4]
    return results_process(performance), results_process(performance_base), results_process(cost_out)
#, resultsOut