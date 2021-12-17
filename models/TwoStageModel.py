'''
Created on Mar 26, 2018

@author: dduque, hyang

This module implements the following two stage model

min cx  + E[h(x,xi)]
s.t. 
    l<=x<=u
    \sum_{i \in I}x_i <=b
    
with 
h(x,xi^w) = min fy + M e^T (s+ + s-)
        s.t
            sum_{j:(i,j)\in A}y_{i,j} - sum_{j:(j,i)\in A}y_{j,i} = \delta_i^w x_i^w - d_i  + s+_i - s-_i \forall i\in N
            0<=y<=u^w
Where:
I is the set of nodes where diesel can be prepositioned.
N is the set of all nodes.
d_i^w is the demand of diesel at node i under scenario w
\delta_i^w is the fraction that remains usable under scenario w 
u^w is the capacity of each arch under scenario w.
x_i is the amount of diesel to preposition at node i
y_{i,j} is the amount of flow from i to j
s+ and s- are slack variables so that the second stage has full recourse. 
'''
import pickle
from gurobipy import *
from math import radians
import numpy as np
from IO.NetworkBuilder import haversineEuclidean



class DSC_Benders():
    '''
    Basic benders implementation for a two stage stochastic linear program
    that models prepositioning decisions of good in a netowrk. The second stage
    distributes the goods to meet demand. 
    '''
    def __init__(self, data):
        self.data = data
        self.master = DSC_Master(data)
        self.sub = DSC_Sub(data)
        self.lb = -np.inf
        self.ub = np.inf
        self.x_bar = None
    
    def check_stop(self):
        '''
        
        '''
        if self.iteration >= 12:
            return True  
        if self.ub - self.lb <= np.minimum(np.abs(self.ub),np.abs(self.lb))*1E-5:
            return False
        
        return False    
    def run(self):
        '''
        Run benders on the DSC model
        '''
        
        self.master.build_model()
        self.sub.build_model()
        self.iteration = 0 
        while True:
            '''Solve Master'''
            status, lb, cx_hat, x_hat, theta_hat = self.master.solve()
            self.lb = lb
            
            '''Solve subs'''
            self.sub.updateX(x_hat)
            single_cut_grad = {i:0 for i in self.data.I}
            single_cut_int = 0
            h_hat = 0
            for w in range(self.data.num_scenarios):
                p_w, sample_w = self.data.sample_scenario(w)
                self.sub.updateForScenario(w, sample_w)
                sub_objval, sub_cut_grad, sub_cut_int = self.sub.solve()
                h_hat += p_w*sub_objval 
                for i in self.data.I:
                    single_cut_grad[i] = single_cut_grad[i] + p_w*sub_cut_grad[i]
                single_cut_int =  single_cut_int + p_w*sub_cut_int
            ub_k = cx_hat + h_hat
            self.ub = ub_k if ub_k < self.ub else self.ub
            '''Termination criteria'''
            print('%4i %10.4f %10.4f' %(self.iteration,self.lb, self.ub))
            if self.check_stop():
                break
            
            else:
                self.master.add_cut(single_cut_grad, single_cut_int)
                self.iteration +=1
        self.master.print_sol()
    
class DSC_Sub():
    '''
    Implements the second stage of the prepositioning model
    '''
    def __init__(self,data):
        self.data=data
        self.model = Model("sub")
        self.model.params.OutputFlag = 0
        self.y = None #Flow variable
        self.x = None #First stage place holder
        self.flow_bal = None #Flow balance constraint holder
        
    def build_model(self):
        '''Set the problem orientation'''
        self.model.modelSense = GRB.MINIMIZE
        data = self.data
        
        '''Create decision variables'''
        # First stage placeholder
        self.x = self.model.addVars(data.I,
                                    vtype=GRB.CONTINUOUS, 
                                    lb=0,
                                    ub=0,
                                    obj=0,
                                    name='x_hat');
        self.y = tupledict()
        for (i,j) in data.arcs_data:
            arc = data.arcs_data[i,j]
            self.y[i,j] = self.model.addVar(lb=0, ub=arc['cap'], obj=arc['cost'], vtype=GRB.CONTINUOUS, name='flow[%i,%i]' %(i,j))
                
        self.s_p = self.model.addVars(self.data.N,
                                    vtype=GRB.CONTINUOUS, 
                                    lb=0,
                                    obj=data.BigM,
                                    name='s_plus');
        self.s_m = self.model.addVars(self.data.N,
                                    vtype=GRB.CONTINUOUS, 
                                    lb=0,
                                    obj=data.BigM,
                                    name='s_minus');
                                    
        self.model.update()
        
        '''Flow balance constraints''' 
        self.flow_bal = self.model.addConstrs((self.y.sum(i,'*') - self.y.sum('*', i)  - self.s_p[i] + self.s_m[i] == 0 for i in data.N), 'flow_balance')
        self.model.update()
        self.ready = False
        
    def updateForScenario(self, scenario, sample):
        '''
        Update the subproblem for a new scenario
        Args:
            scenario (int): id of the scenario
            sample (tuple): tuple containingg scenario parameters
        '''
        self.sceNumber = scenario
        destroyed = sample[0]
        demand = sample[1]
        caps = sample[2]
        
        for (i,j) in self.y:
            self.y[i,j].ub = caps[i,j]
        
        for i in self.flow_bal:
            ctr = self.flow_bal[i]
            if i in self.data.I:
                self.model.chgCoeff(ctr, self.x[i], -destroyed[i])
            ctr.RHS = -demand[i]
        self.model.update()
        self.ready = True
    
   
    def updateX(self, x_hat):
        '''
        Update the subproblem for a new iterate x_hat or x_k
        '''
        for i in self.x:
            self.x[i].lb = x_hat[i]
            self.x[i].ub = x_hat[i]
        self.ready = True
    
         
    def solve(self):
        '''
        Solve the subproblem defined for the scenario
        
        Return:
            objval (float): subproblem objective
            sub_cut_grad ( TODO: define objecte type)
        '''
        if self.ready == True:
            #Solve the model
            self.model.optimize()
            self.ready = False      
            status = self.model.status
            objval = None
            if status == GRB.OPTIMAL:
                objval = self.model.ObjVal
                sub_cut_grad = {i:self.x[i].rc for i in self.data.I}
                sub_cut_int = objval - sum(sub_cut_grad[i]*self.x[i].X for i in self.data.I)
                return objval, sub_cut_grad, sub_cut_int
            else:
                raise 'Unexpected result for the subproblem'
                      
        else:
            raise 'Subproblem for scenario %i not ready to optimize' %(self.sceNumber)
        
class DSC_Master():
    '''
    Implements a preposition first-stage model
    '''
    
    def __init__(self, data):
        self.model = Model('DSC_master')
        self.model.params.OutputFlag = 0
        self.x = None 
        self.theta = None
        self.cx = None
        self.data = data
        self.cut_count = 0
    def build_model(self):
        
        data = self.data
        self.model.modelSense = GRB.MINIMIZE

        '''Create decision variables'''
        self.x = self.model.addVars(data.I,
                                    vtype=GRB.CONTINUOUS, 
                                    lb=0,
                                    ub=data.max_prep,
                                    obj=data.prep_cost,
                                    name='x');
        self.model.update()
        self.cx = self.model.getObjective()
        self.theta = self.model.addVar(lb=0, obj=1,vtype=GRB.CONTINUOUS, name='theta');
        
        '''
        Add maximum installation capacity constraint
        '''
        self.model.addConstr(sum(self.x[i] for i in self.data.I), GRB.LESS_EQUAL, 1, 'globalCap');
        self.model.update()
    
    def add_cut(self, cut_grad, cut_int):
        rhs = quicksum(cut_grad[i]*self.x[i] for i in self.x) + cut_int
        self.model.addConstr(self.theta, GRB.GREATER_EQUAL, rhs, 'cut_%i' %(self.cut_count))
        self.cut_count = self.cut_count + 1
        return self.cut_count - 1 
    
    def solve(self):
        self.model.optimize()
        status = self.model.status    
        if status == GRB.OPTIMAL:
            #Save x 
            x_hat = {}
            for i in self.data.I:
                x_hat[i] = self.x[i].x
            lb = self.model.ObjVal
            theta_hat = self.theta.x
            return status, lb, self.cx.getValue(), x_hat, theta_hat
        else:
            raise 'Infeasible or unbounded Master'    
    def print_sol(self):
        print('First stage cost:  %5.2f' %(self.cx.getValue()))
        print('Second stage cost: %5.2f' %(self.theta.X))
    
        for i in self.x:
            if self.x[i].X >0:
                print("%15s %12.3f %12.3f" %(self.data.nodes_data_id['County seat'][i], self.x[i].X, self.data.nodes_data_id['demand'][i]))

class Data_TwoStageModel():
    '''
    Class for data and scenario generation
    
    Attrs:
        N (list of int): List of nodes ids .
        I (list of int): List of nodes ids where fuel can be prepositioned.
    '''
    def __init__(self, network_file=None, distance_func = haversineEuclidean):
        
        nodes_data, net_edges = self.read_file(network_file)
        nodes_data['County'] = nodes_data.index
        self.N = [i for i in range(len(nodes_data))]
        nodes_data['ID'] = self.N
        nodes_data_id = nodes_data.set_index('ID')
        self.I = [i for i in nodes_data[nodes_data['supply']>0]['ID']] 
        
        self.max_prep = {i:2*nodes_data_id['supply'][i] + nodes_data_id['demand'][i] for i in self.I}
        self.prep_cost = {i:100 if nodes_data_id['supply'][i]>0 else 200 for i in self.I}
        
        self.arcs_data = {}
        for c_i in net_edges:
            v_i = nodes_data['ID'][c_i]
            for c_j in net_edges[c_i]:
                if c_i!=c_j and c_j in nodes_data.index:
                    v_j = nodes_data['ID'][c_j]
                    dist_ij = distance_func(radians(nodes_data.latitude[c_i]), radians(nodes_data.longitude[c_i]),
                                        radians(nodes_data.latitude[c_j]), radians(nodes_data.longitude[c_j]))
                    arc_data = {'cap':1, 'cost':np.round(dist_ij,3)}
                    self.arcs_data[(v_i,v_j)] = arc_data
        
        self.BigM = 1000
        
        self.num_scenarios = 100
        self.nodes_data = nodes_data
        self.nodes_data_id = nodes_data_id
        
        
    def sample_scenario(self, w):
        p_w = 1.0/self.num_scenarios
        
        destroyed = {i:1 for i in self.I}
        
        for i in self.I:
            u = np.random.uniform()
            if u>0.9:
                destroyed[i] = np.random.uniform(0.2,0.8)
        demand = {i:0 for i in self.N}
        for i in self.N:
            nom_dem = self.nodes_data_id['demand'][i]
            spike = np.random.exponential(3)
            demand[i] = nom_dem*spike
        caps = {(i,j):1 for (i,j) in self.arcs_data}
        
        return p_w, (destroyed,demand,caps)
        
    def read_file(self, netwrok_file):
        fl_df, fl_edges = pickle.load(open(netwrok_file, 'rb'))
        fl_df = fl_df.set_index('County')
        total_population = sum(fl_df.Population)
        fl_df['demand'] = fl_df['Population'] / total_population
        fl_df['supply'] = 0
        
        #===========================================================================
        # Set ports supply
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
        assert np.abs(sum(fl_df['supply']) - 1.0) < 1E-8, 'Supply mismatch %f' % (sum(fl_df['supply']))
        assert np.abs(sum(fl_df['demand']) - 1.0) < 1E-8, 'Demand mismatch %f' % (sum(fl_df['demand']))
        fl_df['b'] = fl_df['supply'] - fl_df['demand']
        
        return fl_df, fl_edges
        
    
if __name__ == '__main__':
    np.random.seed(1)
    data = Data_TwoStageModel(network_file='../data/floridaNetObj.p', distance_func=haversineEuclidean)
    algo = DSC_Benders(data)
    algo.run()
