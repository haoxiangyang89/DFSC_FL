'''
Created on Nov 19, 2018

@author: dduque

This module contains classes and functions to create and handle scenarios of fuel demand under 
hurricane disasters
'''



class Forecast():
    '''
    A class to represent a particular forecast
    
    Attributes:
        f_id (int): Time id of the forecast
        f_time (datetime): Time at which the forecast is valid
        f_values (dict): dictionary containing the values of each time in 
            the future for each county. 
    '''

    def __init__(self,f_id, f_time, f_values):
        self.f_id = f_id
        self.f_time = f_time
        self.f_values = f_values
        
    def transform_forecast(self, f_func):
        '''
        Creates a new forecast object applying the function to every value
        Args:
            f_func (func): Function to transform the forecast
        Return:
            A new forecast object
        '''
        new_f_vals = {}
        for t in self.f_values:
            new_f_vals[t] = {}
            for c in self.f_values[t]:
                new_f_vals[t][c] = f_func(self.f_values[t][c])
        
        return Forecast(self.time, new_f_vals)
        
        