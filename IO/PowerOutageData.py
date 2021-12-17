'''
Created on Aug 29, 2018

@author: dduque
'''
import pickle
import pandas as pd
import numpy as np
import difflib
import datetime

if __name__ == '__main__':
    
    network_file='../data/floridaNetObj.p'
    fl_df, fl_edges = pickle.load(open(network_file, 'rb'))
    #print(fl_df['County'])
    yearH = 2017
    monthH = 9
    dateSet = range(9,15)
    #hourSet = [0,3,6,9,12,15,18,21]
    hourSet = [0,6,12,18]
    demandDict = {}
    
    with open('../data/power_outage_data.p', 'rb') as fp:
        data = pickle.load(fp)
        time_stamps = list(data.keys())
        time_stamps.sort()
        countyList = list(data[time_stamps[1]].keys())
        
        county_map = {}
        for c in countyList:
            str_c = str.title(c)+' County'
            if str_c in list(fl_df['County']):
                county_map[c] = str_c
            else:
                try: 
                    county = difflib.get_close_matches(str_c, list(fl_df['County']))
                    str_c = county[0]
                    county_map[c] = str_c
                except:
                    raise 'County %s was not found'.format(c)
        
        
        
        # for a list of designated time
        timePoint = []
        for day in dateSet:
            for hour in hourSet:
                tp = datetime.datetime(yearH,monthH,day,hour)
                if (tp >= min(time_stamps))and(tp <= max(time_stamps)):
                    timePoint.append(tp)
        # find the closest outage data time points to take a lienar combination
        for tp in timePoint:
            demandDict[tp] = {}
            if tp in time_stamps:
                for c in countyList:
                    demandDict[tp][c] = data[tp][c][0]/data[tp][c][1]
            else:
                tdList = [ts - tp for ts in time_stamps]
                lowerDiff = max([td for td in tdList if td.total_seconds() <= 0])
                largestLess = lowerDiff + tp
                upperDiff = min([td for td in tdList if td.total_seconds() >= 0])
                smallestGreat = upperDiff + tp
                portionLow = upperDiff.total_seconds()/(lowerDiff.total_seconds() + upperDiff.total_seconds())
                portionHigh = lowerDiff.total_seconds()/(lowerDiff.total_seconds() + upperDiff.total_seconds())
                assert np.abs(portionLow + portionHigh - 1)<1E-8,  'Sum adds to %f' %(portionLow + portionHigh)
                # obtain the percentage
                for c in countyList:
                    demandDict[tp][c] = data[largestLess][c][0]/data[largestLess][c][1]*portionLow + \
                        data[smallestGreat][c][0]/data[smallestGreat][c][1]*portionHigh
                        
        # generate the county factor
        # 0 - 20%: 1
        # 20 - 40%: 1.2
        # 40 - 60%: 1.4
        # 60 - 80%: 1.6
        # 80 - 100%: 1.8
        # missing Duval County because of the contaminated data
        factorDict = {}
        for (t,tp) in enumerate(timePoint):
            factorDict[tp] = {}
            for cc in countyList:
                c = county_map[cc]
                if demandDict[tp][cc] <= 0.2:
                    factorDict[tp][c] = 1.0
                elif demandDict[tp][cc] <= 0.4:
                    factorDict[tp][c] = 1.15
                elif demandDict[tp][cc] <= 0.6:
                    factorDict[tp][c] = 1.30
                elif demandDict[tp][cc] <= 0.8:
                    factorDict[tp][c] = 1.45
                else:
                    factorDict[tp][c] = 1.5
                    
                if demandDict[tp][cc] >= 0.0:   
                    print(t, tp, c, '  ',  factorDict[tp][c], )
            print(t+1, tp)  
                    
        with open('../data/factor_data.p', 'wb') as fp2:
            pickle.dump(factorDict, fp2, protocol=pickle.HIGHEST_PROTOCOL)
        

                
       
        
