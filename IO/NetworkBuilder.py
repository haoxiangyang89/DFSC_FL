'''
Created on Feb 16, 2018

@author: dduque
'''

'''
Created on Feb 10, 2018

@author: dduque


Helper module to handle addresses, coordinates, and maps visualization
'''
import gmplot
import numpy as np
from geopy.geocoders import Nominatim
from geopy.geocoders import GoogleV3
import webbrowser, os
import pandas as pd
import pickle
from math import radians, sin, cos, asin,sqrt
project_path = "/Users/haoxiangyang/Desktop/Git/DFSC_FLORIDA"

google_api_key = 'AIzaSyA00v5PtM4GweIHCRJJDSFex96ZDmOM1oE'


geolocator = GoogleV3(api_key=google_api_key) #Key from ddv account. 

R = 6371 #Earth radius in Km
def haversineEuclidean(lat1, lon1, lat2, lon2):
    '''
    Latitudes and longitudes in radians
    '''
    dLat = lat2 - lat1;
    dLon = lon2 - lon1;
    a = sin(dLat/2)**2 + (sin(dLon / 2)** 2)*cos(lat1)*cos(lat2);
    c = 2*asin(sqrt(a));
    return R * c;

def addresses_to_coordinates(book_addres):
    '''
    Build a dictionary of addresses and coordinates
    Args:
        book_address (list of str): a list of address to be translated
    Returns:
        A map of address to lat-lon coordinates
    '''
    assert type(book_addres)==list, 'Input should be a list'
    
    addresses_mapping = {}
    
    for ars in book_addres:
        location = geolocator.geocode(ars, timeout=100)
        if location!=None:
            addresses_mapping[ars] = {'latitude':location.latitude, 'longitude':location.longitude}
        else:
            location = geolocator.geocode(ars.replace('seat',''), timeout=100)
            if location!=None:
                addresses_mapping[ars] = {'latitude':location.latitude, 'longitude':location.longitude}
            
    return addresses_mapping

def vizualize_addresses(addresses_list):
    
    coordinates = addresses_to_coordinates(addresses_list)
    lats = []
    lons =[]
    for ac in coordinates:
        lats.append(coordinates[ac]['latitude'])
        lons.append(coordinates[ac]['longitude'])
    print(coordinates)
    lat_bar = sum(lats)/len(lats)
    lon_bar = sum(lons)/len(lons)
    filename = '../output/mymap.html'
    gmap = gmplot.GoogleMapPlotter(lat_bar,lon_bar, 16)
    gmap.plot(lats, lons, 'cornflowerblue', edge_width=5)
    gmap.draw(filename)
    webbrowser.open('file://' + os.path.realpath(filename))


def read_florida(file_name = "../data/florida_counties.txt"):
    my_file = open(file_name, 'r')
    counties = []
    for l in my_file:
        counties.append(l.replace('\n','') + ", Florida")
    return counties    

def read_us_adj(file_name = "../data/adjMatrixUS.txt", visualize_net = False):
    
    florida_data = pd.read_excel('../data/florida_counties.xlsx', sheet_name='CountyInf')
    #florida_data = florida_data.set_index('County')
    florida_data['latitude'] = 0.0
    florida_data['longitude'] = 0.0
    county_seats = list(florida_data['County seat'])
    for i in range(len(florida_data['County seat'])):
        county_seats[i]=county_seats[i]+', FL'
    con_coordinates = addresses_to_coordinates(county_seats)
    assert len(con_coordinates)==len(county_seats)
    
    for i in range(len(florida_data['County seat'])):
        county_seat_i = florida_data['County seat'][i]+', FL'
        coordinates = con_coordinates[county_seat_i]
        florida_data['latitude'][i] = coordinates['latitude']
        florida_data['longitude'][i] = coordinates['longitude']
        
    df = pd.read_table(file_name, header=None) #US adjancency matrix. 
    fl_edges = {}
    current_org = None
    for r_index in range(len(df.get_values())):
        org_county = df[0][r_index]
        if pd.isnull(df[0][r_index]) == False: # is an origin'
            org_county_name = org_county.split(',')[0]
            org_state = org_county.split(',')[1].strip()
            if org_state == 'FL': #is Florida
                current_org = org_county_name
                if org_county not in fl_edges:
                    fl_edges[current_org] = [df[2][r_index].split(',')[0]]
            else:
                current_org = None
        else:
            if current_org!=None:
                fl_edges[current_org].append(df[2][r_index].split(',')[0])
    
    florida_network = (florida_data,fl_edges)
    pickle.dump(florida_network, open('../data/FloridaNetObj.p', 'wb', pickle.HIGHEST_PROTOCOL))
    
    if visualize_net:    
        lats = []
        lons =[]
        new_df = florida_data.set_index('County')
        
        lat_bar = sum(new_df['latitude'])/len(new_df['latitude'])
        lon_bar = sum(new_df['longitude'])/len(new_df['longitude'])
        filename = '../output/florida_network.html'
        gmap = gmplot.GoogleMapPlotter(lat_bar,lon_bar, 6)
        #gmap.scatter(new_df['latitude'], new_df['longitude'], color = 'red', marker ="*" size = 5)
        for con in new_df.index:
            for dest in fl_edges[con]:
                if dest in new_df.index and con in fl_edges[dest]:
                    lats = [new_df['latitude'][con], new_df['latitude'][dest]]
                    lons = [new_df['longitude'][con], new_df['longitude'][dest]]
                    gmap.plot(lats, lons, 'cornflowerblue', edge_width=5)
        gmap.draw(filename)
        webbrowser.open('file://' + os.path.realpath(filename))
    


def load_florida_network(t_lenght, t_ini, T_max ,partition_network=True, zone = 2):
    '''
    Data preparation of Florida network
    
    Return:
        fl_df (Pandas.DataFrame): DF with information of each county
        fl_edges (dic of str-list): Adjacent counties data structure
        demand_nodes (list of str): List of county names that represent demands
        supply_nodeS (list of str): List of county names (with and extra _S) that
                                    represent supply ports.
    '''
    
    truck_speed = 80 #km/h
    totaltrucks = 1000 # number of trucs in the network
    truck_cap = 230#Barrels/Truck
    network_file = project_path + '/data/FloridaNetObj.p'
    fl_df, fl_edges = pickle.load(open(network_file, 'rb'))
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
    if partition_network == False: 
        supply_nodes  = fl_df.loc[fl_df['supply']>0].index.tolist()
        demand_nodes = fl_df.loc[fl_df['demand']>0].index.tolist()
    else:
        # zone  = #1:North 2: Center, 3 South
        lat_23 = 27.391234 #27.391234, -81.338838
        lat_12 = 29.703668#29.703668, -82.329412
        '''Zone definition ''' 
        fl_df['zone'] = 1
        fl_df.loc[fl_df['latitude']<=lat_12 , 'zone'] = 2
        fl_df.loc[fl_df['latitude']<=lat_23 , 'zone'] = 3
        
        supply_nodes  = fl_df.loc[(fl_df['supply']>0) & (fl_df['zone']==zone)].index.tolist()
        demand_nodes = fl_df.loc[(fl_df['demand']>0) & (fl_df['zone']==zone)].index.tolist()
    
    supply_df = fl_df.loc[supply_nodes].copy()
    supply_df['demand'] = 0
    supply_df.index = [nn+'_S' for nn in supply_df.index]
    supply_nodes  = supply_df.index.tolist()
    trucks = {sn:totaltrucks*fl_df['supply'][sn[:-2]] for sn in supply_nodes}#Per Port}
    
    fl_df['supply'] = 0 #Old dataframe only has demands
    fl_df = fl_df.append(supply_df)
    
    avg_fl_daily_consumption = 153000
    fl_df['supply'] = fl_df['supply']*avg_fl_daily_consumption
    fl_df['demand'] = fl_df['demand']*avg_fl_daily_consumption
    #Transform demand and supply to t_steps intervals
    fl_df['demand'] = t_lenght*fl_df['demand']/24
    fl_df['supply'] = t_lenght*fl_df['supply']/24
        
        
    net_nodes = set()
    net_nodes.update(demand_nodes)
    net_nodes.update(supply_nodes)
    net_nodes = list(net_nodes)
    
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
    
    nominal_demand = {(t,i):fl_df['demand'][i] for t in range(t_ini,T_max,t_lenght) for i in demand_nodes}
    print('Demand: ', sum(fl_df['demand'][demand_nodes]) , '  Supply ' , sum(fl_df['supply'][supply_nodes]))
    return fl_df, fl_edges, demand_nodes, supply_nodes, net_nodes, Tau_max, tau_arcs, trucks, truck_cap, nominal_demand


'''
Testing methods
'''
t_mode = 'none'
if __name__ == '__main__':
    if t_mode == 'atc':
        add1 = '1010 Main st Evanston IL'
        add2 = 'Cl. 127a #11b-2 a 11b-52 Bogota'
        test_addres = [add1,add2]
        result = addresses_to_coordinates(test_addres)
        for r in result:
            print(r, result[r])
    
    elif t_mode == 'va':
        add1 = '1010 Main st Evanston IL'
        add2 = 'Pasco county FL'
        vizualize_addresses([add1,add2])
    elif t_mode == 'read':
        read_us_adj( visualize_net=True)
        #counties = read_florida()
        #vizualize_addresses(counties)
        #county_map = addresses_to_coordinates(counties)
