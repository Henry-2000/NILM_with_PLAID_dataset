import os
import json
import numpy as np
import pandas as pd
from six.moves import cPickle as pickle

from harmonics import reconstruct, filter_harmonics, select_significant_harmonics
from steady_samples import generate_rms, get_indices
from utilities import count_progress, lag_value_in_degrees
from kalman_filter import KF_implementation
from s_transform import s_transform

s_matrix_path='data/s_matrix/'

# Load data from submetered metadata file
def metadata_submetered(metadata_file_submetered):
    f = open(metadata_file_submetered,'r')
    metadata = json.load(f)
    appliance_dict={}
    for file_number in metadata:
        appliance_dict[metadata[file_number]["appliance"]["type"]]=[]
    for file_number in metadata:
        if metadata[file_number]["appliance"]["type"] in appliance_dict:
            appliance_dict[metadata[file_number]["appliance"]["type"]].append(file_number)        
    f.close()
    
    return appliance_dict

# Load data from aggregated metadata file
def metadata_aggregated(metadata_file_aggregated):    
    f = open(metadata_file_aggregated,'r')
    metadata = json.load(f)
    aggregated_dict={}
           
    for file_number in metadata:
        aggregated_dict[file_number]={}
        for appliance in metadata[file_number]['appliances']:
            appliance['on']=appliance['on'].strip('][').split(' ')
            appliance['on']=[int(x) for x in appliance['on'] if x]    
            appliance['off']=appliance['off'].strip('][').split(' ')
            appliance['off']=[int(x) for x in appliance['off'] if x]  
            aggregated_dict[file_number][appliance['type']]={'on':appliance['on'],'off':appliance['off']}        
    f.close()
   
    return aggregated_dict

# Construct a dictionary with steady samples of current and voltage signals for each appliance
def steady_samples_submetered(submetered_file,data_dict):
    n_files=0
    for i in data_dict:
        n_files+=len(data_dict[i])
    signal_dict={}
    count=0
    for appliance_type in data_dict:             
        app_n=0
        for file in data_dict[appliance_type]:
            appliance_type=appliance_type.replace(" ","_")  
            app_n+=1
            count_progress(n_files,count)
            count+=1
            signal_dict[f"{appliance_type}_{app_n}_{file}"]={'appliance_type':appliance_type,'indices':None,'current':None,'current_rms':None,'voltage':None,'voltage_rms':None,'error_value':None}
            with open(submetered_file + file +'.csv') as csv_file:
                csv_reader = pd.read_csv(csv_file, header=None, names=(['current','voltage']))
                current=np.array(csv_reader['current'],dtype=np.float64)
                voltage=np.array(csv_reader['voltage'],dtype=np.float64)               
                sample_cycles=12
                error_image=0
                current_rms=generate_rms(current,mode='full_cycle')
                indices=get_indices(current_rms,mode='full_cycle',sample_cycles=sample_cycles)               
                if indices==None:                    
                    error_image=1
                    indices=get_indices(current_rms,None)                       
                signal_dict[f"{appliance_type}_{app_n}_{file}"]['current']=current                
                signal_dict[f"{appliance_type}_{app_n}_{file}"]['voltage']=voltage                
                signal_dict[f"{appliance_type}_{app_n}_{file}"]['indices']=indices
                signal_dict[f"{appliance_type}_{app_n}_{file}"]['error_value']=error_image
                
    return signal_dict


def construct_aggregated_dict(aggregated_file,data_dict):
    n_files=len(data_dict)      
    signal_dict={}    
    count=0    
    for sample_number in data_dict:  
        signal_dict[sample_number]={'current':None,'voltage':None,'appliances':data_dict[sample_number]}
        count_progress(n_files,count)
        count+=1           
        with open(aggregated_file + sample_number +'.csv') as csv_file:
            csv_reader = pd.read_csv(csv_file, header=None, names=(['current','voltage']))

            current=np.array(csv_reader['current'],dtype=np.float64)                          
            voltage=np.array(csv_reader['voltage'],dtype=np.float64)
                                    
            signal_dict[sample_number]['current']=current                
            signal_dict[sample_number]['voltage']=voltage                
                                           
    return signal_dict

def construct_s_matrix_signals(aggregated_dict,data_dict,filename):
    n_files=len(aggregated_dict)      
    
    signal_dict={}   
    # Power signal frequency in Hz
    power_frequency=120 
    # Gauss window width in Hz
    gauss_width=2*power_frequency
    # Range of frequencies from harmonic frequency (see s_transform function)
    range_from_harmonic=1/2*power_frequency
    # Number of significant harmonics to extract
    n_harmonics=10
    count=0
    for i in range(1,n_files+1):
        signal_dict={'combined_harmonics': None,'zero_frequency':None,'noise':None,'harmonic_list':None,'appliances':data_dict[str(i)]}
                    
        count_progress(n_files,count)
        count+=1

        current=aggregated_dict[str(i)]['current']                         
        voltage=aggregated_dict[str(i)]['voltage'] 

        N=len(current)

        combination=voltage*current
        combination_fft=np.fft.fft(combination)

        harmonic_list_by_magnitude=select_significant_harmonics(combination_fft,rank_size=n_harmonics,grid_frequency=power_frequency)
        
        combination_ST_matrix,noise_combination,gauss_comb,combination_fft_ST,fstart_comb,fend_comb=s_transform(combination,gauss_width,range_from_harmonic,harmonic_list_by_magnitude,grid_frequency=power_frequency)
        
        combination_ST=np.zeros(N,dtype='complex_')
        
        for harmonic in combination_ST_matrix:
            if harmonic!=0:
                combination_ST+=combination_ST_matrix[harmonic]
        signal_dict['combined_harmonics']=combination_ST      
        signal_dict['zero_frequency']=combination_ST_matrix[0]
        signal_dict['noise']=noise_combination
        signal_dict['harmonic_list']=harmonic_list_by_magnitude

        if not os.path.exists(filename):
                os.makedirs(filename)
        with open(f"{filename}s_matrix_dict_file{i}.pkl", 'wb') as f: 
            pickle.dump(signal_dict, f, pickle.HIGHEST_PROTOCOL) 
        
                               

def construct_residual_signals(metadata_aggregated,filename,s_matrix_path):
    n_files=len(metadata_aggregated)   
    count=0   
    for i in range(1,n_files+1):    
        count_progress(n_files,count)
        count+=1
                   
        with open(f"{s_matrix_path}s_matrix_dict_file{i}.pkl", 'rb') as f: 
            s_matrix_power=pickle.load(f)
        power=s_matrix_power['combined_harmonics']+s_matrix_power['zero_frequency'] 
        
        harmonic_list=s_matrix_power['harmonic_list']
        
        residual_power=KF_implementation(power,zero_frequency_signal=s_matrix_power['zero_frequency'],harmonic_list=harmonic_list)

        if not os.path.exists(filename):
                os.makedirs(filename)
        with open(f"{filename}residual_power_file{i}.pkl", 'wb') as f: 
            pickle.dump(residual_power, f, pickle.HIGHEST_PROTOCOL) 



def construct_aggregated_harmonics_dict(aggregated_dict,highest_harmonic_order,sample_frequency=30000,grid_frequency=60):
    # Number of samples per cycle
    n_cycle=sample_frequency/grid_frequency

    # List of harmonic numbers
    harmonic_list = range(1,highest_harmonic_order+1)
    
    # Constructing aggregated_harmonic_dict
    aggregated_harmonic_dict={} 
    for sample_number,value in aggregated_dict.items(): 
        aggregated_harmonic_dict[sample_number]={}
        indices = value.get('indices')
        for interval in indices:
            aggregated_harmonic_dict[sample_number][tuple(interval)]={}  
            for harmonic_order in harmonic_list:
                aggregated_harmonic_dict[sample_number][tuple(interval)][harmonic_order]=[] 
         
    
    n_samples=len(aggregated_dict)
    
    count=0
    for sample_number,value in aggregated_dict.items(): 
        count_progress(n_samples,count)
        count+=1       
        current = value.get('current')
        indices = value.get('indices') 
        for interval in indices:
            current_interval=current[interval[0]:interval[1]]
            current_fft_amp,current_fft_phase=filter_harmonics(current_interval,highest_harmonic_order)
            current_fft=current_fft_amp*np.exp(current_fft_phase*1j)
            current_decomposed,THD_current=reconstruct(current_fft,21,int(n_cycle))
            for harmonic_order in current_decomposed:  
                aggregated_harmonic_dict[sample_number][tuple(interval)][harmonic_order]=current_decomposed[harmonic_order]
    
    return aggregated_harmonic_dict
    
        
def construct_harmonics_dict(signal_dict,highest_harmonic_order):
    
    # List of harmonic numbers
    harmonic_list = range(1,highest_harmonic_order+1)
    
    # Constructing harmonic_dict
    harmonic_dict={} 
    for appliance_name,value in signal_dict.items():   
        appliance_type = value.get('appliance_type')            
        harmonic_dict[appliance_type]={'appliance':{},'mean_lag':[],'mean_THD_current':[],'max_current':[],'first_harmonic_mag':[],'harmonics_proportions':{}} 
    
    n_appliances=len(signal_dict)
    
    count=0
    for appliance_name,value in signal_dict.items():        
        appliance_type = value.get('appliance_type')
        error_value = value.get('error_value') 
        harmonic_dict[appliance_type]['appliance'][appliance_name]={'error_value':error_value,'THD_current':None, 'THD_voltage':None,'harmonic_order':{}}
        count_progress(n_appliances,count)
        count+=1    
                   
        for harmonic_order in harmonic_list:
            harmonic_dict[appliance_type]['appliance'][appliance_name]['harmonic_order'][harmonic_order]={'current': [],'voltage':[]}                      
            harmonic_dict[appliance_type]['harmonics_proportions'][harmonic_order]=[]
        indices=value.get('indices')   
        current=value.get('current')
        voltage=value.get('voltage')
        voltage=voltage[indices[0]:indices[1]]
        current=current[indices[0]:indices[1]]

        current_fft_amp,current_fft_phase=filter_harmonics(current,highest_harmonic_order)
        current_fft=current_fft_amp*np.exp(current_fft_phase*1j)
        current_decomposed,THD_current=reconstruct(current_fft,21)
        harmonic_dict[appliance_type]['appliance'][appliance_name]['THD_current']=THD_current

        
        voltage_fft_amp,voltage_fft_phase=filter_harmonics(voltage,1)
        voltage_fft=voltage_fft_amp*np.exp(voltage_fft_phase*1j)
        voltage_decomposed,THD_voltage=reconstruct(voltage_fft,1)
        harmonic_dict[appliance_type]['appliance'][appliance_name]['harmonic_order'][1]['voltage']=(voltage_decomposed[1])
        harmonic_dict[appliance_type]['appliance'][appliance_name]['THD_voltage']=THD_voltage
        harmonic_dict[appliance_type]['first_harmonic_mag'].append(max(current_decomposed[1]))
        lag=lag_value_in_degrees(current,voltage)
        if harmonic_dict[appliance_type]['appliance'][appliance_name]['error_value']==0:
            harmonic_dict[appliance_type]['mean_lag'].append(lag)
            harmonic_dict[appliance_type]['mean_THD_current'].append(THD_current)
            harmonic_dict[appliance_type]['max_current'].append(max(current))

        for harmonic_order in current_decomposed:         
            harmonic_dict[appliance_type]['appliance'][appliance_name]['harmonic_order'][harmonic_order]['current']=(current_decomposed[harmonic_order])
            harmonic_dict[appliance_type]['harmonics_proportions'][harmonic_order].append(max(current_decomposed[harmonic_order])/max(current_decomposed[1]))           
           
    for appliance_type in harmonic_dict:
        harmonic_dict[appliance_type]['mean_lag']=int(np.mean(harmonic_dict[appliance_type]['mean_lag'])) 
        harmonic_dict[appliance_type]['mean_THD_current']=np.mean(harmonic_dict[appliance_type]['mean_THD_current'])
        harmonic_dict[appliance_type]['max_current']=max(harmonic_dict[appliance_type]['max_current'])
        for harmonic_order in harmonic_dict[appliance_type]['harmonics_proportions']:
            harmonic_dict[appliance_type]['harmonics_proportions'][harmonic_order]=np.mean(harmonic_dict[appliance_type]['harmonics_proportions'][harmonic_order])
    return harmonic_dict














    





