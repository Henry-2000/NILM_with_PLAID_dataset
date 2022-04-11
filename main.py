import os
import time

import pandas as pd
import numpy as np
from tabulate import tabulate
from six.moves import cPickle as pickle
from tkinter import *

from process_data import *
from generate_graphs import *
from utilities import get_obj_size
from cnn import cnn_main

# Datasets paths
plaid_path = ""
metadata_submetered_path = plaid_path +"metadata_submetered.json"
metadata_aggregated_path = plaid_path + "metadata_aggregated.json"
submetered_path = plaid_path + "submetered_new/"
aggregated_path = plaid_path + "aggregated/"
data_path = plaid_path + "data/"
s_matrix_path = data_path + '/s_matrix/'
residue_path = data_path + '/residue/'


def main():
    available_options=np.arange(1,14)
    print("\nChoose an option below: \n\
            \n(1) See appliances mapping of submetered data\
            \n(2) See appliances mapping of aggregated data\
            \n\n############ LOAD CLASSIFICATION APPROACH ############\
            \n(3) Construct steady samples dictionary\
            \n(4) Construct harmonic dictionary from steady samples\
            \n(5) Generate graphs of submetered data with steady samples (Optional)\
            \n(6) Generate graphs of submetered data with phase and amplitude spectrum (Optional)\
            \n(7) Generate V-I trajectories images\
            \n(8) Construct Convolutional Neural Network for V-I trajectories classification\
            \n\n############ EVENT DETECTION APPROACH ############\
            \n(9) Construct aggregated data dictionary\
            \n(10) Construct S-Transform matrix dictionary of aggregated data\
            \n(11) Construct residuals of power signals\
            \n(12) Generate graphs of aggregated data\
            \n(13) Exit\n")
    while True:
        x = int(input("Option: "))
        if x in available_options:
            break
        print("Invalid option.")

    if x==1:
        # Show CSV files numbers of submetered data for each appliance type
        submetered_dict=metadata_submetered(metadata_submetered_path)
        df=pd.DataFrame()
        appliance_type=[]
        for appliance_type in submetered_dict:
            print("\n\n" + 30*"--" +f"\n '{appliance_type}' file numbers\n"+30*"--"+"\n")
            i=1
            for file_number in submetered_dict[appliance_type]:
                if i%11==0:
                    print('\n')
                    i=1
                print(f"{file_number:5s}",end=" ")
                i+=1
        print("\n\n" + 30*"--"+"\n\n")
        main()

    if x==2:
        # Show appliances types involved in each CSV file number of aggregated data with on/off samples
        aggregated_dict=metadata_aggregated(metadata_aggregated_path)
        df=pd.DataFrame()
        sample=[]
        appliances=[]
        
        for sample in aggregated_dict:
            print("\n" + 30*"--" +f"\nFile number: {sample}\n"+30*"--"+"\n")
            for appliance_type in aggregated_dict[sample]:
                #print(f"{appliance}: ON = sample {appliance['on'][0]} / OFF = sample {appliance['off'][0]}")
                print(f"{appliance_type}: ON = samples {aggregated_dict[sample][appliance_type]['on']} / OFF = samples {aggregated_dict[sample][appliance_type]['off']}")
        print("\n" + 30*"--"+"\n")
        main()

    if x==3:
        # Get steady samples of current from submetered data and save in dictionary whose keys are appliances names
        start_time = time.time()
        print("Getting steady samples...")
        metadata_dict=metadata_submetered(metadata_submetered_path)
        steady_samples_dict=steady_samples_submetered(submetered_path,metadata_dict)
        if not os.path.exists(data_path):
            os.makedirs(data_path)  
        print("Saving dictionary...")
        with open(data_path + '/steady_samples_dict.pkl', 'wb') as f: 
            pickle.dump(steady_samples_dict, f, pickle.HIGHEST_PROTOCOL)        
        print(f"\nExecution Time: {int(time.time() - start_time)} seconds\n")
        print(f"Steady samples dictionary saved in '{data_path}'")
        main()

    if x==4:
        # Construct harmonic dictionary from steady samples dictionary of submetered data
        start_time = time.time()    
        print("Loading steady samples dictionary...")
        with open(data_path + "steady_samples_dict.pkl", "rb") as f:
            signal_dict = pickle.load(f)        
        print("Constructing harmonics dictionary...")           
        harmonic_dict = construct_harmonics_dict(signal_dict,21) 
        print("Saving harmonics dictionary...") 
        with open(data_path + '/harmonic_dict.pkl', 'wb') as f: 
            pickle.dump(harmonic_dict, f, pickle.HIGHEST_PROTOCOL)       
        print(f"\nExecution Time: {int(time.time() - start_time)} seconds\n")
        print(f"Harmonics dictionary saved in '{data_path}'")
        main()

    if x==5:
        # Generate graphs of current from submetered data showing selected steady sample intervals
        start_time = time.time()
        print("Loading steady samples dictionary...")
        with open(data_path + "steady_samples_dict.pkl", "rb") as f:
            signal_dict = pickle.load(f)         
        print("\nSaving images...")
        generate_graphs_submetered(signal_dict)
        print(f"\nExecution Time: {time.time() - start_time} seconds")
        main()

    if x==6:
        # Generate graphs from submetered data with reconstructed harmonic signals in frequency domain 
        start_time = time.time()
        print("Loading harmonics dictionary...")
        with open(data_path + "harmonic_dict.pkl", "rb") as f:
            harmonic_dict = pickle.load(f)      
        print("Loading steady samples dictionary...") 
        with open(data_path + "steady_samples_dict.pkl", "rb") as f:
            signal_dict = pickle.load(f)     
        print("\nSaving images...")
        generate_graphs_frequency_domain(signal_dict, harmonic_dict)
        print(f"\nExecution Time: {time.time() - start_time} seconds")
        main()

    if x==7:
        # Generate VI images from steady samples signals
        start_time = time.time()
        print("Loading harmonics dictionary...")
        with open(data_path + "harmonic_dict.pkl", "rb") as f:
            harmonic_dict = pickle.load(f)          
        print("\nSaving images...")
        generate_VI_images(harmonic_dict,highest_odd_harmonic_order=21)
        print(f"\nExecution Time: {time.time() - start_time} seconds")
        main()

    if x==8: 
        # CNN implementation for V-I trajectories classification
        cnn_main()
        main()

    if x==9:
        # Construct aggregated dictionary of aggregated data whose keys are file numbers
        start_time = time.time()   
        metadata_dict=metadata_aggregated(metadata_aggregated_path)
        print("Constructing aggregated dictionary...") 
        aggregated_dict=construct_aggregated_dict(aggregated_path,metadata_dict) 
        print("Saving aggregated dictionary...") 
        with open(data_path + '/aggregated_dict.pkl', 'wb') as f: 
            pickle.dump(aggregated_dict, f, pickle.HIGHEST_PROTOCOL)
        print(f"\nExecution Time: {int(time.time() - start_time)} seconds\n")
        main()

    if x==10:
        # Construct S matrix of of power signals as of voltage and current from aggregated data  (each row of matrix are one harmonic component)
        start_time = time.time()  
        data_dict=metadata_aggregated(metadata_aggregated_path)
        print("Loading aggregated dictionary...") 
        with open(data_path + 'aggregated_dict.pkl', "rb") as f:
            aggregated_dict = pickle.load(f) 
        print("Constructing S matrix files...") 
        construct_s_matrix_signals(aggregated_dict,data_dict,s_matrix_path) 
        print(f"\nExecution Time: {int(time.time() - start_time)} seconds\n")
        main()

    if x==11:
        # Construct residuals of power signals as of voltage and current from aggregated data 
        start_time = time.time()
        data_dict=metadata_aggregated(metadata_aggregated_path)
        print("Constructing residue files...") 
        construct_residual_signals(data_dict,residue_path,s_matrix_path) 
        print(f"\nExecution Time: {int(time.time() - start_time)} seconds\n")
        main()

    if x==12:
        # Generate graphs from aggregated data 
        start_time = time.time()
        with open("Metadata/aggregated_dict.pkl", "rb") as f:
            aggregated_dict = pickle.load(f)         
        print("\nSaving images...")
        generate_graphs_aggregated(aggregated_dict)
        print(f"\nExecution Time: {time.time() - start_time} seconds")
        main()
        
    if x==13:
        exit()


if __name__ == '__main__':
    main()
    





