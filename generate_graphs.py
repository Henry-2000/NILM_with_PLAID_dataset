import os
from os.path import join
import json
import numpy as np
from cycler import cycler
from tkinter import *

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter

from steady_samples import generate_rms
from utilities import shift_phase, count_progress
from harmonics import filter_harmonics, harmonics_selection

# File paths for saving images
filepath_time_domain="graphics/submetered/time_domain"
filepath_frequency_domain="graphics/submetered/frequecy_domain"
filepath_aggregated="graphics/aggregated"
filepath_VI_images="images/v-i_images"

# Default targets signals 
target_appliances=[
    'Air_Conditioner_49_963',
    'Blender_2_1876',
    'Coffee_maker_8_1775',
    'Compact_Fluorescent_Lamp_67_412',
    'Fan_21_211',
    'Fridge_33_912',
    'Hair_Iron_10_1844',
    'Hairdryer_218_1548',
    'Heater_84_1565',
    'Incandescent_Light_Bulb_140_1524',
    'Laptop_102_663',
    'Microwave_35_340',
    'Soldering_Iron_6_1860',
    'Vacuum_4_48',
    'Washing_Machine_18_595',
    'Water_kettle_1_1805'
]

# Defaults text formats
SMALLER_SIZE = 14
SMALL_SIZE = 16
MEDIUM_SIZE = 18
BIGGER_SIZE = 12
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALLER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# Monochrome images
monochrome = (cycler('color', ['k']))
               
# Load data from metadata file
def load_data(dataset_file,metadata_file):
    metadata = join(dataset_file,metadata_file)
    f = open(metadata,'r')
    metadata = json.load(f)
    #print(metadata)
    appliance_dict={}
    for file_number in metadata:
        appliance_dict[metadata[file_number]["appliance"]["type"]]=[]
    for file_number in metadata:
        if metadata[file_number]["appliance"]["type"] in appliance_dict:
            appliance_dict[metadata[file_number]["appliance"]["type"]].append(file_number)        
    f.close()
    return appliance_dict

# Maps the appliances (images) with csv files
def mapping_appliances(dataset_file,metadata_file):
    appliance_dict=load_data(dataset_file,metadata_file)
    #print(appliance_dict)
    map_appliance={}
    for appliance in appliance_dict:
        i=0
        for file_number in appliance_dict[appliance]:
            i+=1
            map_appliance[appliance +str(i)]=file_number
    return map_appliance

def generate_graphs_submetered(signal_dict_original,target_appliances=[],filepath=filepath_time_domain,sample_frequency=30000):
    dt=1/sample_frequency
    count=0
    signal_dict={}

    if target_appliances !=[]:
        for target in target_appliances:
            signal_dict[target]=signal_dict_original[target]
    else:
        signal_dict=signal_dict_original
    n_appliances=len(signal_dict)
    for appliance_name in signal_dict:       
        count_progress(n_appliances,count)
        if count>=1238:
            current=signal_dict[appliance_name]['current']
            indices=signal_dict[appliance_name]['indices']
            appliance_type=signal_dict[appliance_name]['appliance_type']
            current_rms=generate_rms(current,mode='full_cycle')
            fig1,axes1 = plt.subplots(2,1)
            fig1 = plt.gcf()
            fig1.set_size_inches(20, 10)
            fig1.tight_layout(pad=5.0)

            duration=len(current)/sample_frequency
            time= np.linspace(0,duration,num=int(np.ceil(duration/dt)))
            
            plt.grid()
            plt.sca(axes1[0])
            plt.title('(a)',fontsize=20,pad=15)
            axes1[0].plot(time,current,'b',label=f'Corrente - {appliance_name}')
            axes1[0].plot(time,current_rms,'r',label=f'Corrente RMS - {appliance_name}')
            plt.axvline(time[indices[0]],0,1,color='g',lw=3)
            plt.axvline(time[indices[1]-1],0,1,color='g',lw=3)
            plt.xlabel('Tempo [s]')
            plt.ylabel('Corrente [A]')
            plt.xticks(np.arange(0, max(time)+1000*dt,np.around(max(time),1)/20))
            plt.xlim(0,max(time))
            axes1[0].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=1, borderaxespad=0.)
                
            plt.grid()
            plt.sca(axes1[1])
            plt.title('(b)',fontsize=20,pad=15)
            axes1[1].plot(time[indices[0]:indices[1]],current[indices[0]:indices[1]],'b',label=f'Corrente - {appliance_name}')
            axes1[1].plot(time[indices[0]:indices[1]],current_rms[indices[0]:indices[1]],'r',label=f'Corrente RMS - {appliance_name}')
            plt.xlabel('Tempo [s]')
            plt.ylabel('Corrente [A]')
            plt.xlim(time[indices[0]],time[indices[1]-1])
            inf=time[indices[0]]
            sup=time[indices[1]-1]
            axes1[1].xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
            step=((sup-inf)-1000*dt)/10
            plt.xticks(np.arange(inf, sup+100*dt, step))
            axes1[1].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=1, borderaxespad=0.)
            plt.tight_layout()

            if signal_dict[appliance_name]['error_value']==0:
                if not os.path.exists(filepath + "/valid_graphics/"):                
                    os.makedirs(f"{filepath}/valid_graphics")
                if not os.path.exists(f"{filepath}/valid_graphics/{appliance_type}"):
                    os.makedirs(f"{filepath}/valid_graphics/{appliance_type}")                
                plt.savefig(f"{filepath}/valid_graphics/{appliance_type}/{appliance_name}.png") 
            else:
                if not os.path.exists(f"{filepath}/error_graphics"):                
                    os.makedirs(f"{filepath}/error_graphics/")
                if not os.path.exists(f"{filepath}/error_graphics/{appliance_type}"):                
                    os.makedirs(f"{filepath}/error_graphics/{appliance_type}")
                plt.savefig(f"{filepath}/error_graphics/{appliance_type}/{appliance_name}.png")
            plt.close(fig1)
        count+=1

def generate_graphs_aggregated(aggregated_dict_original,target_samples=[],filepath=filepath_aggregated,sample_frequency=30000):

    dt=1/sample_frequency
    count=0
    aggregated_dict={}

    if target_samples!=[]:
        for target in target_samples:
            aggregated_dict[target]=aggregated_dict_original[target]
    else:
        aggregated_dict=aggregated_dict_original
    n_samples=len(aggregated_dict)
    for sample in aggregated_dict:    
        count_progress(n_samples,count)   
        current=aggregated_dict[sample]['current']
        voltage=aggregated_dict[sample]['voltage']
        appliances=aggregated_dict[sample]['appliances']
        appliances_names=appliances[0]
        for i in range(1,len(appliances)):
            appliances_names+=f" + {appliances[i]}"
            
        current_rms=generate_rms(current)
        voltage_rms=generate_rms(voltage)

        fig1,axes1 = plt.subplots(2,1)
        fig1 = plt.gcf()
        fig1.set_size_inches(20, 10)
        fig1.tight_layout(pad=5.0)

        duration=len(current)/sample_frequency
        time= np.linspace(0,duration,num=int(np.ceil(duration/dt)))
        
        plt.grid()
        plt.sca(axes1[0])
        plt.title(f'{appliances_names} - Sample {sample}',fontsize=20,pad=15)
        axes1[0].plot(time,current,'b',label='Current')
        axes1[0].plot(time,current_rms,'r',label='Current RMS')
        plt.xlabel('Time [s]')
        plt.ylabel('Current [A]')
        plt.xticks(np.arange(0, max(time)+1000*dt,np.around(max(time),1)/20))
        plt.xlim(0,max(time))
        axes1[0].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=1, borderaxespad=0.)
               
        plt.grid()
        plt.sca(axes1[1])
        plt.title(f'{appliances_names} - Sample {sample}',fontsize=20,pad=15)
        axes1[1].plot(time,voltage,'b',label=f'Voltage')
        axes1[1].plot(time,voltage_rms,'r',label=f'Voltage RMS')
        plt.xlabel('Time [s]')
        plt.ylabel('Voltage [V]')
        plt.xticks(np.arange(0, max(time)+1000*dt,np.around(max(time),1)/20))
        plt.xlim(0,max(time))
        axes1[1].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=1, borderaxespad=0.)

        plt.tight_layout()

        if not os.path.exists(f"{filepath}"):
            os.makedirs(f"{filepath}")
        plt.savefig(f"{filepath}/{sample}.png") 
        plt.close(fig1)
        count+=1


def generate_graphs_frequency_domain(signal_dict_original,harmonic_dict,target_appliances=[],filepath=filepath_frequency_domain,sample_frequency=30000,grid_frequency=60,highest_harmonic_order=21):
    harmonic_list=range(1,highest_harmonic_order+1,1)
    dt=1/sample_frequency
    signal_dict={}
    if target_appliances !=[]:
        for target in target_appliances:
            signal_dict[target]=signal_dict_original[target]
    else:
        signal_dict=signal_dict_original
    count=0
    n_appliances=len(signal_dict)
    for appliance_name in signal_dict:       
        count_progress(n_appliances,count)
        
        current_original=signal_dict[appliance_name]['current']
        indices=signal_dict[appliance_name]['indices']
        appliance_type=signal_dict[appliance_name]['appliance_type']
    
        n_original=len(current_original)
        duration_original=n_original/sample_frequency
        time_original=np.linspace(0,duration_original,num=int(np.ceil(duration_original/dt)))

        current_interval=current_original[indices[0]:indices[1]]
        time_interval   = time_original[indices[0]:indices[1]]
        n_interval=len(current_interval)

        current_harmonics=np.zeros(len(current_interval))

        for harmonic_order in harmonic_list:    
            current_harmonics+=harmonic_dict[appliance_type]['appliance'][appliance_name]['harmonic_order'][harmonic_order]['current']
        freq_axes = np.fft.fftfreq(n_interval, d=dt)
        fft_amp_interval,fft_phase_interval=filter_harmonics(current_harmonics,highest_harmonic_order)
        fft_amp_interval=fft_amp_interval/n_interval
        
        indices_harmonics=np.where(fft_amp_interval!=0)
        mag_x=[]
        mag_y=[]
        fft_phase_cut_correct=np.zeros(len(fft_phase_interval))
        for i in indices_harmonics[0][:highest_harmonic_order]:
            fft_phase_cut_correct[i]=fft_phase_interval[i]
            mag_x.append(freq_axes[i])
            mag_y.append(fft_amp_interval[i])
            
            
        fig = plt.figure(constrained_layout=True)
        gs = GridSpec(4, 1, figure=fig)
        fig = plt.gcf()
        fig.set_size_inches(15, 15)

        ax1 = fig.add_subplot(gs[0, :])
        plt.plot(time_original,current_original,'b',label='current original')
        plt.ylabel('Corrente [A]',fontsize=14)
        plt.xlabel('Time (s)',fontsize=14)
        plt.axvline(time_original[indices[0]],0,1,color='g',lw=5)
        plt.axvline(time_original[indices[1]-1],0,1,color='g',lw=5)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend()
        plt.grid()

        ax1 = fig.add_subplot(gs[1, :])
        plt.plot(time_interval,current_interval,'b',label='current original')       
        plt.plot(time_interval,current_harmonics,'g',label='current all harmonics')        
        plt.ylabel('Current [A]',fontsize=14)
        plt.xlabel('Time (s)',fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend()
        plt.grid()

        ax1 = fig.add_subplot(gs[2, :])
        plt.stem(freq_axes,fft_amp_interval,'b')
        plt.plot(mag_x,mag_y,'r')
        plt.xlabel('Frequency [Hz]',fontsize=12)
        plt.ylabel('Amplitude I(f) [A]',fontsize=12)
        plt.xticks(np.arange(0,(highest_harmonic_order+1)*grid_frequency,grid_frequency),fontsize=10)
        for i in range(0,highest_harmonic_order):
            plt.text(freq_axes[indices_harmonics[0][i]],fft_amp_interval[indices_harmonics[0][i]],f"{fft_amp_interval[indices_harmonics[0][i]]:.4f}")
        plt.grid()
        plt.xlim(0,(highest_harmonic_order+1)*grid_frequency)

        ax1 = fig.add_subplot(gs[3, :])
        plt.stem(freq_axes,fft_phase_cut_correct,'b')
        plt.xlabel('Frequency [Hz]',fontsize=12)
        plt.ylabel('Phase I(f) [degrees]',fontsize=12)
        for i in range(highest_harmonic_order):
            plt.text(freq_axes[indices_harmonics[0][i]],fft_phase_cut_correct[indices_harmonics[0][i]]*180/np.pi,f"{fft_phase_cut_correct[indices_harmonics[0][i]]:.2f}")
        plt.grid()
        plt.xticks(np.arange(0,(highest_harmonic_order+1)*grid_frequency,grid_frequency),fontsize=10)
        plt.xlim(0,(highest_harmonic_order+1)*grid_frequency)

        if signal_dict[appliance_name]['error_value']==0:
            if not os.path.exists(f"{filepath}"):
                os.makedirs(f"{filepath}")
            if not os.path.exists(f"{filepath}/valid_graphics/{appliance_type}"):
                os.makedirs(f"{filepath}/valid_graphics/{appliance_type}")
            plt.savefig(f"{filepath}/valid_graphics/{appliance_type}/{appliance_name}.png") 
        else:
            if not os.path.exists(f"{filepath}/error_graphics"):
                os.makedirs(f"{filepath}/error_graphics")
            if not os.path.exists(f"{filepath}/error_graphics/{appliance_type}"):
                os.makedirs(f"{filepath}/error_graphics/{appliance_type}")
            plt.savefig(f"{filepath}/error_graphics/{appliance_type}/{appliance_name}.png")
        plt.close(fig)
        count+=1


# Save images of VxI of appliances
def generate_VI_images(harmonic_dict,filepath=filepath_VI_images,highest_odd_harmonic_order=21):   
    count=0   
    n_images=0
    max_current=[]
    low_THD_appliance_type=[]
    for appliance_type in harmonic_dict:
        for appliance_name in harmonic_dict[appliance_type]['appliance']:
            n_images+=1
        if harmonic_dict[appliance_type]['mean_THD_current']<0.05:
            low_THD_appliance_type.append(appliance_type)
            max_current.append(harmonic_dict[appliance_type]['max_current'])
    imax=max(max_current)
    for appliance_type in harmonic_dict:          
        for appliance_name in harmonic_dict[appliance_type]['appliance']:                                    
            mode=0
            odd=0
            for i in range(2):                                 
                current,voltage=harmonics_selection(harmonic_dict,highest_odd_harmonic_order,appliance_type,appliance_name,lag=i,odd=odd)
                image_saving(harmonic_dict,filepath,appliance_type,appliance_name,voltage,current,low_THD_appliance_type,imax,mode)
                mode+=1 
                odd+=1
                current,voltage=harmonics_selection(harmonic_dict,highest_odd_harmonic_order,appliance_type,appliance_name,lag=i,odd=odd)
                image_saving(harmonic_dict,filepath,appliance_type,appliance_name,voltage,current,low_THD_appliance_type,imax,mode)
                mode+=1
                odd-=1
                               
            count_progress(n_images,count)                                  
            count+=1


def generate_VI_images_4x4(harmonic_dict,signal_dict,target_appliances=target_appliances,filepath=filepath_VI_images):
    letters=['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)','(j)','(k)','(l)','(m)','(n)','(o)','(p)']
    current_original=[]
    voltage_original=[]
    current_harmonics=[]
    voltage_harmonics=[]
    appliance_list=[]
    low_THD_appliance_type=[]
    max_current=[]
    for appliance_type in harmonic_dict:
        if harmonic_dict[appliance_type]['mean_THD_current']<0.05:
            low_THD_appliance_type.append(appliance_type)
            max_current.append(harmonic_dict[appliance_type]['max_current'])
    imax=max(max_current)
    print(imax)
    print(low_THD_appliance_type)
    highest_odd_harmonic_order=21
    for appliance_type in harmonic_dict:          
        for appliance_name in harmonic_dict[appliance_type]['appliance']:
            if appliance_name in target_appliances:
                print(appliance_name)
                appliance_list.append(appliance_name)
                voltage_har=harmonic_dict[appliance_type]['appliance'][appliance_name]['harmonic_order'][1]['voltage'] 
                
                current_har=np.zeros(len(harmonic_dict[appliance_type]['appliance'][appliance_name]['harmonic_order'][1]['current']))
                    
                        
                for i in range(1,highest_odd_harmonic_order+1,1):
                    harmonic=harmonic_dict[appliance_type]['appliance'][appliance_name]['harmonic_order'][i]['current']
                    current_har+=harmonic

                current_har,voltage_har,phase=shift_phase(current_har,voltage_har,harmonic_dict[appliance_type]['mean_lag'])

                
                
                current_harmonics.append(current_har)

                voltage_harmonics.append(voltage_har)

                indices=signal_dict[appliance_name]['indices']
                current_dict=signal_dict[appliance_name]['current'][indices[0]:indices[1]]
                voltage_dict=signal_dict[appliance_name]['voltage'][indices[0]:indices[1]]
                current_original.append(current_dict)
                voltage_original.append(voltage_dict)

                
    
    count=0
    fig1,axes1 = plt.subplots(4,4,figsize=(8,8))
    fig1 = plt.figure(frameon = False)
    fig1.set_size_inches(1, 1) 
    for i in range(int(len(current_original)/4)): 
        for j in range(int(len(current_original)/4)):                                
            plt.sca(axes1[i,j])    
            plt.title(f'{letters[count]} {appliance_list[count]}',fontsize=6)
            axes1[i,j].set_prop_cycle(monochrome)
            axes1[i,j].plot(voltage_original[count],current_original[count],'k')
            plt.xlabel('Tensão [V]',fontsize=7)
            plt.ylabel('Corrente [A]',fontsize=7)
            plt.xticks(fontsize=6)
            plt.yticks(fontsize=6)           
            print('hey ya')
            count+=1
    if not os.path.exists(plaid_path + f"{filepath}"):
        os.makedirs(plaid_path + f"{filepath}")
    plt.tight_layout()
    plt.savefig(plaid_path + f"{filepath}/Original Signal Trajectories - VI.png") 
    plt.close(fig1)
    print('damn')
    count=0
    fig2,axes2 = plt.subplots(4,4,figsize=(8,8))
    fig2 = plt.figure(frameon = False)
    fig2.set_size_inches(1, 1) 
    for i in range(int(len(current_harmonics)/4)):      
        for j in range(int(len(current_harmonics)/4)):               
            plt.sca(axes2[i,j])    
            plt.title(f'{letters[count]} {appliance_list[count]}',fontsize=6)
            axes2[i,j].set_prop_cycle(monochrome)
            axes2[i,j].plot(voltage_harmonics[count],current_harmonics[count],'k')
            plt.xlabel('Tensão [V]}',fontsize=7)
            plt.ylabel('Corrente [A]',fontsize=7)
            plt.xticks(fontsize=6)
            plt.yticks(fontsize=6)
            for appliance_type in harmonic_dict:
                if appliance_type in low_THD_appliance_type:
                    if appliance_list[count] in harmonic_dict[appliance_type]['appliance']:
                        plt.ylim(-imax,imax)    
            count+=1
    if not os.path.exists(plaid_path + f"{filepath}"):
        os.makedirs(plaid_path + f"{filepath}")
    plt.tight_layout()
    plt.savefig(plaid_path + f"{filepath}/Filtered Trajectories - VI.png") 
    plt.close(fig2)





def image_saving(harmonic_dict,filepath,appliance_type,appliance_name,voltage,current,low_THD_appliance_type,imax,mode=0):
    fig = plt.figure(frameon = False)
    fig.set_size_inches(1, 1)   
    ax = plt.axes()
    ax.set_prop_cycle(monochrome)
    ax.set_axis_off()
    fig.add_axes(ax)

    img=plt.plot(voltage,current) 

    if appliance_type in low_THD_appliance_type:               
        plt.ylim(-imax,imax)

    if harmonic_dict[appliance_type]['appliance'][appliance_name]['error_value']==0:
        if not os.path.exists(f"{filepath}/valid_images/{appliance_type}"):
            os.makedirs(f"{filepath}/valid_images/{appliance_type}")
        plt.savefig(f"{filepath}/valid_images/{appliance_type}/{appliance_name}_" + str(mode) + ".png",dpi=128) 
    else:
        if not os.path.exists(f"{filepath}/error_images/{appliance_type}"):
            os.makedirs(f"{filepath}/error_images/{appliance_type}")
        plt.savefig(f"{filepath}/error_images/{appliance_type}/{appliance_name}_" + str(mode) + ".png",dpi=128) 
    plt.close(fig)



def format_axes(fig):
    for i, ax in enumerate(fig.axes):
        ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center")
        ax.tick_params(labelbottom=False, labelleft=False)


    





