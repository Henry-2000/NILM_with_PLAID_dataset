
import numpy as np
from utilities import shift_phase

# Separate harmonics in a dictionary in the format: {harmonic number(int):[array of harmonic values],...}
def reconstruct(signal_in_fft,highest_harmonic_order,length=None):
    dict_harmonics={}
    harmonic_pairs=[]
    harmonic_indices=np.where(abs(signal_in_fft)!=0)  # Gets all indices where signal in frequency domain isn't zero
    for i in range(len(harmonic_indices[0])//2):    # Construct list of indices pairs as tuples (negative and positive frequencies)
        harmonic_pairs.append((harmonic_indices[0][i],harmonic_indices[0][-i-1]))    
    mag_list=[]
    harmonic_number=1   # Starts with first harmonic
    for i,j in harmonic_pairs:  
        if harmonic_number<=highest_harmonic_order:
            harmonic_in_time_domain=np.zeros(len(signal_in_fft)).astype('complex128')  # Zero complex array in range of signal
            harmonic_in_time_domain[i]=signal_in_fft[i]          # First index of pair
            harmonic_in_time_domain[j]=signal_in_fft[j]          # Second index of pair 
            harmonic_in_time_domain=np.fft.ifft(harmonic_in_time_domain)        # Inverse Fourier Transform of pair
            mag_list.append(max(harmonic_in_time_domain.real))
            if length==None:
                dict_harmonics[harmonic_number]=harmonic_in_time_domain.real
            else:
                dict_harmonics[harmonic_number]=harmonic_in_time_domain.real[0:length]
            harmonic_number+=1          
    THD=sum(np.square(mag_list[1:]))**0.5/mag_list[0]           # Total Harmonic Distortion (THD) formula
    return dict_harmonics,THD             #Returns harmonic dictionary and THD value

# Filter noise from signal
def filter_harmonics(signal,highest_harmonic_order=None,sample_frequency=30000,grid_frequency=60):
    signal=np.array(signal,dtype=np.float32) 
    # Sample interval (s)
    dt=1/sample_frequency
    # Samples per cycle
    n_cycle=sample_frequency/grid_frequency
    # Number of samples
    n=len(signal)
    # Remainder samples of integer number of cycles
    remainder = n % n_cycle
    # Signal has to have integer number of cycles to do appropriate FFT 
    if remainder !=0:
        signal=signal[:-remainder]
        n = len(signal) 
    # Fast Fourier Transform (FFT) of signal
    fft_signal=np.fft.fft(signal,n)
    # Signal magnitude on frequency domain
    fft_signal_amp=np.abs(fft_signal)
    # Signal phase on frequency domain in radians
    fft_signal_phase=np.angle(fft_signal)
    # Frequency axes
    freq_axes = np.fft.fftfreq(n, d=dt)
    # Get only the odd harmonic frequencies
    harmonic_indices=[]
    if highest_harmonic_order==None:                   # If no odd order limit is given, gets indices 
        first_half=np.arange(-grid_frequency,min(freq_axes),-grid_frequency)   # through all range of frequencies, i.e. -fs/2 to fs/2                                     
        second_half=np.arange(grid_frequency,max(freq_axes),grid_frequency)    # in multiples of fn
        harmonic_indices=np.append(first_half,second_half,axis=0) 
    else:                                                               # Else, gets indices through range of -harmonic_order*fn
        first_half=np.arange(-grid_frequency,-grid_frequency*(highest_harmonic_order+1),-grid_frequency)    # to +harmonic_order*fn                                     
        second_half=np.arange(grid_frequency,grid_frequency*(highest_harmonic_order+1),grid_frequency)    
        harmonic_indices=np.append(first_half,second_half,axis=0)
    # Extract frequency indices around each harmonic order frequency
    ind_freq=[np.where((freq_axes >= (harmonic_indices[i]-grid_frequency/10)) & (freq_axes <= (harmonic_indices[i]+grid_frequency/10))) for i in range(len(harmonic_indices))]
    indices=[]
    for i in ind_freq:
        ind_max=np.where(fft_signal_amp == max(fft_signal_amp[i])) # get index where fft_signal is maximum
        ind=np.intersect1d(ind_max,i)    # discard rest of indices
        indices.append(ind)              # create list of indices selected
    # Create fft signal with only the harmonic values
    fft_signal_clean_amp=np.zeros(len(fft_signal_amp))
    fft_signal_clean_phase=np.zeros(len(fft_signal_phase))
    for i in indices:
        fft_signal_clean_amp[i]=fft_signal_amp[i]           # fft_signal clean = fft_signal at indices, 0 otherwise
        fft_signal_clean_phase[i]=fft_signal_phase[i]   

    return fft_signal_clean_amp,fft_signal_clean_phase # Returns magnitude and phase signal without noise in frequency domain


def harmonics_selection(harmonic_dict,highest_odd_harmonic_order,appliance_type,appliance_name,lag=None,odd=0):
    voltage=harmonic_dict[appliance_type]['appliance'][appliance_name]['harmonic_order'][1]['voltage']
    current=np.zeros(len(harmonic_dict[appliance_type]['appliance'][appliance_name]['harmonic_order'][1]['current']))

    if odd==0:
        for i in range(1,highest_odd_harmonic_order+1,1):
            harmonic=harmonic_dict[appliance_type]['appliance'][appliance_name]['harmonic_order'][i]['current']
            current+=harmonic
    else:
        for i in range(1,highest_odd_harmonic_order+1,2):
            harmonic=harmonic_dict[appliance_type]['appliance'][appliance_name]['harmonic_order'][i]['current']
            current+=harmonic
    if lag==1:
        current,voltage,phase=shift_phase(current,voltage,harmonic_dict[appliance_type]['mean_lag'])
    if lag==0:
        current,voltage,phase=shift_phase(current,voltage,lag)

    return current,voltage

# Select harmonic components based on magnitude at frequency domain
def select_significant_harmonics(signal_fft,rank_size=10,sample_frequency=30000,grid_frequency=60):
    N = len(signal_fft)       
    T=1/sample_frequency
    width=int(N*T*grid_frequency)
    signal_fft=np.abs(signal_fft)
    steps=np.arange(width/2,N/2,width)
    harmonic_dict={}
    for i in steps:    
        peak_ind=np.argpartition(signal_fft[int(i):int(i+width)],-1)[-1:]
        ind_value=i+peak_ind[0]
        harmonic_dict[int(ind_value)]=signal_fft[int(ind_value)]
    
    harmonic_list=sorted(harmonic_dict, key=harmonic_dict.get,reverse=True)[:rank_size]
    for k in range(len(harmonic_list)):
        harmonic_list[k]=round(harmonic_list[k]/(N*T*grid_frequency))
    harmonic_list=sorted(harmonic_list) 
        
    return harmonic_list


    





