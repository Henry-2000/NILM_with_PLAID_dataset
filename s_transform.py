
import numpy as np

# Shifts spectrum by number of Hz given
def shift_spectrum(signal_spectrum,shift_value_in_hz,sample_frequency=30000):
    dt=1/sample_frequency
    N=len(signal_spectrum)
    scaled_frequency=int(shift_value_in_hz*N*dt)
    shift=np.ones(abs(scaled_frequency))*signal_spectrum[int(len(signal_spectrum)/2+1)]
    
    pos_freq=signal_spectrum[0:int(len(signal_spectrum)/2)]
    neg_freq=signal_spectrum[int(len(signal_spectrum)/2):]
    
    if scaled_frequency>0:
        pos_freq=pos_freq[0:-scaled_frequency]
        neg_freq=np.append(shift,neg_freq)
    else:
        pos_freq=np.append(pos_freq,shift)
        neg_freq=neg_freq[-scaled_frequency:]    
    
    signal_spectrum_shifted=np.append(neg_freq,pos_freq)
    signal_spectrum_shifted=np.fft.fftshift(signal_spectrum_shifted)
    
    return signal_spectrum_shifted
    

def s_transform(signal,gauss_width,range_from_harmonic,harmonic_list,sample_frequency=30000,grid_frequency=60):
    # Number of samples
    N= len(signal)
    # Sample period
    T=1/sample_frequency
    # Extract FFT of signal and make a copy
    signal_fft=np.fft.fft(signal)
    signal_fft_copy=np.copy(signal_fft)
    # Noise (start with all frequencies)
    noise_fft = signal_fft
    # List of harmonic frequencies
    frequency_list=[i*grid_frequency for i in harmonic_list]
    frequency_list.insert(0,0)
    # Initial Gaussian window width
    fwidth=gauss_width*N*T
    # Initialize empty dictionaries and vectors
    signal_reconstructed={}
    gauss_shifted={}
    signal_fft_aux={}
    fstart_ind=[]
    fend_ind=[]
    # Start of algorithm   
    for fcenter in frequency_list:
        # Initialize frequency boundaries
        fcenter*=N*T
        fcenter=int(fcenter)         
        if fcenter!=0:          
            fstart=int(fcenter-range_from_harmonic*N*T)
            if fstart==0:
                fstart_neg=-1
            else:
                fstart_neg=-fstart
        else:                    
            fstart=0
            fstart_neg=None                            
        fend=int(fcenter+range_from_harmonic*N*T)
        fcenter_neg=-fcenter
        fend_neg=-fend           
        # Gaussian window FFT 
        gauss=[]
        for k in range(int(-N/2),int(N/2)):
            gauss.append(np.exp(-2*(np.pi**2)*(k**2)/(fwidth)**2))
        # Shift zero frequency of gaussian window FFT to be centered in spcetrum domain 
        gauss=np.fft.fftshift(gauss)
        # Step 1 - Shift gaussian window to be centered in harmonic frequency
        gauss_shifted[int(fcenter/(N*T))]=shift_spectrum(gauss,fcenter/(N*T))        
        # Step 2 - Multiply signal FFT by gaussian FFT of frequencies between fstart and fend 
        signal_fft_copy[fstart:fend]=signal_fft[fstart:fend]*gauss_shifted[int(fcenter/(N*T))][fstart:fend]
        # Step 3 - Repeat steps 1 and 2 for the negatives frequencies (fstart_neg, fend_neg, fcenter_neg)
        if fcenter_neg==0:
            gauss_shifted[int((fcenter_neg-N*T)/(N*T))]=shift_spectrum(gauss,fcenter_neg/(N*T)) 
            signal_fft_copy[fend_neg:fstart_neg]=signal_fft[fend_neg:fstart_neg]*gauss_shifted[int((fcenter_neg-N*T)/(N*T))][fend_neg:fstart_neg]       
        else:
            gauss_shifted[int((fcenter_neg)/(N*T))]=shift_spectrum(gauss,fcenter_neg/(N*T)) 
            signal_fft_copy[fend_neg:fstart_neg+1]=signal_fft[fend_neg:fstart_neg+1]*gauss_shifted[int(fcenter_neg/(N*T))][fend_neg:fstart_neg+1]
        # Step 4 - Feed auxiliar fft signal dictionary with fft signal of designated harmonic frequency (each key is one harmonic)
        signal_fft_aux[int(fcenter/(N*T))]=np.zeros(N,dtype='complex64')
        signal_fft_aux[int(fcenter/(N*T))][fstart:fend]=signal_fft_copy[fstart:fend]
        if fcenter==0:
            signal_fft_aux[int(fcenter/(N*T))][fend_neg:fstart_neg]=signal_fft_copy[fend_neg:fstart_neg]
        else:
            signal_fft_aux[int(fcenter/(N*T))][fend_neg:fstart_neg+1]=signal_fft_copy[fend_neg:fstart_neg+1]
        # Zero the frequencies inside boundaries of noise fft signal  
        noise_fft[fstart:fend]=0
        if fcenter==0:
            noise_fft[fend_neg:fstart_neg]=0
        else:
            noise_fft[fend_neg:fstart_neg+1]=0
        # Feed an empty array with the Inverse FFT to obtain separated harmonics in time domain
        signal_reconstructed[int(fcenter/(grid_frequency*N*T))]=np.zeros(N,dtype='complex64')
        signal_reconstructed[int(fcenter/(grid_frequency*N*T))]+=np.fft.ifft(signal_fft_aux[int(fcenter/(N*T))])
        # Convert from 'complex64' to 'float32' to save memory and discard imaginary values (too small to be considered)
    #signal_reconstructed[int(fcenter/(grid_frequency*N*T))]=np.float32(signal_reconstructed[int(fcenter/(grid_frequency*N*T))])
        # Append frequency boundaries to generate gaussian graphs if needed 
        fstart_ind.append(fstart) 
        if fstart_neg==None:
            fstart_ind.append(-1)
        else:
            fstart_ind.append(fstart_neg)
               
        fend_ind.append(fend)
        fend_ind.append(fend_neg)
    # For the remain of noise frequency spectrum, perform Inverse FFT to obtain noise signal approximation
    noise=np.fft.ifft(noise_fft)
        
    return signal_reconstructed,noise,gauss_shifted,signal_fft_aux,fstart_ind,fend_ind





    





