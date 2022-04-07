
import numpy as np

fs=30000
fn=60
dt=1/fs
n_cycle=fs/fn

# Construct RMS signal
def generate_rms(signal,mode=None,number_of_cycles=12,sample_frequency=30000,grid_frequency=60):
    n = len(signal)   
    samples_per_cycle=sample_frequency/grid_frequency
    duration=n/sample_frequency   
    time   = np.linspace(0,duration,n)
    signal_rms=np.array([])
    if mode=='half_cycle':
        resolution=samples_per_cycle/2
    elif mode=='full_cycle':
        resolution=samples_per_cycle
    else:
        resolution=number_of_cycles
    interv=np.arange(0,len(time),resolution)
    for i in interv:
        signal_pow=0                      
        if (i+resolution)<=(len(time)):
            signal_pow=[signal[j]**2 for j in range(int(i),int(i+(resolution)))]
            signal_pow=sum(signal_pow)
            i_rms=[np.sqrt(signal_pow/(resolution))]*int(resolution)
            signal_rms=np.concatenate((signal_rms, i_rms), axis=None)  
        else:
            signal_pow=[signal[j]**2 for j in range(int(i),int(len(time)-i))]
            signal_pow=sum(signal_pow)
            i_rms=[np.sqrt(signal_pow/(len(time)-i))]*int(len(time)-i)
            signal_rms=np.concatenate((signal_rms, i_rms), axis=None)  
    return signal_rms

# Get indices of samples that have most steady cycles
def get_indices(signal_rms,mode=None,sample_cycles=None,aggregated=0,sample_frequency=30000,grid_frequency=60):
    sample_dict={}
    sample_frag=[]
    n = len(signal_rms) 
    samples_per_cycle=sample_frequency/grid_frequency
    if mode=='half_cycle':
        resolution=samples_per_cycle/2
    else:
        resolution=samples_per_cycle
   
    #med_rms=np.mean(signal_rms)
    if sample_cycles==None:
        
        sample_cycles=12
        for k in range(int(n/(resolution)-sample_cycles*samples_per_cycle/resolution)+1):
            inf=int(k*resolution)
            sup=int(inf+sample_cycles*samples_per_cycle)        
            med=np.mean(signal_rms[inf:sup])     
            sample_dict[(inf,sup)]=np.var(signal_rms[inf:sup]/med)
        indices=min(sample_dict, key=sample_dict.get)
        indices=list(indices)
        return indices
    elif aggregated==0:                
        for k in range(int(n/(resolution)-sample_cycles*samples_per_cycle/resolution)+1):
            inf=int(k*resolution)
            sup=int(inf+sample_cycles*samples_per_cycle)        
            med=np.mean(signal_rms[inf:sup])                   
            flag=0            
            if med>0.03:
                if all(signal_rms[inf:sup]>0.9*med) and all(signal_rms[inf:sup]<1.1*med):
                    sample_dict[(inf,sup)]=np.var(signal_rms[inf:sup]/med)
        if sample_dict!={}:            
            indices=min(sample_dict, key=sample_dict.get)
            indices=list(indices)           
            return indices
        else:
            return None
    else:
        k=0
        while k<=int(n/(n_cycle/2)-sample_cycles):
            inf=int(k*n_cycle/2)
            sup=int(inf+n_cycle*sample_cycles)        
            med=np.mean(signal_rms[inf:sup])                   
            flag=0          
            if med>0.01:
                for j in range(2*sample_cycles):   
                    med_local=(signal_rms[inf+(j-1)*(int(n_cycle/2))]+signal_rms[inf+j*(int(n_cycle/2))])/2                         
                    if med_local>1.01*med or med_local<0.99*med:
                        flag=1
                        break
                if flag==0:
                    if sample_frag!=[]:
                        if sample_frag[-1][1]==inf:
                            sample_frag[-1][1]=sup
                        else:
                            sample_frag.append([inf,sup])
                    else:
                        sample_frag.append([inf,sup])
                    k+=2*sample_cycles-1
            k+=1
        print(sample_frag)
        return sample_frag





# Check voltage rms signal for any rapid voltage changes (RVC) according to IEC 61000-4-30
def check_rvc(signal_rms,threshold=3.3/100,sample_cycles=120,sample_frequency=30000,grid_frequency=60):
    n_cycle=sample_frequency/grid_frequency
    hysteresis=0.5*threshold    
    event_indexes=[]
    n = len(signal_rms)   
    inf=0
    while inf<=n-sample_cycles*n_cycle/2:
        start_rvc=None
        end_rvc=None    
        start_event=None
        end_event=None
        sup=inf+sample_cycles*n_cycle/2
        if signal_is_steady_state(signal_rms[int(inf):int(sup)])==False:
            start_rvc=int(inf+j*n_cycle/2)
            start_event=start_rvc
            flag=0
            while flag==0 or sup<n:                          
                inf+=n_cycle/2
                sup=inf+sample_cycles*n_cycle/2
                mean_rms=np.mean(signal_rms[int(inf):int(sup)])+0.1                                                             
                if all(signal_rms[int(inf):int(sup)]+0.1<(1+hysteresis)*mean_rms) and all(signal_rms[int(inf):int(sup)]+0.1>(1-hysteresis)*mean_rms) and end_rvc==None:
                    end_rvc=int(inf)                                        
                    if sup>end_rvc+sample_cycles*n_cycle/2:
                        flag=1 
                        end_event=int(sup)                               
            event_indexes.append((start_event,end_event))
        else:
            inf+=n_cycle/2        
    return event_indexes


def signal_is_steady_state(signal,threshold=3.3/100,sample_frequency=30000,signal_frequency=60):    
    signal_rms=generate_rms(signal)
    mean=np.mean(signal_rms)+0.1
    samples_per_cycle=sample_frequency/signal_frequency
     
    n = len(signal_rms)   
    for i in range(int(n/(samples_per_cycle/2))):
        if (signal_rms[i]+0.1>mean*(1+threshold) or signal_rms[i]+0.1<mean*(1-threshold)):
            return False
    return True

    





