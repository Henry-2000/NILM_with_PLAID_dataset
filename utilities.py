import numpy as np
import sys 
import gc


# Do operations to list of arrays of different sizes 
def handle_arrays(arrays,operation=None):
    lens = [len(i) for i in arrays]
    arr = np.ma.empty((np.max(lens),len(arrays)))
    arr.mask = True
    for idx, l in enumerate(arrays):
        arr[:len(l),idx] = l
    if operation==None:
        return arr.T
    elif operation=='sum':
        return list(arrays.sum(axis=1))
    elif operation=='variance':
        return arr.var(axis=None)
    else:
        return None


def get_best_tags(clf, X, lb, n_tags=3):
    decfun = clf.decision_function(X)
    best_tags = np.argsort(decfun)[:, :-(n_tags+1): -1]
    return lb.classes_[best_tags]           

# Utility to display progress in percentage
def count_progress(number_of_files,count):
    if (count)%(int(np.ceil(number_of_files/1000)))==0:
        progress=np.round(count/number_of_files*100,decimals=1)
        print(f"Progress: {progress}%",end='\r')

# Calculate lag by number of samples
def lag_value(current,voltage,sample_frequency=30000,grid_frequency=60): 
    # Samples per cycle
    n_cycle=sample_frequency/grid_frequency
    imax=max(current)
    imin=min(current)
    if abs(imax)>abs(imin):
        i_amp=imax
    else:
        i_amp=imin
    ind_current=find_nearest_index(current,i_amp)
    ind_current+=125
    ind_current=(ind_current)%(n_cycle/2)
    if ind_current>n_cycle/4:
        ind_current=ind_current-n_cycle/2
        
    vmax=max(voltage)
    vmin=min(voltage)
    if abs(vmax)>abs(vmin):
        v_amp=vmax
    else:
        v_amp=vmin
    ind_voltage=find_nearest_index(voltage,v_amp)
    ind_voltage+=125
    ind_voltage=(ind_voltage)%(n_cycle/2)
    if ind_voltage>n_cycle/4:
        ind_voltage=ind_voltage-n_cycle/2
        
    lag=ind_current-ind_voltage    
    
    if lag>n_cycle/4:
        lag=lag-n_cycle/2
    if lag<-n_cycle/4:
        lag=n_cycle/2+lag     
    return int(lag)

# Force lag between signals (lag in number of samples)
def shift_phase(current,voltage,lag=0): 
    phase=lag_value(current,voltage)
    
    phase+=lag
      
    if int(phase)>0:
        current=current[int(phase):]
        voltage=voltage[:-int(phase)]        
    elif int(phase)<0:
        current=current[:int(phase)]
        voltage=voltage[int(abs(phase)):]  
    return current,voltage,phase

# Return index of array based on closest value
def find_nearest_index(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def get_obj_size(obj):
    marked = {id(obj)}
    obj_q = [obj]
    sz = 0

    while obj_q:
        sz += sum(map(sys.getsizeof, obj_q))

        # Lookup all the object referred to by the object in obj_q.
        # See: https://docs.python.org/3.7/library/gc.html#gc.get_referents
        all_refr = ((id(o), o) for o in gc.get_referents(*obj_q))

        # Filter object that are already marked.
        # Using dict notation will prevent repeated objects.
        new_refr = {o_id: o for o_id, o in all_refr if o_id not in marked and not isinstance(o, type)}

        # The new obj_q will be the ones that were not marked,
        # and we will update marked with their ids so we will
        # not traverse them again.
        obj_q = new_refr.values()
        marked.update(new_refr.keys())

    return sz

def find_peaks_rms(signal_rms,sample_frequency=30000,grid_frequency=60,factor=5,width=3,mode=None,number_of_samples=12):
    samples_per_cycle=sample_frequency/grid_frequency
    n = len(signal_rms)  
    if mode=='full_cycle':
        resolution=int(samples_per_cycle)
    elif mode=='half_cycle':
        resolution=int(samples_per_cycle/2)
    else:
        resolution=int(number_of_samples)
    limit=int(factor*resolution)
    target_width=int(width*resolution)
    event_indexes=[]
    i=int(limit+target_width/2)
    while i<int(n-(limit+target_width/2)):
        superior_total=sum(signal_rms[int(i+target_width):int(i+target_width+limit):resolution])
        inferior_total=sum(signal_rms[int(i-target_width-limit):int(i-target_width+1):resolution])
        if width>1:
            target_total=sum(signal_rms[int(i-target_width/2):int(i+target_width/2+1):resolution])     
        else:
            target_total=signal_rms[int(i)]
        if target_total>inferior_total+superior_total:
            signal_rms[int(i-target_width/2):int(i+target_width/2+1):resolution]
            event_indexes.append(np.argmax(signal_rms[int(i-target_width/2):int(i+target_width/2+1)], axis=0)+i-target_width/2)
            i+=limit
        else:
            i+=resolution
    return event_indexes

def flatten(list_to_be_flatten):
    return [item for sublist in list_to_be_flatten for item in sublist]





