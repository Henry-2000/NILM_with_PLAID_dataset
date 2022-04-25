import numpy as np
import sys 
import gc
import re

from steady_samples import generate_rms, get_indices


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

# Given a classificator (clf) and a numerical vector of classes (X), returns best probabilities
# in literal form with a label binarizer (lb)
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

def lag_value_in_degrees(current,voltage,sample_frequency=30000,grid_frequency=60): 
    # Samples per cycle
    samples_per_cycle=int(sample_frequency/grid_frequency)
    i_cross_zero=find_nearest_index(current,0)
    v_cross_zero=find_nearest_index(voltage[i_cross_zero:i_cross_zero+int(samples_per_cycle/2)+1],0)+i_cross_zero
    if i_cross_zero-v_cross_zero<-samples_per_cycle/4:
        lag=-int(i_cross_zero+samples_per_cycle/2-v_cross_zero)*360/samples_per_cycle
    else:
        lag=-int(i_cross_zero-v_cross_zero)*360/samples_per_cycle
        
    return lag

# Force lag between signals (lag in degrees)
def shift_phase(current,voltage,lag=0,sample_frequency=30000,grid_frequency=60): 
    samples_per_cycle=int(sample_frequency/grid_frequency)
    phase=int(lag_value_in_degrees(current,voltage)*samples_per_cycle/360)
    
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
    """
    Given a resolution of samples (fixed number of samples), returns indexes of samples considered
    to be signal peaks

    'width': integer number of resolution samples that will be summed 
    to give the peak magnitude
    
    'factor': integer number of resolution samples above and below width 
    that will be summed to give neighbors magnitude 
    """ 
    samples_per_cycle=sample_frequency/grid_frequency
    n = len(signal_rms)  
    if mode=='full_cycle':
        resolution=int(samples_per_cycle)
    elif mode=='half_cycle':
        resolution=int(samples_per_cycle/2)
    else:
        resolution=int(number_of_samples)
    # Range of samples above and below target width
    limit=int(factor*resolution)
    # Number of samples inside target width
    target_width=int(width*resolution)
    # List of peak indexes
    event_indexes=[]
    # Number of samples from center of target to limit value
    i=int(limit+target_width/2)
    while i<int(n-(limit+target_width/2)):
        # Sum all neighbor samples after target width
        superior_total=sum(signal_rms[int(i+target_width):int(i+target_width+limit):resolution])
        # Sum all neighbor samples before target width
        inferior_total=sum(signal_rms[int(i-target_width-limit):int(i-target_width+1):resolution])
        if width>1:
            # Sum all samples within target width
            target_total=sum(signal_rms[int(i-target_width/2):int(i+target_width/2+1):resolution])     
        else:
            target_total=signal_rms[int(i)]
        # If sum of samples of target width are greater than its neighbors, it's considered a peak
        if target_total>inferior_total+superior_total:
            # Discretize RMS signal inside target width to only return indexes at the center of cycles
            signal_rms[int(i-target_width/2):int(i+target_width/2+1):resolution]
            # Add maximum value index inside target width to list of peak indexes
            event_indexes.append(int(np.argmax(signal_rms[int(i-target_width/2):int(i+target_width/2+1)], axis=0)+i-target_width/2))
            # Jump to next batch of samples outside limit
            i+=limit
        else:
            # If not a peak, go to next batch of samples
            i+=resolution
    return event_indexes

def acronym_maker(list_of_appliances):
    for number,appliance in enumerate(list_of_appliances):
        if re.search("[B|b]ulb",appliance)!=None:
            list_of_appliances[number]="ILB"
        elif re.search("[W|w]ashing",appliance)!=None:
            list_of_appliances[number]="WM"
        elif re.search("[A|a]ir",appliance)!=None:
            list_of_appliances[number]="AC"
        elif re.search("[F|f]ridge",appliance)!=None:
            list_of_appliances[number]="Fridge"
        elif re.search("[L|l]amp",appliance)!=None:
            list_of_appliances[number]="FCL"
        elif re.search("[W|w]ater",appliance)!=None:
            list_of_appliances[number]="W. Kettle"
        elif re.search("[C|c]offee",appliance)!=None:
            list_of_appliances[number]="Coffee M."
        elif re.search("[I|i]ron",appliance)!=None:
            list_of_appliances[number]="S. Iron"
    
    return list_of_appliances


# Turn multidimensional lists into unidimensional list
def flatten(list_to_be_flatten):
    return [item for sublist in list_to_be_flatten for item in sublist]







