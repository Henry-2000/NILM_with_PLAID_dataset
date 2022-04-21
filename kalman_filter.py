import numpy as np

# Execute EKF implementation for transients detection 
def KF_implementation(signal,zero_frequency_signal=None,harmonic_list=None,sample_frequency=30000,grid_frequency=60):
    # Samples per signal cycle
    samples_per_cycle=int(sample_frequency/grid_frequency)
    # Number of samples
    n=len(signal)   
    # Number of harmonics
    N=len(harmonic_list)
    # Dimension of state_vector
    dim=2*N+1
    # Frequency (rad/s) times sample period (s) -> w.dt
    phi=2*np.pi/samples_per_cycle
    # Construct Transition Matrix
    T_matrix = np.zeros((dim,dim))   
    j=0    
    for i in range(0,N*2,2):
        T_matrix[i][i]=np.cos(2*(harmonic_list[j])*phi)       
        T_matrix[i+1][i+1]=np.cos(2*(harmonic_list[j])*phi)       
        T_matrix[i+1][i]=np.sin(2*(harmonic_list[j])*phi)        
        T_matrix[i][i+1]=-np.sin(2*(harmonic_list[j])*phi)    
        j+=1      
    T_matrix[-1][-1]=1
    # Construct Process Matrix   
    H_matrix=[1,0]*N
    H_matrix.append(1)
    
    H_matrix=np.array(H_matrix,dtype=int)   
    H_matrix=np.expand_dims(H_matrix, axis=0)   

    # Construct Process Covariance Error Matrix (Priori (k-1) and Posteriori Previous (k-1 (-)))  
    PCE_priori = 1e-2*max(signal)*np.identity(dim)  
    
    # Construct Process Error Covariance Matrix and set it's singular scalar
    Qk=np.identity(dim)
    qk=1
    # Set Measurement Error Covariance Scalar initial value
    Rk=1
    # Auxiliar identity matrix
    I=np.identity(dim)
    # Construct State Vector (Priori)
    X_k_est_prior=np.zeros((dim,1),dtype='complex_')
    # Inicialize Residual Vector
    residual=[]
    # Set smooth constant
    alfa=0.15
    # Begin iterations  
    for k in range(n):   
        # Last element of column vector X_k_est_prior gets DC component value
        X_k_est_prior[-1][0]=zero_frequency_signal[k]

        X_k_pred = T_matrix @ X_k_est_prior
        
        PCE_posteriori = T_matrix @ PCE_priori @ T_matrix.T + Qk  
        
        Kg = PCE_posteriori @ H_matrix.T @ np.linalg.pinv(H_matrix @ PCE_posteriori @ H_matrix.T + Rk)        
        
        residual.append(signal[k] - (H_matrix @ X_k_pred))
        
        X_k_est_post= X_k_pred + Kg @ residual[k]
        
        Rk = alfa*Rk + (1-alfa)*np.abs(residual[k]**2 - H_matrix @ PCE_priori @ H_matrix.T)       
        
        PCE_posteriori_plus= (I - Kg @ H_matrix) @ PCE_posteriori
        
        Sk= H_matrix  @ PCE_posteriori_plus @ H_matrix.T + Rk
        
        qk = alfa*qk + (1-alfa)*np.abs((residual[k]**2 -Sk) @ np.linalg.pinv(H_matrix @ H_matrix.T)) 
        
        Qk=qk*I
        
        X_k_est_prior=X_k_est_post

        PCE_priori = PCE_posteriori_plus

    residual=np.reshape(residual,(-1))
      
    return residual
        





