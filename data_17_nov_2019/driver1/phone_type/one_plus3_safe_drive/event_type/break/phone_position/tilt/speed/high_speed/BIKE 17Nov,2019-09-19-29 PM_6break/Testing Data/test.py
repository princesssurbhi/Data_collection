import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('All_Details.txt')
#%%
def kalmanFilter(data):
    data = np.array(data)
    size = len(data)
    n_iter = size
    
    Q = 1e-5     # process variance
    
    xhat=np.zeros(size)      # a posteri estimate of x
    P=np.zeros(size)         # a posteri error estimate
    xhatminus=np.zeros(size) # a priori estimate of x
    Pminus=np.zeros(size)    # a priori error estimate
    K=np.zeros(size)         # gain or blending factor
    
    R = 0.1**2 # estimate of measurement variance, change to see effect

    # intial guesses
    xhat[0] = 0.0
    P[0] = 1.0
    
    for k in range(1,n_iter):
        # time update
        xhatminus[k] = xhat[k-1]
        Pminus[k] = P[k-1]+Q
    
        # measurement update
        K[k] = Pminus[k]/( Pminus[k]+R )
        xhat[k] = xhatminus[k]+K[k]*(data[k]-xhatminus[k])
        P[k] = (1-K[k])*Pminus[k]
    
    return xhat
#%%
threshold_value = 4
y_filter = kalmanFilter(data['y'])
y_double = kalmanFilter(y_filter)
y_triple = kalmanFilter(y_double)
y_tres = y_triple[1:] <= threshold_value
count = 0
state = 'f'
for d in y_tres : 
    if state == 'f':
        if d:
            count += 1
            state = 't'
    else:
        if not d:
            state = 'f'
print(count)