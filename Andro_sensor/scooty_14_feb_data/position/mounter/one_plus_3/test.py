import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
#%%
ll = pd.read_csv('left_left.csv', skiprows=1)

col = ['ACCELEROMETER X (m/s²)', 'ACCELEROMETER Y (m/s²)',
       'ACCELEROMETER Z (m/s²)', 'GRAVITY X (m/s²)', 'GRAVITY Y (m/s²)',
       'GRAVITY Z (m/s²)', 'LINEAR ACCELERATION X (m/s²)',
       'LINEAR ACCELERATION Y (m/s²)', 'LINEAR ACCELERATION Z (m/s²)',
       'LIGHT (lux)', 'MAGNETIC FIELD X (μT)', 'MAGNETIC FIELD Y (μT)',
       'MAGNETIC FIELD Z (μT)','Satellites in range','SOUND LEVEL (dB)',
       'YYYY-MO-DD HH-MI-SS_SSS']


ll.drop(col, axis=1, inplace=True)

#%%
def plotting(colno):
    plt.plot(ll[ll.columns[colno]])
    plt.show()

scaler_o_z = MinMaxScaler()
scaler_o_z.fit(ll[['ORIENTATION Z (azimuth °)']])
plt.plot(scaler_o_z.transform(ll[['ORIENTATION Z (azimuth °)']]))


scaler_g_z = MinMaxScaler()
scaler_g_z.fit(ll[['GYROSCOPE Z (rad/s)']])
plt.plot(scaler_g_z.transform(ll[['GYROSCOPE Z (rad/s)']]))

scaler_o_y = MinMaxScaler()
scaler_o_y.fit(ll[['ORIENTATION Y (roll °)']])
plt.plot(scaler_o_y.transform(ll[['ORIENTATION Y (roll °)']]))