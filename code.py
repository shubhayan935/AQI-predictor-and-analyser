#Importing required libraries

import pandas as pd
import numpy as np
import time
import math
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn import linear_model
 
#Defining Functions

def get_PM25_subindex(x):
    if x <= 30:
        return x * 50 / 30
    elif x <= 60:
        return 50 + (x - 30) * 50 / 30
    elif x <= 90:
        return 100 + (x - 60) * 100 / 30
    elif x <= 120:
        return 200 + (x - 90) * 100 / 30
    elif x <= 250:
        return 300 + (x - 120) * 100 / 130
    elif x > 250:
        return 400 + (x - 250) * 100 / 130
    else:
        return 0

def get_SO2_subindex(x):
    if x <= 40:
        return x * 50 / 40
    elif x <= 80:
        return 50 + (x - 40) * 50 / 40
    elif x <= 380:
        return 100 + (x - 80) * 100 / 300
    elif x <= 800:
        return 200 + (x - 380) * 100 / 420
    elif x <= 1600:
        return 300 + (x - 800) * 100 / 800
    elif x > 1600:
        return 400 + (x - 1600) * 100 / 800
    else:
        return 0

def get_NOx_subindex(x):
    if x <= 40:
        return x * 50 / 40
    elif x <= 80:
        return 50 + (x - 40) * 50 / 40
    elif x <= 180:
        return 100 + (x - 80) * 100 / 100
    elif x <= 280:
        return 200 + (x - 180) * 100 / 100
    elif x <= 400:
        return 300 + (x - 280) * 100 / 120
    elif x > 400:
        return 400 + (x - 400) * 100 / 120
    else:
        return 0

def get_CO_subindex(x):
    if x <= 1:
        return x * 50 / 1
    elif x <= 2:
        return 50 + (x - 1) * 50 / 1
    elif x <= 10:
        return 100 + (x - 2) * 100 / 8
    elif x <= 17:
        return 200 + (x - 10) * 100 / 7
    elif x <= 34:
        return 300 + (x - 17) * 100 / 17
    elif x > 34:
        return 400 + (x - 34) * 100 / 17
    else:
        return 0

def get_O3_subindex(x):
    if x <= 50:
        return x * 50 / 50
    elif x <= 100:
        return 50 + (x - 50) * 50 / 50
    elif x <= 168:
        return 100 + (x - 100) * 100 / 68
    elif x <= 208:
        return 200 + (x - 168) * 100 / 40
    elif x <= 748:
        return 300 + (x - 208) * 100 / 539
    elif x > 748:
        return 400 + (x - 400) * 100 / 539
    else:
        return 0

def predict_visualization(x,y):
    poly_reg = PolynomialFeatures(degree=10)
    X_poly = poly_reg.fit_transform(x.reshape(-1,1))
    pol_reg = LinearRegression()
    pol_reg.fit(X_poly, y)
    ext = pol_reg.predict(poly_reg.fit_transform(x.reshape(-1,1)))
    return ext

def predict_values(x,y,n):
    x.to_numpy()
    poly_reg = PolynomialFeatures(degree=10)
    X_poly = poly_reg.fit_transform(x.values.reshape(-1,1))
    pol_reg = LinearRegression()
    pol_reg.fit(X_poly, y)
    ext = pol_reg.predict(poly_reg.fit_transform(np.array(n).reshape(-1,1)))
    return ext

def aqi_calc(so2,co,no,pm2,o3):
    a = [so2,co,no,pm2,o3]
    a.sort()
    aqi = a[4]
    return aqi

def predict_aqi(x,y,n):
    #Predict aqi based on all features on the data
    poly_reg = PolynomialFeatures(degree=2)
    X_poly = poly_reg.fit_transform(x)
    predict_ = poly_reg.fit_transform(n.reshape(1,-1))
    clf = linear_model.LinearRegression()
    clf.fit(X_poly,y)
    answer = clf.predict(predict_)
    return answer

#Import data
filepath = '/Users/shubhayansrivastava/Documents/Python/Practice/Machine_Learning_Project/Air Quality Index Data.xlsx'
df = pd.read_excel(filepath, sheet_name=None)
areas = df.keys()

#Removing additional rows
for i in areas:
    df[i] = df[i].iloc[0:236,:]

for i in areas:
    df[i].fillna(df[i].mean(), inplace=True)

#Plotting all the parameters
for i in areas:
    x = df[i].iloc[:,0]
    x.to_numpy()
    for j in df[i].columns[3:]:
        y = df[i][j]
        plt.figure()
        plt.xlabel(i+'_'+j)
        plt.plot(x,y)

#Checking how many parameters have all NaN values
nan_parameters = list()
for i in areas:
    for j in df[i].columns[3:]:
        if np.isnan(df[i][j][0]):
            nan_parameters.append([i+'_'+j])

for i in range(len(nan_parameters)):
            print(nan_parameters[i],'\n')

#Predicting future values of each parameter with Linear Regression
for i in areas:
    x = pd.Series(df['Bandra']['Sr no.'])
    for j in df[i].columns[3:]:
        if np.isnan(df[i][j][0])==False:
            y = df[i][j]
            y.to_numpy()
            reg = LinearRegression().fit(x.to_numpy().reshape(-1,1),y)
            ext = reg.predict(np.array([[237]]))
            plt_y = np.append(y, ext)
            plt_x = np.append(x, 237)
            if ext < 0:
                print(i,j,ext,'\n')
            plt.figure()
            plt.plot(plt_x, plt_y)


#Creating required dictionaries
pm2 = {'Bandra Kurla':[],
      'Bandra':[],
      'Borivali East':[],
      'Andheri East':[],
      'Airport':[],
      'Colaba':[],
      'Deonar':[],
      'Khandavali East':[],
      'Khindipada-Bhandup West':[],
      'Kurla':[],
      'Malad West':[],
       'Mulund West':[],
      'Navy Nagar-Colaba':[],
      'Powai':[],
      'Siddarth Nagar-Worli':[],
      'Worli':[],
      'Sion':[],
      'Vasai West':[],
      'Vile Parle West':[]}

co = {'Bandra Kurla':[],
      'Bandra':[],
      'Borivali East':[],
      'Andheri East':[],
      'Airport':[],
      'Colaba':[],
      'Deonar':[],
      'Khandavali East':[],
      'Khindipada-Bhandup West':[],
      'Kurla':[],
      'Malad West':[],
       'Mulund West':[],
      'Navy Nagar-Colaba':[],
      'Powai':[],
      'Siddarth Nagar-Worli':[],
      'Worli':[],
      'Sion':[],
      'Vasai West':[],
      'Vile Parle West':[]}

no = {'Bandra Kurla':[],
      'Bandra':[],
      'Borivali East':[],
      'Andheri East':[],
      'Airport':[],
      'Colaba':[],
      'Deonar':[],
      'Khandavali East':[],
      'Khindipada-Bhandup West':[],
      'Kurla':[],
      'Malad West':[],
       'Mulund West':[],
      'Navy Nagar-Colaba':[],
      'Powai':[],
      'Siddarth Nagar-Worli':[],
      'Worli':[],
      'Sion':[],
      'Vasai West':[],
      'Vile Parle West':[]}

so2 = {'Bandra Kurla':[],
      'Bandra':[],
      'Borivali East':[],
      'Andheri East':[],
      'Airport':[],
      'Colaba':[],
      'Deonar':[],
      'Khandavali East':[],
      'Khindipada-Bhandup West':[],
      'Kurla':[],
      'Malad West':[],
       'Mulund West':[],
      'Navy Nagar-Colaba':[],
      'Powai':[],
      'Siddarth Nagar-Worli':[],
      'Worli':[],
      'Sion':[],
      'Vasai West':[],
      'Vile Parle West':[]}

o3 = {'Bandra Kurla':[],
      'Bandra':[],
      'Borivali East':[],
      'Andheri East':[],
      'Airport':[],
      'Colaba':[],
      'Deonar':[],
      'Khandavali East':[],
      'Khindipada-Bhandup West':[],
      'Kurla':[],
      'Malad West':[],
       'Mulund West':[],
      'Navy Nagar-Colaba':[],
      'Powai':[],
      'Siddarth Nagar-Worli':[],
      'Worli':[],
      'Sion':[],
      'Vasai West':[],
      'Vile Parle West':[]}

aqi = {'Bandra Kurla':[],
      'Bandra':[],
      'Borivali East':[],
      'Andheri East':[],
      'Airport':[],
      'Colaba':[],
      'Deonar':[],
      'Khandavali East':[],
      'Khindipada-Bhandup West':[],
      'Kurla':[],
      'Malad West':[],
       'Mulund West':[],
      'Navy Nagar-Colaba':[],
      'Powai':[],
      'Siddarth Nagar-Worli':[],
      'Worli':[],
      'Sion':[],
      'Vasai West':[],
      'Vile Parle West':[]}

X = {'Bandra Kurla':[],
      'Bandra':[],
      'Borivali East':[],
      'Andheri East':[],
      'Airport':[],
      'Colaba':[],
      'Deonar':[],
      'Khandavali East':[],
      'Khindipada-Bhandup West':[],
      'Kurla':[],
      'Malad West':[],
       'Mulund West':[],
      'Navy Nagar-Colaba':[],
      'Powai':[],
      'Siddarth Nagar-Worli':[],
      'Worli':[],
      'Sion':[],
      'Vasai West':[],
      'Vile Parle West':[]}

factors = {'Bandra Kurla':[],
      'Bandra':[],
      'Borivali East':[],
      'Andheri East':[],
      'Airport':[],
      'Colaba':[],
      'Deonar':[],
      'Khandavali East':[],
      'Khindipada-Bhandup West':[],
      'Kurla':[],
      'Malad West':[],
       'Mulund West':[],
      'Navy Nagar-Colaba':[],
      'Powai':[],
      'Siddarth Nagar-Worli':[],
      'Worli':[],
      'Sion':[],
      'Vasai West':[],
      'Vile Parle West':[]}

factors = {'Bandra Kurla':[],
      'Bandra':[],
      'Borivali East':[],
      'Andheri East':[],
      'Airport':[],
      'Colaba':[],
      'Deonar':[],
      'Khandavali East':[],
      'Khindipada-Bhandup West':[],
      'Kurla':[],
      'Malad West':[],
       'Mulund West':[],
      'Navy Nagar-Colaba':[],
      'Powai':[],
      'Siddarth Nagar-Worli':[],
      'Worli':[],
      'Sion':[],
      'Vasai West':[],
      'Vile Parle West':[]}

change = {'Bandra Kurla':{'PM2.5 (ug/m3)':[],
                          'NO (ug/m3)':[],
                          'NO2 (ug/m3)':[],
                          'NOx (ppb)':[],
                          'NH3 (ug/m3)':[],
                          'SO2 (ug/m3)':[],
                          'CO (mg/m3)':[],
                          'Ozone (ug/m3)':[],
                          'Benzene (ug/m3)':[],
                          'Toluene (ug/m3)':[],
                          'Eth-Benzene (ug/m3)':[],
                          'MP-Xylene (ug/m3)':[], 'RH (%)':[], 'WS (m/s)':[], 'WD (degree)':[]},
      'Bandra':{'PM2.5 (ug/m3)':[], 'NO (ug/m3)':[], 'NO2 (ug/m3)':[], 'NOx (ppb)':[],
       'SO2 (ug/m3)':[], 'CO (mg/m3)':[], 'Ozone (ug/m3)':[], 'Benzene (ug/m3)':[],
       'RH (%)':[], 'WS (m/s)':[], 'WD (deg)':[], 'SR (W/mt2)':[], 'BP (mmHg)':[],
       'Xylene (ug/m3)':[], 'AT (degree C)':[], 'RF (mm)':[]},
      'Borivali East':{'PM2.5 (ug/m3)':[], 'NO (ug/m3)':[], 'NO2 (ug/m3)':[], 'NOx (ppb)':[],
       'NH3 (ug/m3)':[], 'SO2 (ug/m3)':[], 'CO (mg/m3)':[], 'Ozone (ug/m3)':[], 'RH (%)':[],
       'WS (m/s)':[], 'WD (degree)':[], 'BP (mmHg)':[], 'RF (mm)':[]},
      'Andheri East':{'PM2.5 (ug/m3)':[], 'NO (ug/m3)':[], 'NO2 (ug/m3)':[], 'NOx (ppb)':[],
       'NH3 (ug/m3)':[], 'SO2 (ug/m3)':[], 'CO (mg/m3)':[], 'Ozone (ug/m3)':[],
       'Benzene (ug/m3)':[], 'Toluene (ug/m3)':[], 'Eth-Benzene (ug/m3)':[],
       'MP-Xylene (ug/m3)':[], 'RH (%)':[], 'WS (m/s)':[], 'WD (degree)':[], 'BP (mmHg)':[],
       'Xylene (ug/m3)':[]},
      'Airport':{'PM2.5 (ug/m3)':[], 'NO (ug/m3)':[], 'NO2 (ug/m3)':[], 'NOx (ppb)':[],
       'NH3 (ug/m3)':[], 'SO2 (ug/m3)':[], 'CO (mg/m3)':[], 'Ozone (ug/m3)':[],
       'Benzene (ug/m3)':[], 'Eth-Benzene (ug/m3)':[], 'MP-Xylene (ug/m3)':[], 'RH (%)':[],
       'WS (m/s)':[], 'WD (degree)':[], 'SR (W/mt2)':[], 'BP (mmHg)':[], 'AT (degree C)':[]},
      'Colaba':{'PM2.5 (ug/m3)':[], 'NO (ug/m3)':[], 'NO2 (ug/m3)':[], 'NOx (ppb)':[],
       'NH3 (ug/m3)':[], 'SO2 (ug/m3)':[], 'CO (mg/m3)':[], 'Ozone (ug/m3)':[],
       'Benzene (ug/m3)':[], 'Eth-Benzene (ug/m3)':[], 'MP-Xylene (ug/m3)':[], 'RH (%)':[],
       'WS (m/s)':[], 'WD (degree)':[], 'SR (W/mt2)':[], 'BP (mmHg)':[], 'AT (degree C)':[]},
      'Deonar':{'PM2.5 (ug/m3)':[], 'NO (ug/m3)':[], 'NO2 (ug/m3)':[], 'NOx (ppb)':[],
       'NH3 (ug/m3)':[], 'SO2 (ug/m3)':[], 'CO (mg/m3)':[], 'Ozone (ug/m3)':[],
       'Benzene (ug/m3)':[], 'Toluene (ug/m3)':[], 'Eth-Benzene (ug/m3)':[],
       'MP-Xylene (ug/m3)':[], 'RH (%)':[], 'WS (m/s)':[], 'WD (degree)':[], 'BP (mmHg)':[],
       'Xylene (ug/m3)':[], 'AT (degree C)':[], 'RF (mm)':[]},
      'Khandavali East':{'PM2.5 (ug/m3)':[], 'NO (ug/m3)':[], 'NO2 (ug/m3)':[], 'NOx (ppb)':[],
       'NH3 (ug/m3)':[], 'SO2 (ug/m3)':[], 'CO (mg/m3)':[], 'Ozone (ug/m3)':[],
       'Benzene (ug/m3)':[], 'Eth-Benzene (ug/m3)':[], 'MP-Xylene (ug/m3)':[], 'RH (%)':[],
       'WS (m/s)':[], 'WD (degree)':[], 'SR (W/mt2)':[], 'BP (mmHg)':[], 'AT (degree C)':[]},
      'Khindipada-Bhandup West':{'PM2.5 (ug/m3)':[], 'NO (ug/m3)':[], 'NO2 (ug/m3)':[], 'NOx (ppb)':[],
       'NH3 (ug/m3)':[], 'SO2 (ug/m3)':[], 'CO (mg/m3)':[], 'Ozone (ug/m3)':[],
       'Benzene (ug/m3)':[], 'Toluene (ug/m3)':[], 'MP-Xylene (ug/m3)':[], 'RH (%)':[],
       'WS (m/s)':[], 'WD (degree)':[], 'BP (mmHg)':[], 'Xylene (ug/m3)':[], 'RF (mm)':[]},
      'Kurla':{'PM2.5 (ug/m3)':[], 'NO (ug/m3)':[], 'NO2 (ug/m3)':[], 'NOx (ppb)':[],
       'NH3 (ug/m3)':[], 'SO2 (ug/m3)':[], 'CO (mg/m3)':[], 'Ozone (ug/m3)':[],
       'Benzene (ug/m3)':[], 'Eth-Benzene (ug/m3)':[], 'MP-Xylene (ug/m3)':[], 'RH (%)':[],
       'WS (m/s)':[], 'WD (degree)':[], 'SR (W/mt2)':[], 'BP (mmHg)':[], 'AT (degree C)':[]},
      'Malad West':{'PM2.5 (ug/m3)':[], 'NO (ug/m3)':[], 'NO2 (ug/m3)':[], 'NOx (ppb)':[],
       'NH3 (ug/m3)':[], 'SO2 (ug/m3)':[], 'CO (mg/m3)':[], 'Ozone (ug/m3)':[],
       'Benzene (ug/m3)':[], 'Eth-Benzene (ug/m3)':[], 'MP-Xylene (ug/m3)':[], 'RH (%)':[],
       'WS (m/s)':[], 'WD (degree)':[], 'BP (mmHg)':[], 'AT (degree C)':[], 'RF (mm)':[]},
       'Mulund West':{'PM2.5 (ug/m3)':[], 'NO (ug/m3)':[], 'NO2 (ug/m3)':[], 'NOx (ppb)':[],
       'NH3 (ug/m3)':[], 'SO2 (ug/m3)':[], 'CO (mg/m3)':[], 'Ozone (ug/m3)':[],
       'Benzene (ug/m3)':[], 'Eth-Benzene (ug/m3)':[], 'MP-Xylene (ug/m3)':[], 'RH (%)':[],
       'WS (m/s)':[], 'WD (degree)':[], 'SR (W/mt2)':[], 'BP (mmHg)':[], 'AT (degree C)':[]},
      'Navy Nagar-Colaba':{'PM2.5 (ug/m3)':[], 'NO (ug/m3)':[], 'NO2 (ug/m3)':[], 'NOx (ppb)':[],
       'NH3 (ug/m3)':[], 'SO2 (ug/m3)':[], 'CO (mg/m3)':[], 'Ozone (ug/m3)':[],
       'Benzene (ug/m3)':[], 'Toluene (ug/m3)':[], 'Eth-Benzene (ug/m3)':[],
       'MP-Xylene (ug/m3)':[], 'RH (%)':[], 'WS (m/s)':[], 'WD (degree)':[], 'BP (mmHg)':[],
       'Xylene (ug/m3)':[], 'AT (degree C)':[], 'RF (mm)':[]},
      'Powai':{'PM2.5 (ug/m3)':[], 'NO (ug/m3)':[], 'NO2 (ug/m3)':[], 'NOx (ppb)':[],
       'NH3 (ug/m3)':[], 'SO2 (ug/m3)':[], 'CO (mg/m3)':[], 'Ozone (ug/m3)':[],
       'Benzene (ug/m3)':[], 'Eth-Benzene (ug/m3)':[], 'MP-Xylene (ug/m3)':[], 'RH (%)':[],
       'WS (m/s)':[], 'WD (degree)':[], 'SR (W/mt2)':[], 'BP (mmHg)':[], 'AT (degree C)':[]},
      'Siddarth Nagar-Worli':{'PM2.5 (ug/m3)':[], 'PM10 (ug/m3)':[], 'NO (ug/m3)':[], 'NO2 (ug/m3)':[],
       'NOx (ppb)':[], 'SO2 (ug/m3)':[], 'NH3 (ug/m3)':[], 'CO (mg/m3)':[],
       'Ozone (ug/m3)':[], 'RH (%)':[], 'WS (m/s)':[], 'WD (degree)':[], 'BP (mmHg)':[],
       'AT (degree C)':[]},
      'Worli':{'PM2.5 (ug/m3)':[], 'PM10 (ug/m3)':[], 'NO (ug/m3)':[], 'NO2 (ug/m3)':[],
       'NOx (ppb)':[], 'NH3 (ug/m3)':[], 'SO2 (ug/m3)':[], 'CO (mg/m3)':[],
       'Ozone (ug/m3)':[], 'Benzene (ug/m3)':[], 'Eth-Benzene (ug/m3)':[],
       'MP-Xylene (ug/m3)':[], 'RH (%)':[], 'WS (m/s)':[], 'WD (degree)':[], 'SR (W/mt2)':[],
       'BP (mmHg)':[], 'AT (degree C)':[]},
      'Sion':{'PM2.5 (ug/m3)':[], 'PM10 (ug/m3)':[], 'NO (ug/m3)':[], 'NO2 (ug/m3)':[],
       'NOx (ppb)':[], 'NH3 (ug/m3)':[], 'SO2 (ug/m3)':[], 'CO (mg/m3)':[],
       'Ozone (ug/m3)':[], 'Benzene (ug/m3)':[], 'Eth-Benzene (ug/m3)':[],
       'MP-Xylene (ug/m3)':[], 'RH (%)':[], 'WS (m/s)':[], 'WD (degree)':[], 'SR (W/mt2)':[],
       'BP (mmHg)':[], 'AT (degree C)':[]},
      'Vasai West':{'PM2.5 (ug/m3)':[], 'NO (ug/m3)':[], 'NO2 (ug/m3)':[], 'NOx (ppb)':[],
       'NH3 (ug/m3)':[], 'SO2 (ug/m3)':[], 'CO (mg/m3)':[], 'Ozone (ug/m3)':[], 'RH (%)':[],
       'WS (m/s)':[], 'WD (degree)':[], 'SR (W/mt2)':[], 'BP (mmHg)':[], 'AT (degree C)':[]},
      'Vile Parle West':{'PM2.5 (ug/m3)':[], 'PM10 (ug/m3)':[], 'NO (ug/m3)':[], 'NO2 (ug/m3)':[],
       'NOx (ppb)':[], 'NH3 (ug/m3)':[], 'SO2 (ug/m3)':[], 'CO (mg/m3)':[],
       'Ozone (ug/m3)':[], 'Benzene (ug/m3)':[], 'Eth-Benzene (ug/m3)':[],
       'MP-Xylene (ug/m3)':[], 'RH (%)':[], 'WS (m/s)':[], 'WD (degree)':[], 'SR (W/mt2)':[],
       'BP (mmHg)':[], 'AT (degree C)':[]}}

err = {'Bandra Kurla':{'PM2.5 (ug/m3)':[],
                          'NO (ug/m3)':[],
                          'NO2 (ug/m3)':[],
                          'NOx (ppb)':[],
                          'NH3 (ug/m3)':[],
                          'SO2 (ug/m3)':[],
                          'CO (mg/m3)':[],
                          'Ozone (ug/m3)':[],
                          'Benzene (ug/m3)':[],
                          'Toluene (ug/m3)':[],
                          'Eth-Benzene (ug/m3)':[],
                          'MP-Xylene (ug/m3)':[], 'RH (%)':[], 'WS (m/s)':[], 'WD (degree)':[]},
      'Bandra':{'PM2.5 (ug/m3)':[], 'NO (ug/m3)':[], 'NO2 (ug/m3)':[], 'NOx (ppb)':[],
       'SO2 (ug/m3)':[], 'CO (mg/m3)':[], 'Ozone (ug/m3)':[], 'Benzene (ug/m3)':[],
       'RH (%)':[], 'WS (m/s)':[], 'WD (deg)':[], 'SR (W/mt2)':[], 'BP (mmHg)':[],
       'Xylene (ug/m3)':[], 'AT (degree C)':[], 'RF (mm)':[]},
      'Borivali East':{'PM2.5 (ug/m3)':[], 'NO (ug/m3)':[], 'NO2 (ug/m3)':[], 'NOx (ppb)':[],
       'NH3 (ug/m3)':[], 'SO2 (ug/m3)':[], 'CO (mg/m3)':[], 'Ozone (ug/m3)':[], 'RH (%)':[],
       'WS (m/s)':[], 'WD (degree)':[], 'BP (mmHg)':[], 'RF (mm)':[]},
      'Andheri East':{'PM2.5 (ug/m3)':[], 'NO (ug/m3)':[], 'NO2 (ug/m3)':[], 'NOx (ppb)':[],
       'NH3 (ug/m3)':[], 'SO2 (ug/m3)':[], 'CO (mg/m3)':[], 'Ozone (ug/m3)':[],
       'Benzene (ug/m3)':[], 'Toluene (ug/m3)':[], 'Eth-Benzene (ug/m3)':[],
       'MP-Xylene (ug/m3)':[], 'RH (%)':[], 'WS (m/s)':[], 'WD (degree)':[], 'BP (mmHg)':[],
       'Xylene (ug/m3)':[]},
      'Airport':{'PM2.5 (ug/m3)':[], 'NO (ug/m3)':[], 'NO2 (ug/m3)':[], 'NOx (ppb)':[],
       'NH3 (ug/m3)':[], 'SO2 (ug/m3)':[], 'CO (mg/m3)':[], 'Ozone (ug/m3)':[],
       'Benzene (ug/m3)':[], 'Eth-Benzene (ug/m3)':[], 'MP-Xylene (ug/m3)':[], 'RH (%)':[],
       'WS (m/s)':[], 'WD (degree)':[], 'SR (W/mt2)':[], 'BP (mmHg)':[], 'AT (degree C)':[]},
      'Colaba':{'PM2.5 (ug/m3)':[], 'NO (ug/m3)':[], 'NO2 (ug/m3)':[], 'NOx (ppb)':[],
       'NH3 (ug/m3)':[], 'SO2 (ug/m3)':[], 'CO (mg/m3)':[], 'Ozone (ug/m3)':[],
       'Benzene (ug/m3)':[], 'Eth-Benzene (ug/m3)':[], 'MP-Xylene (ug/m3)':[], 'RH (%)':[],
       'WS (m/s)':[], 'WD (degree)':[], 'SR (W/mt2)':[], 'BP (mmHg)':[], 'AT (degree C)':[]},
      'Deonar':{'PM2.5 (ug/m3)':[], 'NO (ug/m3)':[], 'NO2 (ug/m3)':[], 'NOx (ppb)':[],
       'NH3 (ug/m3)':[], 'SO2 (ug/m3)':[], 'CO (mg/m3)':[], 'Ozone (ug/m3)':[],
       'Benzene (ug/m3)':[], 'Toluene (ug/m3)':[], 'Eth-Benzene (ug/m3)':[],
       'MP-Xylene (ug/m3)':[], 'RH (%)':[], 'WS (m/s)':[], 'WD (degree)':[], 'BP (mmHg)':[],
       'Xylene (ug/m3)':[], 'AT (degree C)':[], 'RF (mm)':[]},
      'Khandavali East':{'PM2.5 (ug/m3)':[], 'NO (ug/m3)':[], 'NO2 (ug/m3)':[], 'NOx (ppb)':[],
       'NH3 (ug/m3)':[], 'SO2 (ug/m3)':[], 'CO (mg/m3)':[], 'Ozone (ug/m3)':[],
       'Benzene (ug/m3)':[], 'Eth-Benzene (ug/m3)':[], 'MP-Xylene (ug/m3)':[], 'RH (%)':[],
       'WS (m/s)':[], 'WD (degree)':[], 'SR (W/mt2)':[], 'BP (mmHg)':[], 'AT (degree C)':[]},
      'Khindipada-Bhandup West':{'PM2.5 (ug/m3)':[], 'NO (ug/m3)':[], 'NO2 (ug/m3)':[], 'NOx (ppb)':[],
       'NH3 (ug/m3)':[], 'SO2 (ug/m3)':[], 'CO (mg/m3)':[], 'Ozone (ug/m3)':[],
       'Benzene (ug/m3)':[], 'Toluene (ug/m3)':[], 'MP-Xylene (ug/m3)':[], 'RH (%)':[],
       'WS (m/s)':[], 'WD (degree)':[], 'BP (mmHg)':[], 'Xylene (ug/m3)':[], 'RF (mm)':[]},
      'Kurla':{'PM2.5 (ug/m3)':[], 'NO (ug/m3)':[], 'NO2 (ug/m3)':[], 'NOx (ppb)':[],
       'NH3 (ug/m3)':[], 'SO2 (ug/m3)':[], 'CO (mg/m3)':[], 'Ozone (ug/m3)':[],
       'Benzene (ug/m3)':[], 'Eth-Benzene (ug/m3)':[], 'MP-Xylene (ug/m3)':[], 'RH (%)':[],
       'WS (m/s)':[], 'WD (degree)':[], 'SR (W/mt2)':[], 'BP (mmHg)':[], 'AT (degree C)':[]},
      'Malad West':{'PM2.5 (ug/m3)':[], 'NO (ug/m3)':[], 'NO2 (ug/m3)':[], 'NOx (ppb)':[],
       'NH3 (ug/m3)':[], 'SO2 (ug/m3)':[], 'CO (mg/m3)':[], 'Ozone (ug/m3)':[],
       'Benzene (ug/m3)':[], 'Eth-Benzene (ug/m3)':[], 'MP-Xylene (ug/m3)':[], 'RH (%)':[],
       'WS (m/s)':[], 'WD (degree)':[], 'BP (mmHg)':[], 'AT (degree C)':[], 'RF (mm)':[]},
       'Mulund West':{'PM2.5 (ug/m3)':[], 'NO (ug/m3)':[], 'NO2 (ug/m3)':[], 'NOx (ppb)':[],
       'NH3 (ug/m3)':[], 'SO2 (ug/m3)':[], 'CO (mg/m3)':[], 'Ozone (ug/m3)':[],
       'Benzene (ug/m3)':[], 'Eth-Benzene (ug/m3)':[], 'MP-Xylene (ug/m3)':[], 'RH (%)':[],
       'WS (m/s)':[], 'WD (degree)':[], 'SR (W/mt2)':[], 'BP (mmHg)':[], 'AT (degree C)':[]},
      'Navy Nagar-Colaba':{'PM2.5 (ug/m3)':[], 'NO (ug/m3)':[], 'NO2 (ug/m3)':[], 'NOx (ppb)':[],
       'NH3 (ug/m3)':[], 'SO2 (ug/m3)':[], 'CO (mg/m3)':[], 'Ozone (ug/m3)':[],
       'Benzene (ug/m3)':[], 'Toluene (ug/m3)':[], 'Eth-Benzene (ug/m3)':[],
       'MP-Xylene (ug/m3)':[], 'RH (%)':[], 'WS (m/s)':[], 'WD (degree)':[], 'BP (mmHg)':[],
       'Xylene (ug/m3)':[], 'AT (degree C)':[], 'RF (mm)':[]},
      'Powai':{'PM2.5 (ug/m3)':[], 'NO (ug/m3)':[], 'NO2 (ug/m3)':[], 'NOx (ppb)':[],
       'NH3 (ug/m3)':[], 'SO2 (ug/m3)':[], 'CO (mg/m3)':[], 'Ozone (ug/m3)':[],
       'Benzene (ug/m3)':[], 'Eth-Benzene (ug/m3)':[], 'MP-Xylene (ug/m3)':[], 'RH (%)':[],
       'WS (m/s)':[], 'WD (degree)':[], 'SR (W/mt2)':[], 'BP (mmHg)':[], 'AT (degree C)':[]},
      'Siddarth Nagar-Worli':{'PM2.5 (ug/m3)':[], 'PM10 (ug/m3)':[], 'NO (ug/m3)':[], 'NO2 (ug/m3)':[],
       'NOx (ppb)':[], 'SO2 (ug/m3)':[], 'NH3 (ug/m3)':[], 'CO (mg/m3)':[],
       'Ozone (ug/m3)':[], 'RH (%)':[], 'WS (m/s)':[], 'WD (degree)':[], 'BP (mmHg)':[],
       'AT (degree C)':[]},
      'Worli':{'PM2.5 (ug/m3)':[], 'PM10 (ug/m3)':[], 'NO (ug/m3)':[], 'NO2 (ug/m3)':[],
       'NOx (ppb)':[], 'NH3 (ug/m3)':[], 'SO2 (ug/m3)':[], 'CO (mg/m3)':[],
       'Ozone (ug/m3)':[], 'Benzene (ug/m3)':[], 'Eth-Benzene (ug/m3)':[],
       'MP-Xylene (ug/m3)':[], 'RH (%)':[], 'WS (m/s)':[], 'WD (degree)':[], 'SR (W/mt2)':[],
       'BP (mmHg)':[], 'AT (degree C)':[]},
      'Sion':{'PM2.5 (ug/m3)':[], 'PM10 (ug/m3)':[], 'NO (ug/m3)':[], 'NO2 (ug/m3)':[],
       'NOx (ppb)':[], 'NH3 (ug/m3)':[], 'SO2 (ug/m3)':[], 'CO (mg/m3)':[],
       'Ozone (ug/m3)':[], 'Benzene (ug/m3)':[], 'Eth-Benzene (ug/m3)':[],
       'MP-Xylene (ug/m3)':[], 'RH (%)':[], 'WS (m/s)':[], 'WD (degree)':[], 'SR (W/mt2)':[],
       'BP (mmHg)':[], 'AT (degree C)':[]},
      'Vasai West':{'PM2.5 (ug/m3)':[], 'NO (ug/m3)':[], 'NO2 (ug/m3)':[], 'NOx (ppb)':[],
       'NH3 (ug/m3)':[], 'SO2 (ug/m3)':[], 'CO (mg/m3)':[], 'Ozone (ug/m3)':[], 'RH (%)':[],
       'WS (m/s)':[], 'WD (degree)':[], 'SR (W/mt2)':[], 'BP (mmHg)':[], 'AT (degree C)':[]},
      'Vile Parle West':{'PM2.5 (ug/m3)':[], 'PM10 (ug/m3)':[], 'NO (ug/m3)':[], 'NO2 (ug/m3)':[],
       'NOx (ppb)':[], 'NH3 (ug/m3)':[], 'SO2 (ug/m3)':[], 'CO (mg/m3)':[],
       'Ozone (ug/m3)':[], 'Benzene (ug/m3)':[], 'Eth-Benzene (ug/m3)':[],
       'MP-Xylene (ug/m3)':[], 'RH (%)':[], 'WS (m/s)':[], 'WD (degree)':[], 'SR (W/mt2)':[],
       'BP (mmHg)':[], 'AT (degree C)':[]}}
pm2_ = list()
co_ = list()
so2_ = list()
o3_ = list()
no_ = list()

#Calculating Sub-Indexes of the above the factors
for i in areas:
    pm2_ = list()
    co_ = list()
    so2_ = list()
    o3_ = list()
    no_ = list()
    for j in ['PM2.5 (ug/m3)', 'NOx (ppb)', 'SO2 (ug/m3)', 'CO (mg/m3)', 'Ozone (ug/m3)']:
        for k in range(len(df[i][j])):
            if j == 'PM2.5 (ug/m3)':
                pm2_.append(get_PM25_subindex(df[i][j][k]))
            elif j == 'NOx (ppb)':
                no_.append(get_NOx_subindex(df[i][j][k]))
            elif j == 'SO2 (ug/m3)':
                so2_.append(get_SO2_subindex(df[i][j][k]))
            elif j == 'CO (mg/m3)':
                co_.append(get_CO_subindex(df[i][j][k]))
            elif j == 'Ozone (ug/m3)':
                o3_.append(get_O3_subindex(df[i][j][k]))
            pm2[i] = pm2_
            no[i] = no_
            o3[i] = o3_
            co[i] = co_
            so2[i] = so2_


aqi_predict = {'Bandra Kurla':[],
              'Bandra':[],
              'Borivali East':[],
              'Andheri East':[],
              'Airport':[],
              'Colaba':[],
              'Deonar':[],
              'Khandavali East':[],
              'Khindipada-Bhandup West':[],
              'Kurla':[],
              'Malad West':[],
               'Mulund West':[],
              'Navy Nagar-Colaba':[],
              'Powai':[],
              'Siddarth Nagar-Worli':[],
              'Worli':[],
              'Sion':[],
              'Vasai West':[],
              'Vile Parle West':[]}


#Calculating AQI from the sub-indexes
for i in areas:
    for j in range(len(pm2['Bandra'])):
                   aqi_predict[i] += [max([pm2[i][j], no[i][j], o3[i][j], co[i][j], so2[i][j]])]

for i in areas:
    aqi_predict[i] = np.array(aqi_predict[i])
    o3[i] = np.array(o3[i])
    co[i] = np.array(co[i])
    so2[i] = np.array(so2[i])
    no[i] = np.array(no[i])
    pm2[i] = np.array(pm2[i])

#Plotting the values of the subindexes and their respective hypothesis functions
no_predict = predict_visualization(np.array(x), no['Bandra Kurla'])
plt_y = no['Bandra Kurla']
plt.figure(figsize=(20, 10))
plt.plot(x, plt_y)
plt.xlabel('days')
plt.ylabel(f'')
plt.plot(x, no_predict)

for i in areas:
    for j in range(len(so2['Bandra'])):
         aqi[i] += [aqi_calc(so2[i][j],co[i][j],no[i][j],pm2[i][j],o3[i][j])]

for i in areas:
    df[i].dropna(axis=1, how='any', inplace=True)



for i in areas:
    X[i] = df[i]
    X[i] = X[i].iloc[:,3:]

ext = np.empty(236, dtype=object)
error = np.empty(236, dtype=object)
for i in range(0,236):
    ext[i] = predict_aqi(X['Bandra'].to_numpy(),aqi['Bandra'],pd.DataFrame(X['Bandra'].iloc[i,:]).T.to_numpy())
    error[i] = aqi['Bandra'][i] - ext[i]
plt.figure(figsize=(20,10))
plt.plot(x, ext, label = 'Predicted Value')
plt.plot(x, aqi['Bandra'], label = 'Actual Value')
plt.legend()


#Measuring the change in the AQI with respective changes in the individual factors
for k in areas:
    for i in range(0,236):
        for j in range(len(X[k].columns)):
            edit = X[k].iloc[i,:]
            edit[j] = edit[j]/2
            pred = predict_aqi(X[k].to_numpy(),aqi[k],pd.DataFrame(edit).T.to_numpy())
            change[k][list(change[k].keys())[j]].append(ext[i] - pred)
            aqi_new[k][list(aqi_new[k].keys())[j]].append(pred)
    X[k] *= 2
print(change)


for k in areas:
    for j in change[k].keys():
        err[k][j] = abs(sum(change[k][j]))
    print(err[k])


#Listing the most dominantly affecting factors
err_ = err
for k in areas:
    for i in range(5):
        boo = max(err_[k], key=err_[k].get)
        factors[k].append(boo)
        err_[k].pop(boo)
print(factors)


#Plotting change in AQI with respective 50% reduction of individual factors
for k in areas:
    for j in aqi_new[k].keys():
        plt.figure(figsize=(15,7.5))
        plt.plot(x,aqi[k],'r')
        plt.plot(x,aqi_new[k][j],'b')
        plt.xlabel('When the '+j+' in '+k+' is reduced by 50%')

#Printing out the result
for k in areas:
    print(k, factors[k])
