import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn import linear_model
import pandas as pd
import requests
import json, datetime, os.path
import datetime
from sklearn.preprocessing import MinMaxScaler,PolynomialFeatures
from sklearn.metrics import mean_absolute_error,mean_squared_error
import numpy as np
import statsmodels.graphics.tsaplots
from pandas.tools.plotting import autocorrelation_plot
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import acf, pacf

coins=['bitcoin','ethereum','litecoin']
a=[]
for coin in coins:
    r = requests.get(f'https://api.coingecko.com/api/v3/coins/{coin}/market_chart?vs_currency=usd&days=max')
    json_file=json.loads(r.text)
    data={}
    time=[]
    for j in range(len(json_file['market_caps'])):
        time.append(json_file['market_caps'][j][0])
    for i in json_file.keys():
        data[i]=[]
        for j in range(len(json_file['market_caps'])):
            data[i].append(json_file[i][j][1])
    crypto=pd.DataFrame(data)
    times=pd.Series(time,name='time')
    crypto['time']=times
    crypto['coin']=coin
    a.append(crypto)
crypto=pd.concat(a)

def convert_to_time(a):
    return pd.to_datetime(datetime.datetime.fromtimestamp(a/1000).strftime('%Y-%m-%d'))
#Date is number of milli seconds since first computer clock. 
#To use strftime in the above format, no. has to be divided by 1000 to give seconds    

bitcoin=crypto.loc[crypto['coin']=='bitcoin']
bitcoin.columns=['btc_mcap','btc_price','btc_volume','time','btc']
bitcoin=bitcoin.reset_index()
bitcoin['date']=bitcoin['time'].apply(convert_to_time)

#bitcoin.loc[max(bitcoin.index)+pd.DateOffset(1),:]=None
bitcoin['day']=bitcoin['date'].dt.weekday_name
bitcoin['1d_change']=(bitcoin['btc_price']-bitcoin['btc_price'].shift(1))
bitcoin['3d_change']=(bitcoin['btc_price']-bitcoin['btc_price'].shift(3))
bitcoin['1w_change']=(bitcoin['btc_price']-bitcoin['btc_price'].shift(7))
bitcoin['2w_change']=(bitcoin['btc_price']-bitcoin['btc_price'].shift(14))
bitcoin['1m_change']=(bitcoin['btc_price']-bitcoin['btc_price'].shift(30))
bitcoin['3m_change']=(bitcoin['btc_price']-bitcoin['btc_price'].shift(90))
bitcoin['6m_change']=(bitcoin['btc_price']-bitcoin['btc_price'].shift(180))
bitcoin['1y_change']=(bitcoin['btc_price']-bitcoin['btc_price'].shift(365))

bitcoin['gain'] = bitcoin.loc[bitcoin['1d_change']>0]['1d_change']
bitcoin['loss'] = abs(bitcoin.loc[bitcoin['1d_change']<0]['1d_change'])
bitcoin['gain']=bitcoin['gain'].fillna(0)
bitcoin['loss'] =bitcoin['loss'].fillna(0)

bitcoin['log_price']=np.log(bitcoin['btc_price'])
bitcoin.set_index(bitcoin['date'],inplace=True)
bitcoin = bitcoin[~bitcoin.index.duplicated(keep='first')]
ts=np.log(bitcoin['btc_price'])

print('p-value of augmented Dickey-Fuller test for price: ',adfuller(bitcoin['btc_price'])[1])
print('p-value of augmented Dickey-Fuller test for natural logarithm of price: ',adfuller(ts)[1])
print('p-value of augmented Dickey-Fuller test for natural logarithm after differencing level 1: ',adfuller((ts-ts.shift()).dropna())[1])
#ADFuller test shows the log of the price after 1 level differencing is stationary
#Therefore, the ARIMA model will be fitted to the log of the price with d=1 & compared to logdiff

train=bitcoin.loc['2013-10-26':'2018-05-01']
test=bitcoin.loc['2018-05-02':]


logdiff=ts-ts.shift().dropna()

#plot autocorrelation and partial autocorrelation of the time series to be used as a reference for the ARIMA model

statsmodels.graphics.tsaplots.plot_acf(logdiff.dropna())
plt.title('Autocorrelation')
plt.xlabel('Lag')
plt.ylabel('acf')

statsmodels.graphics.tsaplots.plot_pacf(logdiff.dropna())
plt.title('Partial autocorrelation')
plt.xlabel('Lag')
plt.ylabel('pacf')

#another way to plot autocorrelation in time series directly with pandas:
plt.figure()
autocorrelation_plot(logdiff.dropna())

#ACF and PACF plots show a single positive spike at lag 1, therefore a model with q=0 will be used. p parameter is adjusted by 
#finding good compromise between error and computation time to be 10

model = ARIMA(ts,order=(2,1,0))
fit = model.fit(disp=0)
print('RSS: %.4f'% ((fit.fittedvalues-logdiff)**2).sum())
plt.figure()
plt.plot(logdiff,color='blue')
plt.plot(fit.fittedvalues,color='red')
plt.xlabel('date')
plt.ylabel('differenced log price')
plt.title('model (10,1,0) \n RSS: %.4f'% ((logdiff-fit.fittedvalues)**2).sum())

size = int(len(ts)-120)
# Divide into train and test
train_arima, test_arima = ts[0:size], ts[size:len(ts)]
history = [x for x in train_arima]
predictions = list()
originals = list()
error_list = list()
residual_error=list()
print('Printing Predicted vs Expected Values...')
print('\n')
# We go over each value in the test set and then apply ARIMA model and calculate the predicted value. We have the expected value in the test set therefore we calculate the error between predicted and expected value 
for t in range(len(test_arima)):
    model = ARIMA(history, order=(2, 1, 0))
    model_fit = model.fit(disp=-1)
    
    output = model_fit.forecast()
    
    pred_value = output[0]
    
        
    original_value = test_arima[t]
    history.append(original_value)
    
    pred_value = np.exp(pred_value)
    
    
    original_value = np.exp(original_value)
    
    # Calculating the error
    error = ((abs(pred_value - original_value)) / original_value) * 100
    error_list.append(error)
    print('predicted = %f,   expected = %f,   error = %f ' % (pred_value, original_value, error), '%')
    residual_error.append(pred_value-original_value)
    predictions.append(float(pred_value))
    originals.append(float(original_value))
    
# After iterating over whole test set the overall mean error is calculated.   
print('\n Mean Error in Predicting Test Case Articles : %f ' % (sum(error_list)/float(len(error_list))), '%')
plt.figure()
test_day = [t for t in range(len(test_arima))]
labels={'Orginal','Predicted'}
plt.plot(test_day, predictions, color= 'green')
plt.plot(test_day, originals, color = 'orange')
plt.title('Expected Vs Predicted Views Forecasting')
plt.xlabel('Day')
plt.ylabel('Closing Price')
plt.legend(labels)
