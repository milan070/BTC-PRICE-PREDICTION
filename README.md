# BTC-PRICE-PREDICTION
# By MILAN MAURYA
import yfinance as yf

df = yf.download('BTC-USD')

df

import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import math
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error

plt.plot(df.index, df ['Adj Close'])
plt.show()

#Train test split

#to_row =
#int(len(df)*0.9)
to_row= int(len(df)*0.9)

training_data = list(df[0:to_row]['Adj Close'])
training_data
testing_data = list(df[to_row:]['Adj Close'])
testing_data

#split data into train and test data
plt.figure(figsize=(10,6))
plt.grid(True)
plt.xlabel('Dates')
plt.ylabel('Cloasing Prices')
plt.plot(df[0:to_row]['Adj Close'],'green', label='Train data')
plt.plot(df[to_row:]['Adj Close'],'blue', label='Test data')
plt.legend()

model_predictions =[]
n_test_obser =len(testing_data)
print(n_test_obser)

from statsmodels.tsa.arima.model import ARIMA

for i in range(n_test_obser):
    model = ARIMA(training_data, order=(4, 1, 0))
    model_fit = model.fit()
    output = model_fit.forecast()
    #print(output)
    yhat=output[0]
    model_predictions.append(yhat)
    actual_test_value = testing_data[i]
    training_data.append(actual_test_value)
   # break

print(model_fit.summary())

len(model_predictions)

plt.figure(figsize=(15,9))
plt.grid(True)
date_range=df[to_row:].index
plt.plot(date_range, model_predictions, color='blue', marker='o', linestyle='dashed', label ='BTC predicted value')
plt.plot(date_range, testing_data,color='red', label ='BTC actual value')
plt.title('Bitcoin Price Predition')
plt.xlabel('Dates')
plt.ylabel('Price')
plt.legend()
plt.show()

#report performance
mape=np.mean(np.abs(np.array(model_predictions)
-np.array(testing_data))/np.abs(testing_data))
print('MAPE:'+str(mape))
