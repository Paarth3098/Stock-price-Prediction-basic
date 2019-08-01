#importing basic libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing train dataset and 
dataset_train = pd.read_csv('../input/Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[ : , 1:2].values

#MinMaxScaler to Transforms features by scaling each feature to a given range
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
train_set_scaled = sc.fit_transform(training_set)

X_train = []
y_train = []

#Making an array of transformed data into parameters and output
for i in range(60,1258):
    X_train.append(train_set_scaled[ i-60:i, 0])
    y_train.append(train_set_scaled[ i, 0])
    
X_train, y_train = np.array(X_train), np.array(y_train)

X_train.shape

#Reshaping the X_train to 3d array
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

#Making the RNN Model
#importing Libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

#Making the RNN layers with dropout
regressor = Sequential()
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1],1)))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.1))
regressor.add(Dense(units = 1))

#Compiling and fitting the model
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])
regressor.fit(X_train, y_train, epochs = 30, batch_size = 32)

#Defining the Real stock prices
dataset_test = pd.read_csv('../input/Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values


dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)

#inputs for predicting will be data of 20 days prior to the predicting day
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60: ].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
inputs.shape

X_test = []
for i in range (60,80):
    X_test.append(inputs[i-60:i, 0])
    
X_test = np.array(X_test)
X_test.shape

#reshaping the X_test into 3d array
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

#predicting the stock price
predicted_stock_price = regressor.predict(X_test)

#Inverse transform of predicted values to compare with test set
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

#ploting the variation of real and predicted 
plt.plot(real_stock_price, color = 'red', label = 'Real Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Stock Price')

#rms error value 
rms = np.sqrt(np.mean(np.power(real_stock_price - predicted_stock_price, 2)))


 
