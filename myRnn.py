#data preposessing

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the training set

training_set = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = training_set.iloc[:,1:2].values

#feature scaling

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
training_set = sc.fit_transform(training_set)

X_train = training_set[0:1257]
y_train = training_set[1:1258]


#reshaping

X_train = np.reshape(X_train,(1257, 1, 1))

# building the RNN


from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import LSTM

regressor = Sequential()

# adding the LSTM LAYER

regressor.add(LSTM(units = 4, activation = 'sigmoid', input_shape = (None, 1)))

# ADDING THE OUTPUT LAYER

regressor.add(Dense(units = 1))


# compiling the RNN

regressor.compile(optimizer = 'adam',loss = 'mean_squared_error')


# fitting to training set

regressor.fit(X_train,y_train,batch_size = 32,epochs = 200)


# getting values on test set

test_set = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = test_set.iloc[:,1:2].values

inputs = real_stock_price
inputs = sc.fit_transform(inputs)
inputs = np.reshape(inputs,(20, 1, 1))

predicted = regressor.predict(inputs)

predicted = sc.inverse_transform(predicted)

plt.plot(real_stock_price,color = 'red',label = 'Real Google Stock Price')
plt.plot(predicted,color = 'blue',label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Goole Stock Price')
plt.legend()
plt.show()


#evaluating the RNN root mean squared error

import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(real_stock_price, predicted))
