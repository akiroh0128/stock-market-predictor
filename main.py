import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
import tensorflow as tf 
from tensorflow import keras 
import seaborn as sns 
import os 
from datetime import datetime 
  
#import warnings 
#warnings.filterwarnings("ignore") 

dataset = pd.read_csv('Microsoft_Stock.csv') 

dataset['Date'] = pd.to_datetime(dataset['Date'])
dataset['Date']= dataset['Date'].dt.date

plt.figure(figsize=(14, 6))
plt.plot(dataset['Date'], dataset['Close'], c="r", label="Close", ) 
plt.plot(dataset['Date'], dataset['Open'], c="g", label="Open", ) 
plt.title("Microsoft") 
plt.legend() 
plt.tight_layout()

plt.figure(figsize=(14, 6))
plt.plot(dataset['Date'], dataset['Volume'], c="r", label="Close", ) 
plt.title("Microsoft") 
plt.legend() 
plt.tight_layout()

close_data = dataset['Close'] 
close_data = close_data.values.reshape(-1, 1) 
training_data_index = int(np.ceil(len(close_data) * 0.7))
training_data = close_data[:training_data_index]
print(training_data)

from sklearn.preprocessing import MinMaxScaler 
  
scaler = MinMaxScaler(feature_range=(0, 1)) 
scaled_data = scaler.fit_transform(close_data) 
  
train_data = scaled_data[0:int(training_data_index), :] 
# prepare feature and labels 
x_train = [] 
y_train = [] 
  
for i in range(60, len(train_data)): 
    x_train.append(train_data[i-60:i, 0]) 
    y_train.append(train_data[i, 0]) 
  
x_train, y_train = np.array(x_train), np.array(y_train) 
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1)) 

model = keras.models.Sequential() 
model.add(keras.layers.LSTM(units=50, 
                            return_sequences=True, 
                            input_shape=(x_train.shape[1], 1))) 
model.add(keras.layers.LSTM(units=50)) 
model.add(keras.layers.Dense(32)) 
model.add(keras.layers.Dropout(0.5)) 
model.add(keras.layers.Dense(1)) 
model.summary()

model.compile(optimizer='adam', 
              loss='mean_squared_error') 
history = model.fit(x_train, 
                    y_train, 
                    epochs=10)

test_data = close_data[training_data_index:]
inputs = close_data[len(close_data) - len(test_data) - 60:]
inputs = inputs.reshape(-1,1)
inputs  = scaler.transform(inputs)
x_test = [] 
#y_test = dataset.iloc[training_data:, :] 
for i in range(60,inputs.shape[0]):
        x_test.append(inputs[i-60:i,0])
  
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))
closing_price = model.predict(x_test)
closing_price = scaler.inverse_transform(closing_price)
RMS=np.sqrt(np.mean(np.power((test_data-closing_price),2)))
#y_test = pd.to_numeric(y_test, errors='coerce')
#y_test = y_test.to_numpy()

print("RMS:",RMS)
print("MSE:",RMS**2)
  
train = dataset[:training_data_index] 
test = dataset[training_data_index:] 
test_data_df = pd.DataFrame(test_data, columns=['Close'])
test_data_df['Predictions'] = closing_price
plt.figure(figsize=(10, 8)) 
plt.plot(train['Date'], train['Close']) 
plt.plot(test['Date'], test_data_df[['Close', 'Predictions']]) 
plt.title('Microsoft Stock Close Price') 
plt.xlabel('Date') 
plt.ylabel("Close") 
plt.legend(['Train', 'Test', 'Predictions']) 

