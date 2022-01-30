import main
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import sklearn.preprocessing
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

data = main.pull_data('TSLA').iloc[:, -2:-1].values

data = data[np.logical_not(np.isnan(data))]
data = np.expand_dims(data, axis=1)
Mms = MinMaxScaler(feature_range=(0, 1))
scaled_data = Mms.fit_transform(data)


x_train = []
y_train = []
for i in range(10, len(scaled_data)-2):
    x_train.append(scaled_data[i - 10:i, 0])
    y_train.append([scaled_data[i, 0]])
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

x_test = []
y_test = []
for i in range(len(scaled_data)-2, len(scaled_data)):
    x_test.append(scaled_data[i - 10:i, 0])
    y_test.append([scaled_data[i, 0]])
x_test, y_test = np.array(x_test), np.array(y_test)

x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

RNN = Sequential()
RNN.add(LSTM(units=20, return_sequences=True, input_shape=(x_train.shape[1], 1)))
RNN.add(Dropout(0.2))
RNN.add(LSTM(units=20, return_sequences=True))
RNN.add(Dropout(0.2))
RNN.add(LSTM(units=20, return_sequences=True))
RNN.add(Dropout(0.2))
RNN.add(LSTM(units=20))
RNN.add(Dropout(0.2))
RNN.add(Dense(units=1))
RNN.compile(optimizer='adam', loss='mean_squared_error')

history = RNN.fit(x_train, y_train, epochs=100, batch_size=3, validation_data=(x_test, y_test))


predicted_data = Mms.inverse_transform(RNN.predict(x_test))
y_test = Mms.inverse_transform(y_test)

fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.plot(y_test, color='red', label='Real Stock Price')
ax1.plot(predicted_data, color='blue', label='Predicted  Stock Price')
ax1.set_ylim(300, 1500)
ax1.set_title('Stock Price Prediction')
ax1.set_xlabel('Time')
ax1.set_ylabel('Stock Price')
ax1.legend()

ax2.plot(history.history['loss'], color='yellow', label='Train_loss')
ax2.plot(history.history['val_loss'], color='purple', label='Test_loss')
ax2.set_title('Loss of Model')
ax2.set_xlabel('Time')
ax2.set_ylabel('Loss')
ax2.legend()

fig2, ax3 = plt.subplots(1, 1)
predict_x_train = RNN.predict(x_train)

predicted_train_data = Mms.inverse_transform(RNN.predict(x_train))
y_train = Mms.inverse_transform(y_train)

ax3.plot(y_train, color='red', label='Real Stock Price')
ax3.plot(predicted_train_data, color='blue', label='Predicted Stock Price')
ax3.set_ylim(0, 1500)
ax3.set_title('Stock Price Prediction')
ax3.set_xlabel('Time')
ax3.set_ylabel('Stock Price')
ax3.legend()

plt.show()

