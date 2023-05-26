import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
import web_scraping as ws
import pandas as pd
import numpy as np
from pandas_datareader import data as wb
import matplotlib.pyplot as plt


def data_generator(ticker, direction):

    data = pd.DataFrame()
    data[['Open', 'Close', 'Low', 'High']] = wb.DataReader(
        ticker, data_source='yahoo', start='2013-1-1')[['Open', 'Close', 'Low', 'High']]

    # turn daily data into weekly data, open is the first day's open, close is the last day's close
    data = data.resample('W').agg(
        {'Open': 'first', 'Close': 'last', 'Low': 'min', 'High': 'max'})

    # find ema21
    data['ema21'] = data['Close'].ewm(span=21, adjust=False).mean()
    data['ema21_delta'] = data['ema21'].diff()

    # std
    std = data['ema21_delta'].std()

    data['sup_res'] = 0

    for i in range(2, len(data)-2):

        bot_price_t0 = min(data.iloc[i, 0], data.iloc[i, 1]) + std * 0.5
        top_price_t0 = max(data.iloc[i, 0], data.iloc[i, 1]) - std * 0.5

        bot_price_t_02 = min(data.iloc[i-2, 0], data.iloc[i-2, 1])
        top_price_t_02 = max(data.iloc[i-2, 0], data.iloc[i-2, 1])

        bot_price_t_01 = min(data.iloc[i-1, 0], data.iloc[i-1, 1])
        top_price_t_01 = max(data.iloc[i-1, 0], data.iloc[i-1, 1])

        bot_price_t_11 = min(data.iloc[i+1, 0], data.iloc[i+1, 1])
        top_price_t_11 = max(data.iloc[i+1, 0], data.iloc[i+1, 1])

        bot_price_t_12 = min(data.iloc[i+2, 0], data.iloc[i+2, 1])
        top_price_t_12 = max(data.iloc[i+2, 0], data.iloc[i+2, 1])

        if bot_price_t0 < bot_price_t_02 and bot_price_t0 < bot_price_t_01 and bot_price_t0 < bot_price_t_11 and bot_price_t0 < bot_price_t_12:
            data.iloc[i, 6] = bot_price_t0
        if top_price_t0 > top_price_t_02 and top_price_t0 > top_price_t_01 and top_price_t0 > top_price_t_11 and top_price_t0 > top_price_t_12:
            data.iloc[i, 6] = top_price_t0

    # for each day add the closeset sup_res from the past 10 weeks
    data['adj_sup'] = 0
    data['adj_res'] = 0

    for i in range(30, len(data)):
        past_10 = data.iloc[i-30:i, 6]
        dev = (past_10 - data.iloc[i, 1])
        lower = dev[dev < 0]
        upper = dev[dev > 0]
        if len(lower) > 0:
            data.iloc[i, 7] = data.iloc[i, 1] + max(lower)
        if len(upper) > 0:
            data.iloc[i, 8] = data.iloc[i, 1] + min(upper)

    if direction == 'long':
        data['Low'] = data.Low.shift(-1)
        input = data[['Close', 'Open', 'High',
                      'ema21_delta', 'adj_sup', 'adj_res']].iloc[-1, :]
        train_x = data[['Close', 'Open', 'High',
                        'ema21_delta', 'adj_sup', 'adj_res']].iloc[:-1, :]
        train_y = data[['Low']].iloc[:-1, :]
    else:
        data['High'] = data.High.shift(-1)
        input = data[['Close', 'Open', 'Low',
                      'ema21_delta', 'adj_sup', 'adj_res']].iloc[-1, :]
        train_x = data[['Close', 'Open', 'Low',
                        'ema21_delta', 'adj_sup', 'adj_res']].iloc[:-1, :]
        train_y = data[['High']].iloc[:-1, :]

    train_x, train_y = train_x.dropna(), train_y.dropna().iloc[1:]

    return train_x.values, train_y.values, input.values


ticker, direction = input(
    "Enter the ticker and direction(long/short):").split(' ')
x, y, input = data_generator(ticker, direction)

# 80/20 split
train_x, train_y = x[:int(len(x)*0.8)], y[:int(len(y)*0.8)]
test_x, test_y = x[int(len(x)*0.8):], y[int(len(y)*0.8):]


# fit a basic linear regression model
model = LinearRegression()
model.fit(train_x, train_y)

# MSE
MSE = np.mean((model.predict(test_x) - test_y)**2)
error = model.predict(test_x) - test_y

fig, ax = plt.subplots(2, figsize=(10, 10))

if direction == 'long':
    fig.suptitle(
        f'Linear Regression Model with MSE: {MSE:.2f} \n Tradable: {(error<0).mean():.2f}')
else:
    fig.suptitle(
        f'Linear Regression Model with MSE: {MSE:.2f} \n Tradable: {(error>0).mean():.2f}')

# plot the test error
ax[0].plot(model.predict(test_x), label='prediction')
ax[0].plot(test_y, label='actual')
ax[0].legend()

# plot the distribution of the error
ax[1].hist(model.predict(test_x) - test_y, bins=20)
plt.show()

print(f'Predicted Price: {model.predict([input])[0][0]:.2f}')
# create an multi-layer perceptron model with pytorch

# model = nn.Sequential(
#     nn.Linear(6, 100),
#     nn.ReLU(),
#     nn.Linear(100, 100),
#     nn.ReLU(),
#     nn.Linear(100, 1)
# )

# # define the loss function and optimizer
# loss_fn = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

# # convert the data to tensors
# train_x, train_y = torch.tensor(train_x, dtype=torch.float), torch.tensor(
#     train_y, dtype=torch.float)
# test_x, test_y = torch.tensor(test_x, dtype=torch.float), torch.tensor(
#     test_y, dtype=torch.float)

# # train the model
# for t in range(1000):
#     # forward pass
#     y_pred = model(train_x)

#     # compute and print loss
#     loss = loss_fn(y_pred, train_y)
#     if t % 100 == 99:
#         print(t, loss.item())

#     # zero the gradients before running the backward pass
#     optimizer.zero_grad()

#     # backward pass
#     loss.backward()

#     # update the weights
#     optimizer.step()

# # plot the test error
# fig, ax = plt.subplots(2, figsize=(10, 10))
# fig.suptitle(f'Neural Network Model with MSE: {loss.item():.2f}')

# # plot the test error
# ax[0].plot(model(test_x).detach().numpy(), label='prediction')
# ax[0].plot(test_y.detach().numpy(), label='actual')
# ax[0].legend()

# # plot the distribution of the error
# ax[1].hist(model(test_x).detach().numpy() -
#            test_y.detach().numpy(), bins=20)
# plt.show()
