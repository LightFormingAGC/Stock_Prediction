import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import web_scraping as ws
import os
import math

potential_traded = {}
avg_change_dict = {}

while True:
    asked = input("Enter the ticker and target direction (buy/sell):")
    if asked == "":
        print('\n')
        break
    ticker, direction = asked.split()
    potential_traded[ticker] = direction


for ticker in potential_traded.keys():

    three_consecutive_dis_up = []
    three_consecutive_dis_down = []

    past_days = ws.scrap_data(ticker)
    ordered_past_days = pd.Series(past_days[::-1])
    ordered_past_days = ordered_past_days.apply(lambda x: float(x.replace(',', '')))
    changed_t1 = ordered_past_days.diff()
    move_t1 = changed_t1.apply(lambda x: -1 if x < 0 else 1)

    c = 0
    j = 0
    while j < len(move_t1) - 2:
        if move_t1.values[j] == move_t1.values[j + 1] == move_t1.values[j + 2] == 1:
            three_consecutive_dis_up.append(c)
            c = 0
            j += 3
        elif move_t1.values[j] == move_t1.values[j + 1] == move_t1.values[j + 2] == -1:
            three_consecutive_dis_down.append(c)
            c = 0
            j += 3
        else:
            j += 1
            c += 1

    avg_d_up = np.mean(three_consecutive_dis_up)
    std_d_up = np.std(three_consecutive_dis_up)
    up_buy_days = math.ceil(avg_d_up + 1.3 * std_d_up)

    avg_d_down = np.mean(three_consecutive_dis_down)
    std_d_down = np.std(three_consecutive_dis_down)
    down_buy_days = math.ceil(avg_d_down + 1.3 * std_d_down)

    trade_cond = True
    # check if there are three consecutive days with the same direction

    if potential_traded[ticker] == "buy":
        last_days = move_t1[-up_buy_days:].values
        for i in range(len(last_days) - 2):
            if last_days[i] == last_days[i + 1] == last_days[i + 2] == 1:
                trade_cond = False
                break
    else:
        last_days = move_t1[-down_buy_days:].values
        for i in range(len(last_days) - 2):
            if last_days[i] == last_days[i + 1] == last_days[i + 2] == -1:
                trade_cond = False
                break

    if trade_cond:
        changes_avg_up = []
        changes_avg_down = []
        temp_df = pd.DataFrame({'prices': ordered_past_days[1:], 'move': move_t1[1:]})
        temp_df = temp_df.reset_index(drop=True)
        if potential_traded[ticker] == "buy":
            for j in range(len(temp_df)-2):
                if temp_df.move[j] == temp_df.move[j+1] == temp_df.move[j+2] == 1:
                    changes_avg_up.append(abs((temp_df.prices[j+2]-temp_df.prices[j])/temp_df.prices[j]))
            avg_change = np.median(changes_avg_up)
            print(f"{ticker} has not have three consecutive up days for {up_buy_days} days")
            print(f"it has an median up potential of {round(avg_change * 100, 2)} percent if you buy now")
            print('\n')
            avg_change_dict[ticker] = (ordered_past_days.values[-1], avg_change)
        else:
            for j in range(len(temp_df)-2):
                if temp_df.move[j] == temp_df.move[j+1] == temp_df.move[j+2] == -1:
                    changes_avg_down.append(abs((temp_df.prices[j+2]-temp_df.prices[j])/temp_df.prices[j]))
            avg_change = np.median(changes_avg_down)
            print(f"{ticker} has not have three consecutive down days for {down_buy_days} days")
            print(f"it has an average down potential of {round(avg_change * 100, 2)} percent if you sell now")
            print('\n')
            avg_change_dict[ticker] = (ordered_past_days.values[-1], avg_change)


while True:
    asked = input("Which one did you place order:")
    if asked == "":
        break
    ticker = asked
    if potential_traded[ticker] == "buy":
        print(f"You have bought/called {ticker} at {avg_change_dict[ticker][0]}, stop win is at {round(avg_change_dict[ticker][0]*(1+avg_change_dict[ticker][1]), 2)}")
        print('\n')
    else:
        print(f"You have sold/put {ticker} at {avg_change_dict[ticker][0]}, stop win is at {round(avg_change_dict[ticker][0]*(1-avg_change_dict[ticker][1]), 2)}")
        print('\n')




