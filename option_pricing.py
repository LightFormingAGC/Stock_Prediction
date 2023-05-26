import math

while True:
    option_info = input(
        "Enter option info (ticker_price, stop_win_price, option_cost): ")
    input_greeks = input("Enter the greeks (delta/gamma/theta/days left):")
    if input_greeks == "" or option_info == "":
        print('\n')
        break
    ticker_price, stop_win_price, option_cost = option_info.split(' ')
    option_cost_stable = option_cost
    delta, gamma, theta, days = input_greeks.split(" ")
    # convert all the inputs to float
    ticker_price = float(ticker_price)
    stop_win_price = float(stop_win_price)
    option_cost = float(option_cost)
    delta = float(delta)
    gamma = float(gamma)
    theta = float(theta)
    days = float(days)
    option_cost_stable = float(option_cost_stable)

    changes = abs(math.floor((float(stop_win_price) - float(ticker_price))))
    for i in range(changes):
        option_cost += delta
        delta += gamma
        gamma *= 0.93
        if abs(stop_win_price - ticker_price) - i - 1 < 1:
            option_cost += delta * (abs(stop_win_price - ticker_price) - i - 1)
    option_cost -= (theta * float(days))
    print("target option price: ", round(option_cost, 2))
    print("1:5 stop loss: ", round(option_cost_stable -
          (option_cost-option_cost_stable) * 0.2, 2))
