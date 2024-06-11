# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 16:16:22 2024

@author: 35796
"""
from scipy.stats import pearsonr
import pandas as pd
import yfinance as yf
import datetime
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
import empyrical as ep
import scipy.stats as stats
from scipy.stats import gaussian_kde

start_sp = datetime.datetime(2013, 12, 15)
end_sp = datetime.datetime.today()
NVIDIA = yf.download('NVDA', start_sp, end_sp)
X_path = r'C:\Users\andre\OneDrive\Desktop\MSc Computational Finance\Algorithmic Trading\EFFR.csv'
EFFR = pd.read_csv(X_path, header=0)
EFFR = EFFR.set_index('Effective Date') 
EFFR = EFFR.dropna()

# Converting index to datetime object
EFFR.index = pd.to_datetime(EFFR.index)

# Sorting by index (oldest to newest date)
EFFR = EFFR.sort_index()
# Prepare the feature data
volume = NVIDIA['Volume'].values
close = NVIDIA['Close'].values
low = NVIDIA['Low'].values
high = NVIDIA['High'].values
_open = NVIDIA['Open'].values
# Convert annual to day rate
EFFR = 1/252 * EFFR
EFFR = EFFR['Rate (%)']




NVIDIA_close = NVIDIA['Close']
NVIDIA_returns = NVIDIA_close / NVIDIA_close.shift(1) - 1
NVIDIA_returns.drop(NVIDIA_returns.index[0], inplace=True)


# Get the indices of EFFR and NVIDIA_returns
effr_indices = set(EFFR.index)
NVIDIA_indices = set(NVIDIA_returns.index)

# Find indices that are in EFFR but not in NVIDIA_returns
indices_only_in_effr = effr_indices - NVIDIA_indices

# Find indices that are in NVIDIA_returns but not in EFFR
indices_only_in_NVIDIA = NVIDIA_indices - effr_indices


# Append rows with indices only in NVIDIA_returns to EFFR with NaN values
for index in indices_only_in_NVIDIA:
    EFFR.loc[index] = np.nan

# Sorting by index (oldest to newest date)
EFFR = EFFR.sort_index()
    
# Replace NaN values in EFFR with previous index values
EFFR.fillna(method='ffill', inplace=True)

## Remove indices_only_in_effr from EFFR
EFFR.drop(indices_only_in_effr, inplace=True)

# Perform subtraction
NVIDIA_excess_returns = NVIDIA_returns - EFFR



def plot_time_series(time_series,save_path):
    
 
    
    
    plt.figure(figsize=(10, 6))


    
    plt.plot(time_series.index, time_series)
    plt.tight_layout()
    plt.title('NVIDIA daily returns', fontsize = 18)
    plt.ylabel("NVIDIA return", fontsize = 16)
    plt.xlabel("Date", fontsize = 16)
    plt.grid()
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    if save_path:
        plt.savefig(save_path, format='pdf')
    plt.show()    
   
    
save_path_1 = r'C:\Users\andre\OneDrive\Desktop\MSc Computational Finance\NVIDIA_returns.pdf'
save_path_2 = r'C:\Users\andre\OneDrive\Desktop\MSc Computational Finance\EFFT rate.pdf'
save_path_3 = r'C:\Users\andre\OneDrive\Desktop\MSc Computational Finance\NVIDIA_excess_return.pdf '  

# Plot the data
plot_time_series(NVIDIA_returns, save_path_1)

plt.figure(figsize=(10, 6))

# Plot the close price for each ETF
 
plt.plot(EFFR.index, EFFR)
plt.tight_layout()
plt.title('EFFR daily rate', fontsize = 18)
plt.ylabel("EFFR daily rate", fontsize = 16)
plt.xlabel("Date", fontsize = 16)
plt.grid()
plt.xticks(fontsize=14) 
plt.yticks(fontsize=14) 
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
if save_path_2:
    plt.savefig(save_path_2, format='pdf')
plt.show()   

plt.figure(figsize=(10, 6))
plt.plot(EFFR.index, NVIDIA_excess_returns)
plt.tight_layout()
plt.title('NVIDIA excess daily return per unit', fontsize = 18)
plt.ylabel("Excess return", fontsize = 16)
plt.xlabel("Date", fontsize = 16)
plt.grid()
plt.xticks(fontsize=14) 
plt.yticks(fontsize=14) 
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
if save_path_3:
    plt.savefig(save_path_3, format='pdf')
plt.show()   



start_sp = datetime.datetime(2013, 12, 31)
end_sp = datetime.datetime(2019,12,31)
NVIDIA = yf.download('NVDA', start_sp, end_sp)
close = NVIDIA['Close']





test_size = 300
close_train = close[:-test_size]
close_test = close[-test_size:]


# def correlation_tests(lookback_values, holddays_values, cl):
#     for lookback in lookback_values:
#         for holddays in holddays_values:
#             ret_lag = (cl - np.roll(cl, lookback)) / np.roll(cl, lookback)
#             ret_fut = (np.roll(cl, -holddays) - cl) / cl
            
#             # Handle missing data
#             nan_indices = np.logical_or(np.isnan(ret_lag), np.isnan(ret_fut))
#             inf_indices = np.logical_or(np.isinf(ret_lag), np.isinf(ret_fut))
#             bad_indices = np.logical_or(nan_indices, inf_indices)
#             ret_lag = ret_lag[~bad_indices]
#             ret_fut = ret_fut[~bad_indices]
            
#             if lookback >= holddays:
#                 indep_set = np.arange(0, len(ret_lag), lookback)
#             else:
#                 indep_set = np.arange(0, len(ret_lag), holddays)
                
#             ret_lag = ret_lag[indep_set]
#             ret_fut = ret_fut[indep_set]
            
#             if len(ret_lag) > 1 and len(ret_fut) > 1:  # Ensure there are enough data points for correlation
#                 cc, pval = pearsonr(ret_lag, ret_fut)
#                 print(f"lookback={lookback} holddays={holddays} cc={cc:.4f} pval={pval:.4f}")
#             else:
#                 print(f"lookback={lookback} holddays={holddays} cc=NaN pval=NaN (insufficient data)")

# # Example usage:
# # Define lookback and holddays values
# lookback_values = [1, 2, 5, 10, 15,  25, 40, 60, 80, 100, 120, 140,160,180,200,250]
# holddays_values = [1, 2, 5, 10, 15,  25, 40, 60, 80, 100, 120, 140,160,180,200,250]



# # Perform correlation tests
# correlation_tests(lookback_values, holddays_values, close_train)


# Implement Strategy


# Return pairs with high correlation and low p-value


start_sp = datetime.datetime(2013, 4, 30)
end_sp = datetime.datetime.today()
NVIDIA = yf.download('NVDA', start_sp, end_sp)
close = NVIDIA['Close']
starting_date = '2014-01-02'
NVIDIA_excess_returns = NVIDIA_excess_returns[NVIDIA_excess_returns.index >= starting_date]



rolling_ewm = close.pct_change().ewm(span=3).mean()
rolling_returns = close.pct_change()
rolling_ewm = rolling_ewm[(rolling_ewm.index >= starting_date)] 
rolling_returns = rolling_returns[(rolling_returns.index >= starting_date)] 

# rolling_returns = np.array(rolling_returns)
# close_train = np.array(close_train)
def momentum_strategy_positions(rolling_ewm, returns):
    rolling_ewm = np.array(rolling_ewm)
    positions = []
    i = 0
    while i < len(returns):
        if rolling_ewm[i] > returns.iloc[i]:  # Sell if todays return is lower than ewm
            positions.append(-1) 
            i += 1
            
        elif rolling_ewm[i] < returns.iloc[i]:  # Buy if todays return is higher than ewm
            positions.append(+1)  # 
            i += 1
            
        else:
            positions.append(0)  # No signal
            i += 1
    return positions

positions = momentum_strategy_positions(rolling_ewm, rolling_returns)
positions = positions[:rolling_ewm.size]
positions = pd.Series(positions, index=rolling_ewm.index)
print("")
print("Long positions: ", sum(1 for x in positions if x > 0))
print("Short positions: ", sum(1 for x in positions if x < 0))
# for the strategy to have a smooth start with kelly criterion, rolling returns need to start at first date


positions = positions[positions.index>=starting_date]
close = close[close.index>=starting_date]

# Merge positions and close prices into a single DataFrame
df = pd.DataFrame({'Close': close, 'Positions': positions})

plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Close'], color='black', lw=2, label='Close Price')

# Plotting buy positions (green triangles)
buy_positions = df[df['Positions'] == 1]
plt.scatter(buy_positions.index, buy_positions['Close'], color='green', marker='^', label='Buy Position')

# Plotting sell positions (red triangles)
sell_positions = df[df['Positions'] == -1]
plt.scatter(sell_positions.index, sell_positions['Close'], color='red', marker='v', label='Sell Position')

plt.title('Momentum Strategy Positions vs Close Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

def compute_win_loss_probability(excess_returns):
    positive_returns = np.sum(np.array(excess_returns) > 0)
    total_trades = len(excess_returns)

    win_probability = positive_returns / total_trades
    loss_probability = 1 - win_probability

    return win_probability, loss_probability

def calculate_payout_ratio(returns):
    # Convert portfolio_excess_returns to a numpy array
    returns = np.array(returns)
    
    # Filter excess returns based on the threshold
    winning_trades = returns[returns > 0]
    losing_trades = returns[returns < 0]

    # Calculate average excess returns for winning and losing trades
    avg_win = np.mean(winning_trades) 
    avg_loss = np.abs(np.mean(losing_trades)) 
    
    
    
        

    return avg_win,avg_loss


def momentum_strategy_theta(positions, initial_capital, L, excess_returns):
    V0 = 200000
    margin = 2000000
    portfolio_value = [V0]
    excess_returns = np.array(excess_returns)
    portfolio_excess_returns = []
    positions = np.array(positions)
    theta = []
    for i in range(len(positions)):
        if not portfolio_excess_returns:
            optimal_fraction = 1
        else:
            pwin, ploss = compute_win_loss_probability(portfolio_excess_returns[-15:])
            avg_win,avg_loss = calculate_payout_ratio(portfolio_excess_returns[-15:])

           
            optimal_fraction = pwin/avg_loss - ploss/avg_win
            if np.isnan(pwin/avg_loss):
                optimal_fraction = 1
            if np.isnan(ploss/avg_win):
                optimal_fraction = -1
            optimal_fraction = np.clip(optimal_fraction,-1/10,1/10)
            
        if positions[i] == 1:
            # Max position in theta is 2 million in both long and short side
            if abs(portfolio_value[-1] * optimal_fraction) > margin:
                theta.append(theta[-1])
            else:
             theta.append(portfolio_value[-1]  * optimal_fraction) # buy position
        elif positions[i] == -1:
            if abs(portfolio_value[-1] * optimal_fraction) > margin:
                theta.append(theta[-1])
            else:
                theta.append(-(portfolio_value[-1]  * optimal_fraction)) # sell position
        else:
            if not theta:
                theta.append(0)
                continue
            theta.append(theta[-1]) # hold position
        
        
        portfolio_excess_returns.append((theta[i] * excess_returns[i])/portfolio_value[-1])
        
        portfolio_value.append(portfolio_value[-1] + (theta[i] * excess_returns[i]))
        
        
        
        
        

    return theta

initial_capital = 200000
L = 10
training_date = '2023-1-1'


theta = momentum_strategy_theta(positions, initial_capital , L, NVIDIA_excess_returns)
theta = pd.Series(theta, index=NVIDIA_excess_returns.index)
theta_train = theta[theta.index <= training_date]
theta_test = theta[theta.index > training_date]

def daily_trading_pnl(theta,excess_daily_returns):
    # Incorporate transaction costs as 0.75% of transaction value
    transaction_cost = np.zeros(len(theta))
    for i in range(len(theta)):
     transaction_cost[i] = np.abs(theta.iloc[i]-theta.iloc[i-1]) * 0.0075
    returns = np.array(excess_daily_returns)
    theta = np.array(theta)
    daily_pnl = []
    for i in range(len(theta)):
        daily_pnl.append(theta[i] * returns[i] - transaction_cost[i])
    return daily_pnl



NVIDIA_excess_returns_train = NVIDIA_excess_returns[NVIDIA_excess_returns.index <= training_date] 
NVIDIA_excess_returns_test =  NVIDIA_excess_returns[NVIDIA_excess_returns.index > training_date] 
daily_pnl_train = daily_trading_pnl(theta_train,NVIDIA_excess_returns_train)
daily_pnl_train = pd.Series(daily_pnl_train, index=NVIDIA_excess_returns_train.index)
daily_pnl_test = daily_trading_pnl(theta_test,NVIDIA_excess_returns_test)
daily_pnl_test = pd.Series(daily_pnl_test, index=NVIDIA_excess_returns_test.index)
cumulated_pnl_dv = pd.concat([daily_pnl_train,daily_pnl_test])
cumulated_pnl_dv = cumulated_pnl_dv.cumsum()
cumulated_pnl_dv_train = cumulated_pnl_dv[cumulated_pnl_dv.index <= training_date ]
cumulated_pnl_dv_test = cumulated_pnl_dv[cumulated_pnl_dv.index > training_date ]

fig, axs = plt.subplots(2, 1, figsize=(10, 12))

# Plot for Daily Excess Profit and Loss
axs[0].plot(NVIDIA_excess_returns_train.index, daily_pnl_train, color='blue', label='In-Sample')
axs[0].plot(NVIDIA_excess_returns_test.index, daily_pnl_test, color='green', label='Out-of-Sample')
axs[0].set_title('Daily Excess Profit and Loss', fontsize=18)
axs[0].set_ylabel("Profit and Loss", fontsize=16)
axs[0].set_xlabel("Date", fontsize=16)
axs[0].grid()
axs[0].legend(fontsize=16)
axs[0].tick_params(labelsize=14)

# Plot for Cumulative Excess Profit and Loss
axs[1].plot(NVIDIA_excess_returns_train.index, cumulated_pnl_dv_train/10**6, color='blue', label='In-Sample')
axs[1].plot(NVIDIA_excess_returns_test.index, cumulated_pnl_dv_test/10**6, color='green', label='Out-of-Sample')
axs[1].set_title('Cumulative Excess Profit and Loss', fontsize=18)
axs[1].set_ylabel("Cumulative Excess Profit and Loss (millions)", fontsize=16)
axs[1].set_xlabel("Date", fontsize=16)
axs[1].grid()
axs[1].legend(fontsize=16)
axs[1].tick_params(labelsize=14)
plt.savefig(r'C:\Users\andre\OneDrive\Desktop\MSc Computational Finance\Momentum_Excess_PNL.pdf', format='pdf')
plt.tight_layout()
plt.show()

# Plot theta positions with bounds:
def theta_positions(theta_train,theta_test,V0,L):
    plt.figure(figsize=(10, 6))
    upper_bound = V0 * L
    lower_bound = -V0 * L
    theta_combined = pd.concat([theta_train, theta_test])
    plt.plot(theta_train.index, theta_train/10**6, color='blue', label = 'In-Sample')
    plt.plot(theta_test.index, theta_test/10**6, color='green', label = 'Out-of-Sample')
    plt.plot(theta_combined.index, np.ones_like(theta_combined) * upper_bound /10**6, color='red', linestyle='--', label='Upper Bound')
    plt.plot(theta_combined.index, np.ones_like(theta_combined) * lower_bound /10**6, color='red', linestyle='--', label='Lower Bound')
    plt.tight_layout()
    plt.title('$\\theta$ positions in dollars against time', fontsize = 18)
    plt.ylabel("$\\theta$ (millions)", fontsize=16)
    plt.xlabel("Date", fontsize = 16)
    plt.grid()
    plt.legend(loc='upper right')
    plt.xticks(fontsize=14) 
    plt.yticks(fontsize=14) 
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.savefig(r'C:\Users\andre\OneDrive\Desktop\MSc Computational Finance\Momentum_Theta.pdf', format='pdf')
    plt.show()   

V0 = 200000
L = 10
theta_positions(theta_train,theta_test,V0,L)

# Calculate turnover in dollars and units
def turnover(theta,close):
    theta_difference = theta.shift(1)-theta
    theta_difference.drop(theta_difference.index[0], inplace=True)
    theta_difference = np.array(theta_difference)
    turnover_dollars = np.sum(np.abs(theta_difference))
    units = theta/close
    units_difference = units.shift(1)-units
    units_difference.drop(units_difference.index[0], inplace=True)
    units_difference = np.array(units_difference)
    turnover_units = np.sum(np.abs(units_difference))
    return turnover_dollars,turnover_units




turnover_dollars,turnover_units = turnover(theta,NVIDIA_close[1:])
print("Turnover_dollars: ", turnover_dollars)
print("Turnover_units: ", turnover_units)

def moving_average_turnover(theta,lag):
    theta_difference = theta.shift(1)-theta
    theta_difference.drop(theta_difference.index[0], inplace=True)
    return np.abs(theta_difference).rolling(window=lag).mean()

# 20 -day Volatility of NVIDIA

lag = 30
rolling_vol = NVIDIA_returns.rolling(window=lag).std()
moving_average_turnover = moving_average_turnover(theta,lag)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

ax1.plot(moving_average_turnover.index, moving_average_turnover, color='black')
ax1.set_title('30-day Moving Average Turnover', fontsize=18)
ax1.set_ylabel("Moving Average Turnover", fontsize=16)
ax1.grid()
ax1.legend(loc='upper right')
ax1.tick_params(axis='x', labelsize=14)
ax1.tick_params(axis='y', labelsize=14)

ax2.plot(rolling_vol.index, rolling_vol, color='red')
ax2.set_title('30-day realized Volatility', fontsize=18)
ax2.set_xlabel("Date", fontsize=16)
ax2.set_ylabel("Volatility", fontsize=16)
ax2.grid()
ax2.tick_params(axis='x', labelsize=14)
ax2.tick_params(axis='y', labelsize=14)
plt.savefig(r'C:\Users\andre\OneDrive\Desktop\MSc Computational Finance\Momentum_Moving_Turnover.pdf', format='pdf')
plt.tight_layout()
plt.show()

def daily_total_pnl(theta,L,V0,daily_excess_return,risk_free):
  theta = np.array(theta)
  portfolio_value = [V0]
  risk_free = np.array(risk_free)
  daily_excess_return = np.array(daily_excess_return)
  daily_total_pnl = np.zeros(len(theta))
  for i in range(len(theta+1)):
      # we update the portfolio dynamically 
      daily_total_pnl[i] = daily_excess_return[i] + ((portfolio_value[-1] - np.abs(theta[i])/L) * risk_free[i])
      portfolio_value.append(portfolio_value[-1] + daily_total_pnl[i])
  return daily_total_pnl

daily_pnl = daily_trading_pnl(theta, NVIDIA_excess_returns)
# daily_total_pnl = daily_total_pnl(theta, L, V0, daily_pnl, EFFR)
# daily_total_pnl = pd.Series(daily_total_pnl, index=NVIDIA_excess_returns.index)
# daily_total_train = daily_total_pnl[daily_total_pnl.index <= training_date]
# daily_total_test = daily_total_pnl[daily_total_pnl.index >training_date]
# daily_total_pnl_cum = daily_total_pnl.cumsum()
# daily_total_cum_train = daily_total_pnl_cum[daily_total_pnl_cum.index <= training_date]
# daily_total_cum_test = daily_total_pnl_cum[daily_total_pnl_cum.index > training_date]
# daily_cap_pnl = daily_total_pnl - daily_pnl
# daily_cap_train = daily_cap_pnl[daily_cap_pnl.index <= training_date]
# daily_cap_test = daily_cap_pnl[daily_cap_pnl.index > training_date]
# daily_cap_pnl_cum = daily_cap_pnl.cumsum()
# daily_cap_cum_train = daily_cap_pnl_cum[daily_cap_pnl_cum.index <= training_date]
# daily_cap_cum_test = daily_cap_pnl_cum[daily_cap_pnl_cum.index > training_date]

# # Plot
# fig, axs = plt.subplots(2, 1, figsize=(10, 12))

# # Plot 1: Daily profit and loss of money-market-account
# axs[0].plot(NVIDIA_excess_returns_train.index, daily_cap_train, color='blue', label='In-Sample')
# axs[0].plot(NVIDIA_excess_returns_test.index, daily_cap_test, color='green', label='Out-of-Sample')
# axs[0].set_title('Daily profit and loss of money-market-account', fontsize=18)
# axs[0].set_ylabel("Profit and Loss", fontsize=16)
# axs[0].set_xlabel("Date", fontsize=16)
# axs[0].grid()
# axs[0].legend(fontsize = 16)
# axs[0].tick_params(labelsize=14)

# # Plot 2: Cumulative profit and loss of money-market-account
# axs[1].plot(NVIDIA_excess_returns_train.index, daily_cap_cum_train/10**6, color='blue', label='In-Sample')
# axs[1].plot(NVIDIA_excess_returns_test.index, daily_cap_cum_test/10**6, color='green', label='Out-of-Sample')
# axs[1].set_title('Cumulative profit and loss of money-market-account', fontsize=18)
# axs[1].set_ylabel("Cumulative Profit and Loss(millions)", fontsize=16)
# axs[1].set_xlabel("Date", fontsize=16)
# axs[1].grid()
# axs[1].legend(fontsize = 16)
# axs[1].tick_params(labelsize=14)
# plt.savefig(r'C:\Users\andre\OneDrive\Desktop\MSc Computational Finance\Momentum_market_PNL.pdf', format='pdf')
# plt.tight_layout()
# plt.show()

# fig, axs = plt.subplots(2, 1, figsize=(10, 12))

# # Plot 1: Daily total profit-and-loss
# axs[0].plot(NVIDIA_excess_returns_train.index, daily_total_train, color='blue', label='In-Sample')
# axs[0].plot(NVIDIA_excess_returns_test.index, daily_total_test, color='green', label='Out-of-Sample')
# axs[0].set_title('Daily total profit-and-loss', fontsize=18)
# axs[0].set_ylabel("Profit and Loss", fontsize=16)
# axs[0].set_xlabel("Date", fontsize=16)
# axs[0].grid()
# axs[0].legend(fontsize = 14)
# axs[0].tick_params(labelsize=14)

# # Plot 2: Cumulative total profit-and-loss
# axs[1].plot(NVIDIA_excess_returns_train.index, daily_total_cum_train/10**6, color='blue', label='In-Sample')
# axs[1].plot(NVIDIA_excess_returns_test.index, daily_total_cum_test/10**6, color='green', label='Out-of-Sample')
# axs[1].set_title('Cumulative total profit-and-loss', fontsize=18)
# axs[1].set_ylabel("Cumulative Profit and Loss (millions)", fontsize=16)
# axs[1].set_xlabel("Date", fontsize=16)
# axs[1].grid()
# axs[1].legend(fontsize = 14)
# axs[1].tick_params(labelsize=14)
# plt.savefig(r'C:\Users\andre\OneDrive\Desktop\MSc Computational Finance\Momentum_Total_PNL.pdf', format='pdf')
# plt.tight_layout()
# plt.show()

portfolio_value = np.zeros(len(theta)+1)
portfolio_value[0] = V0
for i in range(len(theta)):
    portfolio_value[i+1] = portfolio_value[i] + daily_pnl[i]
# Sharpe ratio
portfolio_excess_returns =  portfolio_value[1:]/portfolio_value[:-1] - 1
portfolio_excess_returns = pd.Series(portfolio_excess_returns, index = NVIDIA_excess_returns.index)


port_excess_returns_train = portfolio_excess_returns[portfolio_excess_returns.index <= training_date]
port_excess_returns_test = portfolio_excess_returns[portfolio_excess_returns.index > training_date]
sharpe_train = ep.sharpe_ratio(port_excess_returns_train)
sharpe_test = ep.sharpe_ratio(port_excess_returns_test)
sortino_train = ep.sortino_ratio(port_excess_returns_train)
sortino_test = ep.sortino_ratio(port_excess_returns_test)
maximum_drawdown_train = ep.max_drawdown(port_excess_returns_train)
maximum_drawdown_test = ep.max_drawdown(port_excess_returns_test)
calmar_train = ep.calmar_ratio(port_excess_returns_train)
calmar_test = ep.calmar_ratio(port_excess_returns_test)

print(f"Sharpe Ratio (In-sample): {sharpe_train:.2f}")
print(f"Sharpe Ratio (Out-of-sample):  {sharpe_test:.2f}")
print(f"Sortino Ratio (In-sample): {sortino_train:.2f}")
print(f"Sortino Ratio (Out-of-sample): {sortino_test:.2f}")
print(f"Max Drawdown (In-sample): {maximum_drawdown_train:.3f}")
print(f"Max Drawdown (Out-of-sample): {maximum_drawdown_test:.3f}")
print(f"Calmar Ratio (In-sample): {calmar_train:.2f}")
print(f"Calmar Ratio (Out-of-sample): {calmar_test:.2f}")

window = 60
rolling_mean = portfolio_excess_returns.rolling(window=window).mean()
rolling_std = portfolio_excess_returns.rolling(window=window).std()
rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(252)
rolling_sharpe.fillna(0, inplace=True) # To deal with discontinuities when std is zero
rolling_sharpe_train = rolling_sharpe[rolling_sharpe.index <= training_date]
rolling_sharpe_test = rolling_sharpe[rolling_sharpe.index > training_date]


# Plotting the return distribution
plt.figure(figsize=(10, 6))
plt.hist(daily_pnl, bins=30, edgecolor='black', alpha=0.7)
plt.title('Momentum Return Distribution')
plt.xlabel('Returns')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

def var(sample, alpha, bandwidth=0.12):
    
    # Calculate Value at Risk (VaR) using kernel density estimation (KDE).

    sample = -np.array(sample)
    kde = gaussian_kde(sample, bw_method=bandwidth)

    # Define the range of values for the x-axis
    x_values = np.linspace(min(sample), max(sample), 1000)

    # Compute the KDE values at each point on the x-axis
    kde_values = kde(x_values)

    # Compute the cumulative distribution function (CDF) from KDE
    cdf_values = np.cumsum(kde_values) / np.sum(kde_values)

    # Find the index corresponding to the specified tail probability level alpha
    index = np.argmax(cdf_values >= alpha)

    # Return the value at the index corresponding to VaR
    return x_values[index]

def es_mc(sample,alpha):
    ss=np.sort(sample)
    ialpha = int(sample.size * (1-alpha))
    return -ss[:ialpha].mean()

def var_mc(sample,alpha):
    ss=np.sort(sample)
    ialpha = int(sample.size * (1-alpha))
    return -ss[ialpha]

rolling_portfolio_returns = pd.Series(daily_pnl)
alpha = 0.99
rolling_VaR = rolling_portfolio_returns.rolling(window=252).apply(lambda x: var(x, alpha))
rolling_ES = rolling_portfolio_returns.rolling(window=252).apply(lambda x: es_mc(x, alpha))
rolling_VaR_mc = rolling_portfolio_returns.rolling(window=252).apply(lambda x: var_mc(x, alpha))
rolling_VaR.index = NVIDIA_excess_returns.index
rolling_ES.index = NVIDIA_excess_returns.index
rolling_VaR_mc.index = NVIDIA_excess_returns.index

fig, ax = plt.subplots(figsize=(10, 6))

# Plot rolling VaR
rolling_VaR.plot(ax=ax, color='blue', label='Rolling VaR Kernel')
rolling_VaR_mc.plot(ax=ax, color='g', linestyle='--', label='Rolling VaR MC')
# Plot rolling ES
rolling_ES.plot(ax=ax, color='red', label='Rolling ES MC')

# Set titles and labels
ax.set_title('Rolling VaR and ES at confidence level 99%', fontsize = 18)
ax.set_xlabel('Date', fontsize = 16)
ax.set_ylabel('Risk Meassure Value', fontsize = 16)
plt.xticks(fontsize=14) 
plt.yticks(fontsize=14) 
ax.legend(fontsize = 14)

# Show plot
plt.grid(True)
plt.savefig(r'C:\Users\andre\OneDrive\Desktop\MSc Computational Finance\Momentum_VaR_ES.pdf', format='pdf')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(rolling_sharpe_train.index, rolling_sharpe_train, color='blue', label = 'In-Sample')
plt.plot(rolling_sharpe_test.index, rolling_sharpe_test, color='green', label = 'Out-of-Sample')
plt.tight_layout()
plt.title('Rolling-60-day Sharpe Ratio', fontsize = 18)
plt.ylabel("Sharpe Ratio", fontsize = 16)
plt.xlabel("Date", fontsize = 16)
plt.grid()
plt.legend()
plt.xticks(fontsize=14) 
plt.yticks(fontsize=14) 
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
plt.savefig(r'C:\Users\andre\OneDrive\Desktop\MSc Computational Finance\Momentum_Rolling_Sharpe.pdf', format='pdf')
plt.show()   

def drawdown(daily_pnl):
    daily_pnl = np.array(daily_pnl)
    drawdown_array = np.zeros(len(daily_pnl))
    for i in range(len(daily_pnl)):
        drawdown_array[i] = np.max(daily_pnl[:i+1]) - daily_pnl[i]
    return drawdown_array

drawdown_t = drawdown(daily_pnl)
drawdown_t = pd.Series(drawdown_t, index=NVIDIA_excess_returns.index)    
lag = 90
historical_90_vol = NVIDIA_returns.rolling(window=lag).std()


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

ax1.plot(drawdown_t.index, drawdown_t, color='black')
ax1.set_title('Drawdown chart', fontsize=18)
ax1.set_ylabel("Drawdown", fontsize=16)
ax1.grid()
ax1.tick_params(axis='x', labelsize=14)
ax1.tick_params(axis='y', labelsize=14)

ax2.plot(historical_90_vol.index, historical_90_vol, color='red')
ax2.set_title('90-day realized Volatility', fontsize=18)
ax2.set_xlabel("Date", fontsize=16)
ax2.set_ylabel("Volatility", fontsize=16)
ax2.grid()
ax2.tick_params(axis='x', labelsize=14)
ax2.tick_params(axis='y', labelsize=14)
plt.savefig(r'C:\Users\andre\OneDrive\Desktop\MSc Computational Finance\Momentum_Drawdown_Chart.pdf', format='pdf')
plt.tight_layout()
plt.show()