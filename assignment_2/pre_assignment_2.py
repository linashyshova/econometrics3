'''
Filename: pre_assignment_2.py

Purpose: First illustration about handling the data for the assignement
    
Date:
    11 Februari 2025  
'''
###########################################################
### Imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib.dates as mdates
import statsmodels.api as sm

##########################################################

def plot(data):
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True, sharex=True)
    data["date"] = pd.to_datetime(data["date"])

    axs[0].plot(data["date"], data["gdp"], label="GDP", color="b")
    axs[0].set_title("GDP")
    axs[0].grid()
    
    axs[1].plot(data["date"], data["cpi"], label="CPI", color="g")
    axs[1].set_title("CPI")
    axs[1].grid()
    
    axs[2].plot(data["date"], data["ir"], label="Interest Rate", color="r")
    axs[2].set_title("Interest Rate")
    axs[2].grid()

    axs[2].xaxis.set_major_locator(mdates.YearLocator(1))  # Major ticks every year
    axs[2].xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))  # Minor ticks every quarter

    axs[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))  
    for label in axs[2].get_xticklabels():
        if int(label.get_text()) % 5 != 0:  # Hide labels that are not multiples of 5
            label.set_visible(False)

    plt.xticks(rotation=30)    
    plt.show()

def plot_growth(data):
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True, sharex=True)
    data["date"] = pd.to_datetime(data["date"])

    axs[0].plot(data["date"], data["gdp_growth"], label="GDP Growth", color="b")
    axs[0].set_title("GDP")
    axs[0].grid()
    
    axs[1].plot(data["date"], data["cpi_growth"], label="CPI Growth", color="g")
    axs[1].set_title("CPI")
    axs[1].grid()
    
    axs[2].plot(data["date"], data["ir"], label="Interest Rate", color="r")
    axs[2].set_title("Interest Rate")
    axs[2].grid()

    axs[2].xaxis.set_major_locator(mdates.YearLocator(1))  # Major ticks every year
    axs[2].xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))  # Minor ticks every quarter

    axs[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))  
    for label in axs[2].get_xticklabels():
        if int(label.get_text()) % 5 != 0:  # Hide labels that are not multiples of 5
            label.set_visible(False)

    plt.xticks(rotation=30)    
    plt.show()

def prepare_AR_data(data, variable, p, p_max, intercept=True):
    y = data[variable].iloc[p_max:].reset_index(drop=True)
    X = pd.DataFrame()
    if intercept:
        X['intercept'] = np.ones(len(y))
    # Add lagged variables for all lags from 1 to p, but taking p_max into account to make sure we can properly compare the models
    for lag in range(1, p + 1):
        X[f'lag_{lag}'] = data[variable].shift(lag).iloc[p_max:].reset_index(drop=True)
    return X, y

def estimate_AR_model(y,X):
    y_array = np.array(y)
    X_array = np.array(X)
    beta_hat = np.linalg.inv(X_array.T @ X_array) @ X_array.T @ y_array
    residuals = y_array - X_array @ beta_hat
    n = len(y)
    k = X_array.shape[1]
    sigma_hat = residuals.T@residuals / (n - k)
    se_beta = np.sqrt(np.diagonal(sigma_hat * np.linalg.inv(X_array.T @ X_array)))
    t_stats = beta_hat / se_beta
    return beta_hat, residuals, se_beta, t_stats

def calculate_AIC_loglik(residuals, p, N):
    dsigma_hat = np.sum(residuals**2) / (N - p) #this is the one needed for the AIC, based on full sample N
    dsigmalogdet = np.log(dsigma_hat)
    dloglik = -0.5 * (dsigmalogdet + (N - p) * (1 + np.log(2 * np.pi)))
    daic = dsigmalogdet + (2.0 * p / N)
    return dloglik, daic, dsigma_hat

def run_AR_models(data, variable, max_lags, intercept):    
    for lag_p in range(1, max_lags + 1):
        X, y = prepare_AR_data(data, variable, lag_p, max_lags, intercept)        
        beta_hat, residuals, se_beta, t_stats = estimate_AR_model(y, X)        
        N = len(data)
        dloglik, daic, dsigma_hat = calculate_AIC_loglik(residuals, lag_p, N)
        results_df = pd.DataFrame({
            'lag': [lag_p] * (lag_p + 1),
            'variable': ['const'] + [f'{variable}_{i}' for i in range(1, lag_p + 1)],
            'coef': [beta_hat[0]] + list(beta_hat[1:]),
            'se': [se_beta[0]] + list(se_beta[1:]),
            't-value': [t_stats[0]] + list(t_stats[1:])
        })        
        print(results_df)
        print("Log-likelihood: %.4f"%dloglik)
        print("AIC: \t\t%.4f"%daic)    

###########################################################

def main():
    file_path = "data/data_assignment2.csv"
    data = pd.read_csv(file_path)
    total_quarters = len(data)
    years_short = 5                                     # 5 years since covid started
    quarters_short = total_quarters - years_short * 4 -1# Each year has 4 quarters
    
    data['gdp_growth'] = 100 * np.diff(np.log(data['gdp']), prepend=np.nan)
    data['cpi_growth']= 100 * np.diff(np.log(data['cpi']), prepend=np.nan)
    data = data.iloc[1:].reset_index(drop=True)         
    # Loose one observation due to differencing, however the new dataset has columns with dl values so these can also
    # be used, and one observation is not lost
    print(data)

    data_out_covid = data.iloc[:quarters_short, :]
    # Now we have the data without the covid period and with the covid period
    # You can switch the data to make a plot of the data without covid
    plot(data)
    plot_growth(data)

    max_lags=5
    intercept = True
    # Change variabes and data set yourself so desired setting
    run_AR_models(data_out_covid, 'ir', max_lags, intercept)

###########################################################
### call main
if __name__ == "__main__":
    main()