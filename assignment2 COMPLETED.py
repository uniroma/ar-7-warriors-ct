from os import remove
import pandas as pd
import numpy as np
#Preparing data

data_path = '/content/INDPRO.csv'
data = pd.read_csv(data_path)

data['LOG_INDPRO'] = np.log(data['INDPRO'])
data['DIFF_LOG_INDPRO'] = data['LOG_INDPRO'].diff()

print(data)

# Coding likelihood and finding estimates of parameters


from scipy.stats import norm

def conditional_likelihood_ar7(params, data):
    phi = params[:7]
    sigma2 = params[7]
    likelihood = 0
    for t in range(7, len(data)):
        y_pred = np.dot(phi, data[t-7:t][::-1])  # reverse to match the time lag order
        residual = data[t] - y_pred
        likelihood += norm.logpdf(residual, scale=np.sqrt(sigma2))
    return -likelihood  # Return negative likelihood for minimization

def unconditional_likelihood_ar7(params, data):
    # Initial values are treated as random draws from a normal distribution
    initial_mean = np.mean(data[:7])
    initial_var = np.cov(data[:7])
    initial_likelihood = np.sum(norm.logpdf(data[:7], initial_mean, np.sqrt(initial_var)))

    # Continue with the conditional likelihood from the 8th observation
    conditional_likelihood = conditional_likelihood_ar7(params, data)
    return -(initial_likelihood + conditional_likelihood)

# Minimize likelihood


from scipy.optimize import minimize

# Initial guess for parameters: 7 coefficients and 1 variance
initial_params = np.random.rand(8)

# Minimize the negative conditional likelihood
result_conditional = minimize(conditional_likelihood_ar7, initial_params, args=(data['DIFF_LOG_INDPRO'].values,), method='L-BFGS-B')

# Minimize the negative unconditional likelihood
result_unconditional = minimize(unconditional_likelihood_ar7, initial_params, args=(data['DIFF_LOG_INDPRO'].values,), method='L-BFGS-B')

print("Conditional Parameters:", result_conditional.x)
print("Unconditional Parameters:", result_unconditional.x)


#Forecast

def forecast_ar7(params, data, steps):
    phi = params[:7]
    forecast = list(data[-7:])  # start with the last 7 known values
    for _ in range(steps):
        forecast.append(np.dot(phi, forecast[-7:][::-1]))  # predict and append
    return forecast[-steps:]  # return the forecasted values

# Forecasting for the next 8 months
forecast_conditional = forecast_ar7(result_conditional.x, data['DIFF_LOG_INDPRO'].values, 8)
forecast_unconditional = forecast_ar7(result_unconditional.x, data['DIFF_LOG_INDPRO'].values, 8)

print("Forecast using Conditional Likelihood:", forecast_conditional)
print("Forecast using Unconditional Likelihood:", forecast_unconditional)

# Visualization


import matplotlib.pyplot as plt

# Plotting the forecasted values against the actual data
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['DIFF_LOG_INDPRO'], label='Actual', color='blue')
plt.plot(data.index[-8:], forecast_conditional, label='Conditional Forecast', color='red', linestyle='--')
plt.plot(data.index[-8:], forecast_unconditional, label='Unconditional Forecast', color='green', linestyle='--')
plt.xlabel('DATE')
plt.ylabel('Log Differences of INDPRO')
plt.title('Forecasting Log Differences of INDPRO')
plt.legend()
plt.show()
