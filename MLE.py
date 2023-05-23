import numpy as np
from scipy.optimize import minimize

# Demand data
demand_data = np.array([10, 15, 20, 25, 30])

# Log-likelihood function for DWeibull distribution
def dweibull_loglikelihood(params, data):
    k, lam = params
    loglik = np.sum(np.log(k / lam * (data / lam)**(k - 1) * np.exp(-(data / lam)**k)))
    return -loglik

# Initial guess for parameters
initial_params = [1, 1]

# Perform maximum likelihood estimation
result = minimize(dweibull_loglikelihood, initial_params, args=(demand_data,))
estimated_params = result.x

# Extract the estimated parameters
k_param, lam_param = estimated_params

# Print the estimated parameters
print("Shape parameter (k):", k_param)
print("Scale parameter (lambda):", lam_param)