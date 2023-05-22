# Import libraries
import numpy as np
from distfit import distfit
from scipy.optimize import minimize
from scipy.stats import poisson



def distribution_apriori(X):
    # Initialize using the parametric approach.
    dfit = distfit(method='parametric', todf=True)

    # Fit model on input data X.
    dfit.fit_transform(X)

    # Print the bet model results.
    print("Best Model:")
    print(dfit.model)


def poisson_loglikelihood(params, x):
    # Define the log-likelihood function that corresponds to the assumed distribution: assuming Poisson.
    lambd = params[0]  # Parameter of the Poisson distribution
    return -np.sum(poisson.logpmf(x, lambd))



# Will replace with demand data later.
demand_data = [35, 44, 56, 47, 33, 35, 36, 36, 45, 85, 64, 52]
demand_data = np.array(demand_data)

#========STEP 1 : Theoretical distribution apriori using distfit=================================
distribution_apriori(demand_data)

#========STEP 2 : Perform MLE =================================
result = minimize(poisson_loglikelihood, x0=np.ones(1), args=(demand_data,))

estimated_param = result.x
print("Estimated parameters:", estimated_param)

# Generate next 5 years of demand data (assuming 52 weeks per year)
next_5_years_demand = poisson.rvs(mu=estimated_param, size=5*52)

# Print the predicted demand data
print("Predicted demand for the next 5 years:")
print(next_5_years_demand)