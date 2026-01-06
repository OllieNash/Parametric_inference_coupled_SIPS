import numpy as np
import matplotlib.pyplot as plt 
import scipy.stats as stats

#Parameters & inputs 
N = 10000 #Number of particles
T = 100 #Time 
h = 0.01 #time step 
num_steps = int(T/h) #Number of time steps
theta = 1 #Mean reversion parameter 
mu = 0 #true mean 
beta = 1 #diffusion coefficient

theta_estimates = np.empty(N)


#Drift function 
def drift(x, mu):
    return -theta * (x - mu) 

#Diffusion function 
def diffusion(x): 
    return np.sqrt(2/beta)

#Initial positions 
X = np.zeros((N,num_steps+1))
X[:, 0] = np.random.normal(loc=mu,scale=0.2,size=N)

#Simulate the system using Euler-Maruyama method
for t in range(num_steps):
    dW = np.random.normal(loc=0, scale=np.sqrt(h), size=N)
    X[:, t+1] = X[:, t] + drift(X[:, t], mu) * h + diffusion(X[:, t]) * dW 


#plotting the histogram of particle positions at final time
plt.hist(X[:, -1], bins=50, density=True, alpha=0.7)
plt.title('Histogram of Particle Positions at Final Time') 
plt.xlabel('Position')
plt.ylabel('Density')
plt.show() 



#Estimation of parameters using all time steps
X_t = X[:, :-1]                # shape (N, num_steps)
delta_X = X[:, 1:] - X[:, :-1] # shape (N, num_steps)

#Creating estimators for theta
for i in range(N): 
    theta_numerator = np.sum(-(X_t[i,:] - mu) * delta_X[i,:])
    theta_denominator = np.sum((X_t[i,:] - mu)**2) * h
    theta_hat = theta_numerator / theta_denominator
    theta_estimates[i] = theta_hat 


print(f"Estimated theta: {np.average(theta_estimates)}")
print(f"Theta standard deviation: {np.std(theta_estimates)}")

#plotting the estimated theta distribution
