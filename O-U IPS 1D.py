import numpy as np
import matplotlib.pyplot as plt 


#Parameters & inputs 
N = 1000 #Number of particles
T = 200 #Time 
h = 0.01 #time step 
num_steps = int(T/h) #Number of time steps
theta = 1 #Mean reversion parameter 
alpha = 1 #interaction strength 
mu = 0 #true mean 
beta = 10 #diffusion coefficient

theta_estimates = []
alpha_estimates = []

#Drift function 
def drift(x, mu):
    return theta * (mu - x) + alpha * (0 - x) #Expectation of x is 0

#Diffusion function 
def diffusion(x): 
    return np.sqrt(2/beta)

#Initial positions 
X = np.zeros((N,num_steps+1))
X[:, 0] = np.zeros(shape=N)
alpha_estimates = []

#Simulate the system using Euler-Maruyama method
for t in range(num_steps):
    dW = np.random.normal(loc=0, scale=np.sqrt(h), size=N)
    X[:, t+1] = X[:, t] + drift(X[:, t], mu) * h + diffusion(X[:, t]) * dW 



for p in range(N):
        X_p = X[p,:]
        X_p_delta = X_p[1:] - X_p[:-1]

        M_1 = np.dot(X_p,X_p)*h 
        M_2 = np.dot(X_p - mu, X_p - 0)*h
        M_3 = np.dot(X_p[1:],X_p_delta)

        alpha_hat_p = (M_3-M_2)/(M_1) 
        alpha_estimates.append(alpha_hat_p)



alpha_hat = np.mean(alpha_estimates)
print(f"Estimated alpha: {alpha_hat}")
print(f"Alpha standard deviation: {np.std(alpha_estimates)}")

#plotting the histogram of alpha estimates
plt.hist(alpha_estimates, bins=25, density=True, alpha=0.7)
plt.title('Histogram of alpha Estimates for N particles') 
plt.xlabel('Alpha value')
plt.ylabel('Density')
plt.show() 


