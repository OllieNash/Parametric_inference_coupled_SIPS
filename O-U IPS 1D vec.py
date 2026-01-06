import numpy as np
import matplotlib.pyplot as plt 
import scipy.stats as stats


#Parameters & inputs 
N = 100 #Number of particles
T = 100 #Time 
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
X[:, 0] = np.random.normal(0,0.5,size=N) #Initial distribution
alpha_estimates = np.zeros((N,num_steps+1))


#Simulate the system using Euler-Maruyama method
for t in range(num_steps):
    dW = np.random.normal(loc=0, scale=np.sqrt(h), size=N)
    X[:, t+1] = X[:, t] + drift(X[:, t], mu) * h + diffusion(X[:, t]) * dW 
    
    X_slice = X[:, :t+2]                     # shape (N, t+2)
    X_delta = X_slice[:, 1:] - X_slice[:, :-1]  # shape (N, t+1)

    M_1 = np.sum(X_slice**2, axis=1) * h
    M_2 = np.sum((X_slice - mu) * X_slice, axis=1) * h
    M_3 = np.sum(X_slice[:, 1:] * X_delta, axis=1)

    alpha_hat = (M_3 - M_2) / M_1
    alpha_estimates[:, t+1] = alpha_hat

print(alpha_estimates)


    

alpha_hat = np.mean(alpha_estimates[:,-1])
print(f"Estimated alpha: {alpha_hat}")
print(f"Alpha standard deviation: {np.std(alpha_estimates[:,-1])}")

#plotting the histogram of alpha estimates
plt.hist(alpha_estimates[:,-1], bins=20, density=True, alpha=0.7)
plt.title('Histogram of alpha Estimates for N particles') 
plt.axvline(alpha, color='r', linestyle='dashed', linewidth=2, label='True alpha')
plt.axvline(alpha_hat, color='g', linestyle='dashed', linewidth=2, label='Estimated alpha')
plt.xlabel('Alpha value')
plt.ylabel('Density')
plt.legend()
plt.show() 

#Plotting estimators over time
mean_alpha_over_time = np.mean(alpha_estimates, axis=0)
plt.hlines(alpha, color='r', linestyle='dashed', linewidth=2, label='True alpha',xmin=0,xmax=num_steps)
plt.plot(mean_alpha_over_time)
plt.title('Mean Alpha Estimate Over Time')
plt.xlabel('Time Steps')
plt.ylabel('Mean Alpha Estimate')
plt.show()

#PLotting L2 loss over time
L2_loss_over_time = np.sqrt((mean_alpha_over_time[50:] - alpha)**2)
plt.plot(L2_loss_over_time)
plt.title('L2 Loss of Alpha Estimate Over Time')    
plt.xlabel('Time Steps')
plt.ylabel('L2 Loss')
plt.show()
