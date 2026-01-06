import numpy as np
import matplotlib.pyplot as plt 
import scipy.stats as stats
from IPS_generation import IPS_generation


#Parameters & inputs 
N = 10 #Number of particles
T = 200 #Time 
h = 0.01 #time step 
num_steps = int(T/h) #Number of time steps
theta_1 = 1 #Mean reversion parameter for X
theta_2 = 1.5 #Mean reversion parameter for Y
alpha_1 = 1.1 #interaction strength for X
alpha_2 = 1 #interaction strength for Y
gamma = 1 #cross-interaction strength
mu_x_true = 0 #true mean for x
mu_y_true = 0 #true mean for y
beta = 10 #diffusion coefficient



#Drift function 
def drift_x(x, mu):
    return theta_1 * (mu-x) + (alpha_1/2) * (0-x) +(gamma/2) *(0-x) #Cross-interaction term, expection of y is 0 (should change in future)

def drift_y(x, mu):
    return theta_2 * (mu-x) + (alpha_2/2) * (0-x) + (gamma/2)*(0-x)  #Expectation of x is 0

#Diffusion function 
def diffusion(x): 
    return np.sqrt(2/beta)

mu_x_partial, mu_y_partial, mu_x_full, mu_y_full = IPS_generation(100,10000)
#Initial positions 
X = np.zeros((N,num_steps+1))
X[:, 0] = np.random.normal(0,0.45,size=N) #Initial distribution for first system 
Y = np.zeros((N,num_steps+1))
Y[:, 0] = np.random.normal(0,0.45,size=N) #Initial distribution for second system 
gamma_estimates = np.zeros((N,num_steps+1))
theta_estimates = np.zeros((N,num_steps+1))


#Simulate the system using Euler-Maruyama method
def sim(mu_x,mu_y):
    for t in range(num_steps):
        print(t)
        dW_X = np.random.normal(loc=0, scale=np.sqrt(h), size=N)
        X[:, t+1] = X[:, t] + drift_x(X[:, t], mu_x) * h + diffusion(X[:, t]) * dW_X
        
        dW_Y = np.random.normal(loc=0,scale = np.sqrt(h),size=N)
        Y[:, t+1] = Y[:, t] + drift_y(Y[:, t],mu_y)*h + diffusion(Y[:,t])*dW_Y

        X_slice = X[:, :t+2]                     # shape (N, t+2)
        X_delta = X_slice[:, 1:] - X_slice[:, :-1]  # shape (N, t+1)

        Y_slice = Y[:, :t+2]                     # shape (N, t+2)
        Y_delta = Y_slice[:, 1:] - Y_slice[:, :-1]  # shape (N, t+1)


        M_1 = np.sum((X_slice-0)**2, axis=1) * h #
        M_2 = np.sum((X_slice - mu_x) * (X_slice-0), axis=1) * h
        M_3 = np.sum((X_slice-0)*(X_slice-0),axis=1)*h
        M_4 = np.sum((X_slice[:, 1:] -0)* X_delta, axis=1)

        gamma_hat = (2*M_4 - 2*theta_1*M_2 - alpha_1 * M_3) / M_1
        gamma_estimates[:, t+1] = gamma_hat #Estimates for the cross-interaction

        N_1 = np.sum((Y_slice[:,1:]-mu_y)*Y_delta,axis=1)
        N_2 = np.sum((Y_slice-mu_y)*(Y_slice - 0),axis=1)*h
        N_3 = np.sum((Y_slice - mu_y)*(Y_slice - 0),axis=1)*h
        N_4 = np.sum((Y_slice - mu_y)**2,axis=1)*h

        theta_hat = (2*N_1-gamma*N_2-alpha_2*N_3)/(2*N_4)
        theta_estimates[:,t+1] = theta_hat
    return theta_estimates, gamma_estimates

theta_estimates, gamma_estimates = sim(mu_x_full, mu_y_full)


#Cross interaction outputs 

gamma_hat = np.mean(gamma_estimates[:,-1])
print(f"Estimated gamma: {gamma_hat}")
print(f"Gamma standard deviation: {np.std(gamma_estimates[:,-1])}")

print(gamma_estimates, type(gamma_estimates))
#print(theta_estimates, type(theta_estimates))


#plotting the histogram of alpha estimates
plt.hist(gamma_estimates[:,-1], bins=20, density=True, alpha=0.7)
plt.title('Histogram of gamma estimates for N particles') 
plt.axvline(gamma, color='r', linestyle='dashed', linewidth=2, label='True gamma')
plt.axvline(gamma_hat, color='g', linestyle='dashed', linewidth=2, label='Estimated gamma')
plt.xlabel('Cross-interaction value')
plt.ylabel('Density')
plt.legend()
plt.show() 

#Plotting estimators over time
mean_gamma_over_time = np.mean(gamma_estimates, axis=0)
plt.hlines(gamma, color='r', linestyle='dashed', linewidth=2, label='True gamma',xmin=0,xmax=num_steps)
plt.plot(mean_gamma_over_time[50:])
plt.title('Mean Gamma Estimate Over Time')
plt.xlabel('Time Steps')
plt.ylabel('Mean Gamma Estimate')
plt.show()

#PLotting L2 loss over time
L2_loss_over_time = np.sqrt((mean_gamma_over_time[50:] - gamma)**2)
plt.plot(L2_loss_over_time)
plt.title('L2 Loss of gamma Estimate Over Time')    
plt.xlabel('Time Steps')
plt.ylabel('L2 Loss')
plt.show()



# Reversion estimation output 

theta_hat = np.mean(theta_estimates[:,-1])
print(f"Estimated theta: {theta_hat}")
print(f"Gamma standard deviation: {np.std(theta_estimates[:,-1])}")

#plotting the histogram of alpha estimates
plt.hist(theta_estimates[:,-1], bins=20, density=True, alpha=0.7)
plt.title('Histogram of theta estimates for N particles') 
plt.axvline(theta_2, color='r', linestyle='dashed', linewidth=2, label='True theta')
plt.axvline(theta_hat, color='g', linestyle='dashed', linewidth=2, label='Estimated theta')
plt.xlabel('Mean reversion value')
plt.ylabel('Density')
plt.legend()
plt.show() 

#Plotting estimators over time
mean_theta_over_time = np.mean(theta_estimates, axis=0)
plt.hlines(theta_2, color='r', linestyle='dashed', linewidth=2, label='True theta',xmin=0,xmax=num_steps)
plt.plot(mean_theta_over_time[50:])
plt.title('Mean Theta Estimate Over Time')
plt.xlabel('Time Steps')
plt.ylabel('Mean Theta Estimate')
plt.show()

#PLotting L2 loss over time
L2_loss_over_time = np.sqrt((mean_theta_over_time[50:] - theta_2)**2)
plt.plot(L2_loss_over_time)
plt.title('L2 Loss of Theta Estimate Over Time')    
plt.xlabel('Time Steps')
plt.ylabel('L2 Loss')
plt.show()
