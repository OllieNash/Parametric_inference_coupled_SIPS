import numpy as np
import matplotlib.pyplot as plt 
import scipy.stats as stats
from tqdm import tqdm, trange

#Parameters & inputs 
N = 1000 #Number of particles
T = 100 #Time 
h = 0.01 #time step 
num_steps = int(T/h) #Number of time steps
theta_1 = 1 #Mean reversion parameter for X
theta_2 = 1 #Mean reversion parameter for Y
alpha_1 = 1.1 #interaction strength for X
alpha_2 = 1 #interaction strength for Y
gamma = 1 #cross-interaction strength
mu_x = 0 #true mean for x
mu_y = 0 #true mean for y
beta = 10 #diffusion coefficient

def drift_actual_mu(theta, alpha, gamma,x,y,mu): 
    return -theta*(x-mu) - alpha*(x-np.mean(x)) -gamma*(x-np.mean(y))

def drift_estimate_mu(theta,alpha,gamma,x,y): 
    return -theta*(x-np.mean(x)) - alpha*(x-np.mean(x)) -gamma*(x-np.mean(y))

def diffusion(): 
    return np.sqrt(2/beta)

def IPS_generation(N,num_steps):
    
    X = np.zeros((N,num_steps+1))
    X[:, 0] = np.random.normal(0,0.45,size=N) #Initial distribution for first system with actual mu
    Y = np.zeros((N,num_steps+1))
    Y[:, 0] = np.random.normal(0,0.45,size=N) #Initial distribution for second system with actual mu 

    X_est = np.zeros((N,num_steps+1))
    X_est[:, 0] = np.random.normal(0,0.45,size=N) #Initial distribution for first system with actual mu
    Y_est = np.zeros((N,num_steps+1))
    Y_est[:, 0] = np.random.normal(0,0.45,size=N) #Initial distribution for second system with actual mu 


    for t in trange(num_steps):
        dW_X = np.random.normal(loc=0, scale=np.sqrt(h), size=N)
        X[:, t+1] = X[:, t] + drift_actual_mu(theta_1,alpha_1,gamma,X[:, t], Y[:,t],mu_x) * h + diffusion() * dW_X
        
        dW_Y = np.random.normal(loc=0,scale = np.sqrt(h),size=N)
        Y[:, t+1] = Y[:, t] + drift_actual_mu(theta_2, alpha_2,gamma,Y[:, t],X[:,t],mu_y)*h + diffusion()*dW_Y

        dW_X_est = np.random.normal(loc=0, scale=np.sqrt(h), size=N)
        X_est[:, t+1] = X_est[:, t] + drift_estimate_mu(theta_1,alpha_1,gamma,X_est[:, t], Y_est[:,t]) * h + diffusion() * dW_X_est
        
        dW_Y_est = np.random.normal(loc=0,scale = np.sqrt(h),size=N)
        Y_est[:, t+1] = Y_est[:, t] + drift_estimate_mu(theta_2, alpha_2,gamma,Y_est[:, t],X_est[:,t])*h + diffusion()*dW_Y_est

    return np.mean(X[:,-1]), np.mean(Y[:,-1]), np.mean(X_est[:,-1]), np.mean(Y_est[:,-1])

