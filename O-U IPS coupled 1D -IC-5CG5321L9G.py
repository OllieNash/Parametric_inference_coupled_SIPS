import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from tqdm import trange

from IPS_generation import IPS_generation
from Likelihood_Calculator import (
    likelihood_x_over_gamma,
    likelihood_y_over_theta,
    y_likelihood_surface,
)
from plotter import (
    L2_loss,
    estimates_overtime,
    plot_likelihood_parameter,
    plot_y_likelihood_contour,
    plot_y_likelihood_surface3d,
)
from spectral import eignvalues

#Parameters & inputs 
N = 1 #Number of particles
T = 300 #Time 
h = 0.01 #time step 
num_steps = int(T/h) #Number of time steps
theta_1 = 1 #Mean reversion parameter for X
theta_2 = 1.5 #Mean reversion parameter for Y
alpha_1 = 1.1 #interaction strength for X
alpha_2 = 1 #interaction strength for Y
gamma = 1 #cross-interaction strength
mu_x_true = 0 #true mean for x
mu_y_true = 0 #true mean for y
beta = 5 #diffusion coefficient



#Drift function 
def drift_x(x, mu):
    return theta_1 * (mu-x) + (alpha_1/2) * (0-x) +(gamma/2) *(0-x) #Cross-interaction term, expection of y is 0 (should change in future)

def drift_y(x, mu):
    return theta_2 * (mu-x) + (alpha_2/2) * (0-x) + (gamma/2)*(0-x)  #Expectation of x is 0

#Diffusion function 
def diffusion(x): 
    return np.sqrt(2/beta)

mu_x_partial, mu_y_partial, mu_x_full, mu_y_full = IPS_generation(500,10000)
#Initial positions 
X = np.zeros((N,num_steps+1))
X[:, 0] = np.random.normal(0,0.45,size=N) #Initial distribution for first system 
Y = np.zeros((N,num_steps+1))
Y[:, 0] = np.random.normal(0,0.45,size=N) #Initial distribution for second system 

test_theta = np.linspace(-20,20,1000) #We plot the likelihood at final time against these 
test_gamma = np.linspace(-20,20,1000)


#Simulate the system using Euler-Maruyama method
def sim_offline(mu_x,mu_y):
    gamma_estimates = np.zeros((N,num_steps+1))
    theta_estimates = np.zeros((N,num_steps+1))
    L_x = np.zeros((N,num_steps+1))
    L_y = np.zeros((N,num_steps+1))
    for t in trange(num_steps, desc="Simulating offline model"):
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
        M_3 = np.sum((X_slice-0)*(X_slice-mu_x),axis=1)*h
        M_4 = np.sum((X_slice[:, 1:] -0)* X_delta, axis=1)

        gamma_hat = (2*M_4 - 2*theta_1*M_2 - alpha_1 * M_3) / (M_1)
        gamma_estimates[:, t+1] = gamma_hat #Estimates for the cross-interaction

        N_1 = np.sum((Y_slice[:,1:]-mu_y)*Y_delta,axis=1)
        N_2 = np.sum((Y_slice-mu_y)*(Y_slice - 0),axis=1)*h
        N_3 = np.sum((Y_slice - mu_y)*(Y_slice - mu_x),axis=1)*h
        N_4 = np.sum((Y_slice - mu_y)**2,axis=1)*h

        theta_hat = (2*N_1-gamma*N_2-alpha_2*N_3)/(2*N_4)
        theta_estimates[:,t+1] = theta_hat


    e_val, e_vec = eignvalues(Y,mu_y,mu_x,beta,h)

    likelihood_x_parameter = likelihood_x_over_gamma(test_gamma, X_slice, X_delta, theta_1, alpha_1, mu_x, beta, h)
        
    likelihood_y_parameter = likelihood_y_over_theta(test_theta, Y_slice, Y_delta, alpha_2, mu_y, gamma, beta, h)

    surface_likelihood = y_likelihood_surface(test_theta, test_gamma, Y_slice, Y_delta, alpha_2, mu_y, beta, h)

    return theta_estimates, gamma_estimates, likelihood_x_parameter, likelihood_y_parameter, surface_likelihood, e_val, e_vec

theta_estimates, gamma_estimates, likelihood_x_p, likelihood_y_p, surface_y_l, e_val, e_vec = sim_offline(mu_x_full, mu_y_full)




#Cross interaction outputs 

gamma_hat = np.mean(gamma_estimates[:,-1])
print(f"Estimated Cross-interaction: {gamma_hat}")


#Plotting our likelihood function at T vs parameter values 
plot_likelihood_parameter(test_gamma, gamma, gamma_hat,likelihood_x_p,"alpha_{1,2}")


#Plotting alpha estimators over time
estimates_overtime(gamma_estimates, gamma, "Cross-interaction",num_steps)

#PLotting L2 loss over time
L2_loss(gamma_estimates, gamma, "Cross-interaction")



# Reversion estimation output 


theta_hat = np.mean(theta_estimates[:,-1])
print(f"Estimated theta: {theta_hat}")


#Plotting our likelihood function at T vs parameter values 
plot_likelihood_parameter(test_theta, theta_2, theta_hat,likelihood_y_p,"Theta 2")

#Plotting alpha estimators over time
estimates_overtime(theta_estimates, theta_2, "Theta 2",num_steps)

#PLotting L2 loss over time
L2_loss(theta_estimates, theta_2, "Theta 2")


print(f"Empirical distribution for x: {mu_x_full}")
print(f"Empirical distribution for y: {mu_y_full}")

print([(e_val[i], e_vec[:, i]) for i in range(len(e_val))])

plot_y_likelihood_surface3d(surface_y_l, test_theta,test_gamma, theta_hat,gamma_hat, cmap='viridis')
plot_y_likelihood_contour(surface_y_l, test_theta,test_gamma,theta_hat,gamma_hat,e_val,e_vec,100,  cmap='viridis')

