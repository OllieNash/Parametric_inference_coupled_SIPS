import numpy as np
import matplotlib.pyplot as plt 
import scipy.stats as stats
from tqdm import tqdm, trange
from IPS_generation import IPS_generation
from plotter import plot_likelihood_parameter, estimates_overtime, L2_loss

#Parameters & inputs 
N = 1 #Number of particles
T = 3000 #Time 
h = 0.01 #time step 
num_steps = int(T/h) #Number of time steps
theta_1 = 1 #Mean reversion parameter for X
theta_2 = 1.5 #Mean reversion parameter for Y
alpha_1 = 1.1 #interaction strength for X
alpha_2 = 1 #interaction strength for Y
gamma = 1 #cross-interaction strength
mu_x_true = 0.5 #true mean for x
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

mu_x_partial, mu_y_partial, mu_x_full, mu_y_full = IPS_generation(5000,3000)
#Initial positions 
X = np.zeros((N,num_steps+1))
X[:, 0] = np.random.normal(0,0.45,size=N) #Initial distribution for first system 
Y = np.zeros((N,num_steps+1))
Y[:, 0] = np.random.normal(0,0.45,size=N) #Initial distribution for second system 

theta_hat_2 = np.zeros((N,num_steps+1))
theta_hat_2[:,0] = np.random.uniform(-0.5,4.5,size=N) #Initial distribution for theta_2

gamma_hats = np.zeros((N,num_steps+1))
gamma_hats[:,0] = np.random.uniform(-0.5,4.5,size=N)



test_theta = np.linspace(-10,10,5000) #We plot the likelihood at final time against these 
test_gamma = np.linspace(-5,5,5000)


#Simulate the system using Euler-Maruyama method
def sim_online(mu_x,mu_y):
    gamma_estimates = np.zeros((N,num_steps+1))
    theta_estimates = np.zeros((N,num_steps+1))
    for t in trange(num_steps):
        dW_X = np.random.normal(loc=0, scale=np.sqrt(h), size=N)
        X[:, t+1] = X[:, t] + drift_x(X[:, t], mu_x) * h + diffusion(X[:, t]) * dW_X
        
        dW_Y = np.random.normal(loc=0,scale = np.sqrt(h),size=N)
        Y[:, t+1] = Y[:, t] + drift_y(Y[:, t],mu_y)*h + diffusion(Y[:,t])*dW_Y

        X_slice = X[:, :t+2]                     # shape (N, t+2)
        X_delta = X[:,t+1]-X[:,t]# shape (N, t+1)

        Y_delta = Y[:,t+1] - Y[:,t]    
        Y_slice = Y[:, : t+2]              # shape (N, t+2)
         # shape (N, t+1)
        if(t != 0):
            lr_gamma = min(0.1,0.1*t**-0.085)
            lr_theta = min(0.1,0.1*t**-0.125)
        else: 
            lr_gamma = lr_theta = 0.1
            

        N_1 = (Y[:,t]-mu_y)*Y_delta
        N_2 = (Y[:,t]-mu_y)*(Y[:,t]- 0)*h
        N_3 = (Y[:,t]- mu_y)*(Y[:,t] - 0)*h
        N_4 = ((Y[:,t] - mu_y)**2)*h

        M_1 = (X[:,t] - mu_y)*X_delta 
        M_2 = (X[:,t] - mu_y)*(X[:,t]-0)*h
        M_3 = (X[:,t]-mu_y)*(X[:,t]-mu_x)*h
        M_4 = ((X[:,t]-mu_y)**2)*h


        theta_hat_2[:,t+1] = theta_hat_2[:,t] + lr_theta*np.sqrt(beta/2) *(-N_1-theta_hat_2[:,t]*N_4 - 0.5*alpha_2*N_3 - 0.5*gamma*N_2 )
        gamma_hats[:,t+1] = gamma_hats[:,t] + lr_gamma*np.sqrt(beta/2)*(-0.5*M_1 - 0.5*theta_1*M_2 - 0.25*alpha_1*M_3 - 0.25*gamma_hats[:,t]*M_4)


   
    likelihood_x_parameter = np.zeros((N,len(test_gamma)))
    likelihood_y_parameter = np.zeros((N,len(test_theta)))
    for i in trange(len(test_gamma)):
        likelihood_x_parameter[:,i] = -np.sqrt(beta/2)*(np.sum((theta_1 * (mu_x-X_slice[:,1:]) + (alpha_1/2) * (0-X_slice[:,1:]) +(test_gamma[i]/2) *(0-X_slice[:,1:]))*(X_delta),axis=1) + ((1/2)* np.sum((theta_1 * (mu_x-X_slice[:,1:]) + (alpha_1/2) * (0-X_slice[:,1:]) +(test_gamma[i]/2) *(0-X_slice[:,1:]))**2,axis=1))*h)
        likelihood_y_parameter[:,i] = -np.sqrt(beta/2)*(np.sum((test_theta[i] * (mu_y-Y_slice[:,1:]) + (alpha_2/2) * (0-Y_slice[:,1:]) + (gamma/2)*(0-Y_slice[:,1:]) )*(Y_delta),axis=1) + ((1/2)* np.sum((test_theta[i] * (mu_y-Y_slice[:,1:]) + (alpha_2/2) * (0-Y_slice[:,1:]) + (gamma/2)*(0-Y_slice[:,1:]) )**2,axis=1))*h)

    return theta_estimates, gamma_estimates,  likelihood_x_parameter, likelihood_y_parameter

theta_estimates, gamma_estimates, likelihood_x_p, likelihood_y_p = sim_online(mu_x_full, mu_y_full)



theta_2_hat = theta_hat_2[:,-1]
print(f"Theta 2 estimate {theta_2_hat}")

gamma_hat = gamma_hats[:,-1]
print(f"Gamma estimate {gamma_hat}")


#Confining outputs 

plot_likelihood_parameter(test_theta,theta_2,theta_2_hat,likelihood_y_p,"Theta 2")
estimates_overtime(theta_hat_2, theta_2, "Theta 2",num_steps)
L2_loss(theta_hat_2,theta_2,"Theta 2")


#Cross interaction outputs 
plot_likelihood_parameter(test_gamma,gamma,gamma_hat,likelihood_x_p,"Gamma")
estimates_overtime(gamma_hats, gamma, "Gamma",num_steps)
L2_loss(gamma_hats,gamma,"Gamma")





print(f"Empirical distribution for x: {mu_x_full}")
print(f"Empirical distribution for y: {mu_y_full}")
