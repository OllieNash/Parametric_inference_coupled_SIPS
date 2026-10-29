"""Trajectories of the S-L (double-well) drift x(alpha - x^2), driven by averaged noise.

Separate from the O-U inference in the rest of the repo; no estimation is done here.
"""
import numpy as np
import matplotlib.pyplot as plt

#Parameters & inputs
N = 3 #Number of particles
T = 100 #Time  
h = 0.01 #time step
num_steps = int(T/h) #Number of time steps
theta = 5 #Mean reversion parameter
alpha = 2.5 #interaction strength 
mu = 0 #true mean 
beta = 50 #diffusion coefficient 

def drift(x, mu):
    """M-V O-U drift, with the expectation of x taken as 0."""
    return theta * (mu - x) + alpha * (0 - x) #Expectation of x is 0
def diffusion(x):
    """Diffusion coefficient sqrt(2/beta)."""
    return np.sqrt(2/beta)

def drift_SL(x):
    """S-L (double-well) drift x(alpha - x^2)."""
    return x*(alpha - x**2)

def initial_positions(N):
    """Path array of shape (N, num_steps+1) with N(0, 0.5^2) initial positions."""
    X = np.zeros((N,num_steps+1))
    X[:, 0] = np.random.normal(0,0.5,size=N)
    return X

def non_markovian(N, h):
    """Euler steps of the S-L drift with noise (dW_1 + dW_2)/2: the average of two independent increments (variance h/2)."""
    X = initial_positions(N)
    for t in range(num_steps):
        dW_1 = np.random.normal(loc=0, scale=np.sqrt(h), size=N)
        dW_2 = np.random.normal(loc=0, scale=np.sqrt(h), size=N)
        X[:, t+1] = X[:, t] + drift_SL(X[:, t]) * h + diffusion(X[:, t]) * (dW_1 + dW_2)/2 #**Richardson Extrapolation
    return X

def Euler_maruyama(N,h):
    """Standard Euler-Maruyama for the O-U drift, for comparison."""
    X = initial_positions(N)
    for t in range(num_steps):
        dW = np.random.normal(loc=0, scale=np.sqrt(h), size=N)
        X[:, t+1] = X[:, t] + drift(X[:, t],0) * h + diffusion(X[:, t]) *dW #**Richardson Extrapolation
    return X

X = non_markovian(3,0.01)
#X = Euler_maruyama(5,0.01)

#Plotting trajectories of particles
for i in range(N):
    plt.plot(X[i, :], alpha=0.7)
plt.title('Trajectories of Particles over Time')
plt.xlabel('Time Steps')
plt.ylabel('Position')
plt.show()
