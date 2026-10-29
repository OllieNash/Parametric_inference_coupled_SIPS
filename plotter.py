import matplotlib.pyplot as plt  
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np 


#Plotting likelihood vs estimated parameter at each time step
def plot_likelihood_overtime(estimates, estiamte, likelihood): 
    plt.plot(estimates[50:],likelihood,color='blue',marker='o',markersize=1,linestyle='-')
    plt.ylim(-20,5)
    plt.xlabel("Parameter values")
    plt.ylabel("Likelihood")
    plt.title('Likelihood vs parameter')
    plt.axvline(estiamte, color='g', linestyle='dashed', linewidth=2, label='Estimated alpha_{1,2}')
    plt.show()


def plot_likelihood_parameter(test_parameters, true_value, estimated_value, likelihood, parameter): 
    plt.plot(test_parameters,np.mean(likelihood,axis=0))
    plt.axvline(true_value, color='r', linestyle='dashed', linewidth=2, label=f'True {parameter}')
    plt.axvline(estimated_value, color='g', linestyle='dashed', linewidth=2, label=f'Estimated {parameter}')
    plt.xlabel(f"{parameter}")
    plt.ylabel("Likelihood value")
    plt.title(f"Likelihood function for {parameter}")
    plt.legend()
    plt.show()




#Plotting the estimates over the time period
def estimates_overtime(estimates, true_value, parameter,num_steps):
    mean_estimates_over_time = np.mean(estimates, axis=0)
    plt.hlines(true_value, color='r', linestyle='dashed', linewidth=2, label=f'True {parameter}',xmin=0,xmax=num_steps)
    plt.hlines(mean_estimates_over_time[-1],color='g', linestyle='dashed', linewidth=2, label=f'Estimated {parameter}',xmin=0,xmax=num_steps)
    plt.plot(mean_estimates_over_time[1000:])
    plt.title(f'{parameter} Estimate vs time-steps')
    plt.xlabel('Number of time Steps')
    plt.ylabel(f'{parameter} Estimate')
    plt.show()

def L2_loss(estimates,true_value,parameter):
    mean_estimates_over_time = np.mean(estimates, axis=0)
    L2_loss_over_time = np.sqrt((mean_estimates_over_time[1000:] - true_value)**2)
    plt.plot(L2_loss_over_time)
    plt.title(f'L2 Loss of {parameter} Estimate Over Time')    
    plt.xlabel('Number of time Steps')
    plt.ylabel('L2 Loss')
    plt.show()


def plot_y_likelihood_contour(L_surface, test_theta, test_gamma, theta_hat,gamma_hat,e_val,e_vec, levels=100, cmap='viridis'):
    th, gm = np.meshgrid(test_theta, test_gamma, indexing='ij')
    cf = plt.contourf(th, gm, L_surface, levels=levels, cmap=cmap)
    plt.plot(theta_hat,gamma_hat,'o')
    for i in range(len(e_val)): 
        v_x,v_y = e_vec[:,i]
        L=max(test_gamma)
        x = np.array([theta_hat - L*v_x, theta_hat + L*v_x])
        y = np.array([gamma_hat - L*v_y, gamma_hat + L*v_y])
        plt.plot(x, y, '--', linewidth=2, label=f"λ={e_val[i]:.3g}")
    plt.xlabel('theta_2'); plt.ylabel('alpha_{1,2}'); plt.colorbar(cf, label='likelihood'); plt.legend(loc='best'); plt.show()

def plot_y_likelihood_surface3d(L_surface, test_theta, test_gamma, theta_hat, gamma_hat,cmap='viridis'):
    th, gm = np.meshgrid(test_theta, test_gamma, indexing='ij')
    ax = plt.figure(figsize=(8,6)).add_subplot(111, projection='3d')
    ax.plot_surface(th, gm, L_surface, cmap=cmap, edgecolor='none')
    ax.scatter(theta_hat,gamma_hat,np.max(L_surface),color='red',s=50)
    ax.set_xlabel('theta_2'); ax.set_title("Likelihood profile"); ax.set_ylabel('alpha_{1,2}'); ax.set_zlabel('likelihood'); plt.show()
