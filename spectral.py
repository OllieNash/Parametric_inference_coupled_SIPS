import numpy as np 

def eignvalues(X, mu_x, mu_y,beta,h): 
    X_slice = X[:, :-1] 
    A_11 = (beta/2)*np.sum((X_slice-mu_x)**2,axis=1)*h
    A_21 = A_12 = (beta/4)*np.sum((X_slice-mu_x)*(X_slice-mu_y),axis=1)*h
    A_22 = (beta/8)*np.sum((X_slice -mu_y)**2,axis=1)*h
    A = np.array([[A_11.item(),A_12.item()],[A_21.item(),A_22.item()]],dtype=float)
    print(A)
    e_val, e_vec = np.linalg.eig(A)
    
    return e_val, e_vec