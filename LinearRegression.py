import numpy as np
from exceptions import RuntimeError

class LinearRegression:
    def predict(self,X,theta):
        return np.dot(X,theta)
    
    def cost(self,X,theta,y):
        if X.shape[1] != theta.shape[0]:
            raise RuntimeError('Data and weight dimensions dont agree')
        return (np.dot(X,theta) - y)**2/(2.0*X.shape[0])

    def cost_grad(self,x,theta):
        return np.dot(x.T, np.dot(X,theta) - y)/X.shape[0]
