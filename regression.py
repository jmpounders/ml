import numpy as np
from exceptions import RuntimeError

import optimization as opt

class LinearRegression:
    """A basic implementation of linear regression."""
    def __init__(self):
        self.theta_trained = None

    def predict(self,X,theta=None):
        """Predict output based on observations X and linear weights theta."""
        theta = self.theta_trained if theta is None else theta
        if theta is None:
            raise RuntimeError('This model has not been trained.')
        
        return np.dot(X,theta)
    
    def cost(self,X,theta,y):
        """Calculate the cost/objective function given observations X, linear weights theta, and target y."""
        return (np.dot(X,theta) - y)**2/(2.0*X.shape[0])

    def cost_grad(self,X,theta,y):
        """Calculate the gradient of the cost/objective function at theta."""
        return np.dot(X.T, np.dot(X,theta) - y)/X.shape[0]

    def train(self,X,y):
        """Train on data X with target y."""
        def grad_func(theta): return self.cost_grad(X,theta,y)
        self.theta_trained = opt.gradient_descent(grad_func, 0.1, np.zeros(X.shape[1]))
        return self.theta_trained
