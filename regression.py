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
    
    def cost(self,X,theta,y,lmbda=0.0):
        """Calculate the cost/objective function given observations X, linear weights theta, and target y."""
        m = X.shape[0]
        return (np.dot(X,theta) - y)**2/(2.0*m) + self.reg_func(theta,lmbda/m)

    def cost_grad(self,X,theta,y,lmbda=0.0):
        """Calculate the gradient of the cost/objective function at theta."""
        m = X.shape[0]
        return np.dot(X.T, np.dot(X,theta) - y)/m + self.reg_grad(theta,lmbda/m)

    def reg_func(self,theta,coeff):
        """Calculate the regularization term (ridge)."""
        return coeff*(np.dot(theta,theta) - theta[0]**2)

    def reg_grad(self,theta,coeff):
        """Calculate the gradient of the regularization term (ridge)."""
        grad = theta.copy()
        grad[0] = 0.0
        return coeff*grad

    def train(self,X,y,lmbda=0.0,tol=0.0001,max_iters=500):
        """Train on data X with target y."""
        def grad_func(theta): return self.cost_grad(X,theta,y,lmbda)
        self.theta_trained,iter = opt.gradient_descent(grad_func,
                                                       0.1,
                                                       np.zeros(X.shape[1]),
                                                       max_iters,
                                                       tol)
        return self.theta_trained

    def solve_normal_eqns(self,X,y,lmbda=0.0):
        """Solve the normal equations to get the linear regression solution."""
        self.theta_trained = np.linalg.solve(np.dot(X.T,X), np.dot(X.T,y))
        return self.theta_trained
