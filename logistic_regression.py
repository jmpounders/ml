import numpy as np
from exceptions import RuntimeError

import optimization as opt
from auxillary import sigmoid

class LogisticRegression:
    """A basic implementation of logistic regression."""
    def __init__(self):
        self.theta_trained = None
        self.num_iters = None
        self.vtheta_err = None
        self.vcost = None

    def predict(self, X, theta=None, threshold=0.5):
        """Predict output based on observations X and linear weights theta."""
        theta = self.theta_trained if theta is None else theta
        if theta is None:
            raise RuntimeError('This model has not been trained.')

        return np.array(sigmoid(np.dot(X,theta)) >= threshold, dtype=int)
    
    def cost(self, X, theta, y):
        """Calculate the cost/objective function given observations X, linear weights theta, and target y."""
        m = X.shape[0]
        h = sigmoid(np.dot(X,theta))
        return -(np.dot(y,np.log(h)) + np.dot((1-y),np.log(1-h)))/m

    def cost_grad(self, X, theta, y):
        """Calculate the gradient of the cost/objective function at theta."""
        m = X.shape[0]
        h = sigmoid(np.dot(X,theta))
        return np.dot(X.T, h - y)/m

    def train(self, X, y, tol=0.0001, max_iters=500, verbose=False):
        """Train on data X with target y."""
        def grad_func(theta): return self.cost_grad(X,theta,y)
        if not verbose:
            self.theta_trained,self.num_iters = opt.gradient_descent(grad_func,
                                                                     0.1,
                                                                     np.zeros(X.shape[1]),
                                                                     max_iters,
                                                                     tol)
        else:
            def cost_func(theta): return self.cost(X,theta,y)
            self.theta_trained,self.num_iters,self.vtheta_err,self.vcost = opt.gradient_descent_verbose(
                grad_func,
                0.1,
                np.zeros(X.shape[1]),
                max_iters,
                tol,
                cost_func)
            
        return self.theta_trained

