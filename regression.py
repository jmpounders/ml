import numpy as np

class LinearRegression:
    """A basic implementation of linear regression."""

    def predict(self,X,theta):
        """Predict output based on observations X and linear weights theta."""
        return np.dot(X,theta)
    
    def cost(self,X,theta,y):
        """Calculate the cost/objective function given observations X, linear weights theta, and target y."""
        return (np.dot(X,theta) - y)**2/(2.0*X.shape[0])

    def cost_grad(self,X,theta,y):
        """Calculate the gradient of the cost/objective function at theta."""
        return np.dot(X.T, np.dot(X,theta) - y)/X.shape[0]
