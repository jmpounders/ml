import numpy as np
import warnings

def gradient_descent(grad_func, alpha, theta, max_iters=500, tol=0.001):
    for iter in range(max_iters):
        theta_old = theta
        theta = theta - alpha*grad_func(theta)
        err = np.abs(theta_old - theta)
        if np.max(err/theta) < tol: break

    if iter==max_iters-1:
        warnings.warn('Gradient descent did not converge.')

    return theta,iter
