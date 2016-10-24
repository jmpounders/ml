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

def gradient_descent_verbose(grad_func, alpha, theta, max_iters=500, tol=0.001, cost_func=None):
    theta_err = np.zeros(max_iters)
    cost = np.zeros(max_iters)

    if cost_func is None:
        def cost_func(theta): return 0.0
    
    for iter in range(max_iters):
        theta_old = theta
        theta = theta - alpha*grad_func(theta)
        err = np.abs(theta_old - theta)
        theta_err[iter] = np.sum(err)/len(theta)
        cost[iter] = cost_func(theta)
        if np.max(err/theta) < tol: break

    if iter==max_iters-1:
        warnings.warn('Gradient descent did not converge.')

    theta_err = theta_err[0:iter]
    cost = cost[0:iter]

    return theta,iter,theta_err,cost
