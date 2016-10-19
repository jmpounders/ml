import numpy as np

def gradient_descent(grad_func, alpha, theta, max_iters=500):
    for iter in range(max_iters):
        theta_old = theta
        theta = theta - alpha*grad_func(theta)
        err = np.abs(theta_old - theta)
        if np.max(err/theta) < 0.001: break

    return theta
