module Optimization

export gradient_descent, gradient_descent_verbose

"""
    gradient_descent(grad_func, alpha, theta::Array{Float64,1};
                     max_iters=1000, tol=1.0e-4,
                     callback=nothing, objective=nothing)

Return the parameter vector obtained by applying gradient descent with the grad_func
gradient function.

The other parameters are defined as follows:
    alpha is the learning rate
    theta is the initial guess
    max_iters is the maximum number of iterations to perform
    tol is the stopping tolerance
    callback is an optional function to call following every iteration
"""
function gradient_descent(grad_func, alpha, theta::Array{Float64,1};
                          max_iters=1000, tol=1.0e-4,
                          callback=nothing, objective=nothing)
#    kwargs = Dict(kwargs)
    if callback != nothing
        status = Dict()
    end
    
    converged = false
    for iter = 1:max_iters
        theta_old = theta
        theta = theta_old - alpha*grad_func(theta_old)
        err = abs(theta_old - theta)
        if callback != nothing
            if objective != nothing
                status["objective"] = objective(theta)
            end
            status["error"] = err
            callback(status)
        end
        if maximum(err) < tol
            converged = true
            break
        end
    end

    if !converged
        warn("Gradient descent did not converge.")
    end
    theta
end


end
