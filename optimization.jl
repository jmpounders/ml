module Optimization

export gradient_descent

function gradient_descent(grad_func, alpha, theta::Array{Float64,1}, max_iters=500, tol=1.0e-4)
    converged = false
    for iter = 1:max_iters
        theta_old = theta
        theta = theta_old - alpha*grad_func(theta_old)
        err = abs(theta_old - theta)
        if maximum(err) < tol
            converged = true
            break
        end
    end

    if !converged
        println("Gradient descent did not converge.")
    end
    theta
end

end
