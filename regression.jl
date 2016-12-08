using Optimization

type LinearRegression
    theta::Array{Float64,1}
    lmbda::Float64
    alpha::Float64
end

function cost_lr(model, X, y)
    m = size(X)[1]
    err = X*model.theta - y
    dot(err,err)/(2*m)
end

function cost_grad_lr(theta, X, y)
    m = size(X)[1]
    err = X*theta - y
    (X'*err)/m
end

function train(model, X, y)
    grad(theta) = cost_grad_lr(theta, X, y)
    model.theta = gradient_descent(grad, model.alpha, model.theta)
end

function predict(model, X)
    X*model.theta
end
