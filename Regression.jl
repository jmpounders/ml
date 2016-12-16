module Regression

export LinearRegression, LassoRegression, RidgeRegression
export train!, predict, solve_normal_equations!

using Optimization
using Regularization

abstract Model

"Simple linear regression model"
type LinearRegression <: Model
    theta::Array{Float64,1}
    lmbda::Float64
    alpha::Float64
    penalty_norm::Int
end

LinearRegression(theta,lmbda,alpha) = LinearRegression(theta,lmbda,alpha,0)
LassoRegression(theta,lmbda,alpha) = LinearRegression(theta,lmbda,alpha,1)
RidgeRegression(theta,lmbda,alpha) = LinearRegression(theta,lmbda,alpha,2)

# Cost Functions
# --------------
"""
    cost_lr(model, lmbda, X, y)

Compute the cost function associated with a linear regression model.

Note: the cost function is not used directly in training.  It is provided
for information/validation only.
"""
function cost_lr(theta, lmbda, X, y)
    m = size(X)[1]
    err = X*theta - y
    dot(err,err)/(2*m) + penalty(theta,lmbda)
end


"""
    cost_grad_lr(theta, lmbda, X, y)

Compute the gradient of the cost function associated with a linear regression model.
"""
function cost_grad_lr(theta, lmbda, X, y)
    m = size(X)[1]
    err = X*theta - y
    (X'*err)/m + penalty_grad(theta,lmbda)
end

cost(model::LinearRegression) = cost_lr
cost_grad(model::LinearRegression) = cost_grad_lr


# Training and predicting functions
# ---------------------------------
"""
    train!(model, X, y)

Train model using feature matrix X and labels y.
"""
function train!(model::LinearRegression, X, y; params...)

    cost_model = cost(model)
    cost_grad_model = cost_grad(model)

    objective(theta) = cost_model(theta, model.lmbda, X, y)
    grad(theta) = cost_grad_model(theta, model.lmbda, X, y)

    params = Dict{Any,Any}(params)
    if !(:objective in keys(params))
        params[:objective] = objective
    end
    
    model.theta = gradient_descent(grad, model.alpha, model.theta; params...)
end


"""
    predict(model, X)

Make a prediction on feature matrix X given model.
"""
function predict(model::LinearRegression, X)
    X*model.theta
end


"""
    solve_normal_equations!(model, X, y)

Solve the normal equations for regularized linear regression and
return the parameter vector in model.
"""
function solve_normal_equations!(model::LinearRegression, X, y)
    L = eye(length(model.theta))
    L[1,1] = 0.0
    model.theta = (X'*X + model.lmbda*L)\(X'*y)
end

end
