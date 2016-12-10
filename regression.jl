using Optimization

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
    cost(model, X, y)

Compute the cost function associated with a linear regression model.

Note: the cost function is not used directly in training.  It is provided
for information/validation only.
"""
function cost(model::LinearRegression, X, y)
    m = size(X)[1]
    err = X*model.theta - y
    dot(err,err)/(2*m) + penalty(model.theta,model.lmbda)
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

cost_grad(model::LinearRegression) = cost_grad_lr


# Penalty/Regularization Functions
# --------------------------------
"""
    penalty(theta, coeff, p=2)

Compute the regularization penalty.  Currently this is an Lp penaly excluding the bias term.
"""
function penalty(theta, coeff, p=2)
    coeff*(norm(theta,p)^p - abs(theta[1])^p)
end


"""
    penalty_grad(theta, coeff)

Compute the gradient of the penalty term.
"""
function penalty_grad(theta, coeff, p=2)
    grad = coeff*p*theta.*abs(theta).^(p-2)
    grad[1] = 0.0
end


# Training and predicting functions
# ---------------------------------
"""
    train!(model, X, y)

Train model using feature matrix X and labels y.
"""
function train!(model::Model, X, y)
    cost_grad_model = cost_grad(model)
    grad(theta) = cost_grad_model(theta, model.lmbda, X, y)
    model.theta = gradient_descent(grad, model.alpha, model.theta)
end


"""
    predict(model, X)

Make a prediction on feature matrix X given model.
"""
function predict(model, X)
    X*model.theta
end


"""
    solve_normal_equations!(model, X, y)

Solve the normal equations for regularized linear regression and
return the parameter vector in model.
"""
function solve_normal_equations!(model, X, y)
    L = eye(length(model.theta))
    L[1,1] = 0.0
    model.theta = (X'*X + model.lmbda*L)\(X'*y)
end
