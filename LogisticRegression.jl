module LogisticRegression

export LogitRegression
export train!, predict

using Optimization
using Regularization

abstract Model

"Simple logistic regression model"
type LogitRegression <: Model
    theta::Array{Float64,1}
    thresh::Float64
    lmbda::Float64
    alpha::Float64
end

"Sigmoid function"
sigmoid(z) = 1.0./(1.0 + exp(-z))


# Cost Functions
# --------------
"""
    cost_logit(theta, lmbda, X, y)

Compute the cost function associated with logistic regression.
"""
function cost_logit(theta, lmbda, X, y)
    m = size(X)[1]
    h = sigmoid(X*theta)
    (-y'*log(h) - (1-y)'*log(1-h))/m + penalty(theta,lmbda)
end



"""
    cost_grad_logit(theta, lmbda, X, y)

Compute the gradient of the cost function for lostic regression.
"""
function cost_grad_logit(theta, lmbda, X, y)
    m = size(X)[1]
    err = sigmoid(X*theta) - y
    (X'*err)/m + penalty_grad(theta,lmbda)
end

cost(model::LogitRegression) = cost_logit
cost_grad(model::LogitRegression) = cost_grad_logit

# Training and predicting functions
# ---------------------------------
"""
    train!(model, X, y)

Train model using feature matrix X and labels y.

The params argument provides the parameter input to the
optimization routine.  The objective function is added
to these parameters, but it is only called if a callback
function is also provided.
"""
function train!(model::LogitRegression, X, y; params...)

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
function predict(model::LogitRegression, X)
    h = sigmoid(X*model.theta)
    [hi >= model.thresh ? 1 : 0 for hi in h]
end

end
