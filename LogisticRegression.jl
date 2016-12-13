using Optimization

abstract Model

"Simple logistic regression model"
type LogisticRegression <: Model
    theta::Array{Float64,1}
    thresh::Float64
    lmbda::Float64
    alpha::Float64
end

"Sigmoid function"
sigmoid(z) = 1.0./(1.0 + exp(-z))


"""
    cost_grad_logit(theta, lmbda, X, y)

Compute the gradient of the cost function for lostic regression.
"""
function cost_grad_logit(theta, lmbda, X, y)
    m = size(X)[1]
    err = sigmoid(X*theta) - y
    (X'*err)/m + penalty_grad(theta,lmbda)
end


"""
    penalty_grad(theta, coeff, p=2)

Compute the gradient of the penalty term.
"""
function penalty_grad(theta, coeff, p=2)
    grad = coeff*p*theta.*abs(theta).^(p-2)
    grad[1] = 0.0
    return grad
end

cost_grad(model::LogisticRegression) = cost_grad_logit

"""
    train!(model, X, y)

Train model using feature matrix X and labels y.
"""
function train!(model::LogisticRegression, X, y)
    cost_grad_model = cost_grad(model)
    grad(theta) = cost_grad_model(theta, model.lmbda, X, y)
    model.theta = gradient_descent(grad, model.alpha, model.theta)
end

"""
    predict(model, X)

Make a prediction on feature matrix X given model.
"""
function predict(model::LogisticRegression, X)
    sigmoid(X*model.theta) >= model.thresh ? 1 : 0
end
