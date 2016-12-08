using Base.Test

using Optimization


grad(x) = 2*x + 1 
theta0 = [0.01]
@test gradient_descent(grad, 0.5, theta0) == [-0.5]
