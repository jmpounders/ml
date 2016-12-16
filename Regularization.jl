module Regularization

export penalty, penalty_grad

"""
    penalty(theta, coeff, p=2)

Compute the regularization penalty.  Currently this is an Lp penaly excluding the bias term.
"""
function penalty(theta, coeff, p=2)
    coeff*(norm(theta,p)^p - abs(theta[1])^p)
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



end
