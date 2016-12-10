"""
    vandermonde(x::Array{Float64,1}, N::Int)

Return the Vandermonde matrix associated with vector x.
"""
function vandermonde(x::Array{Float64,1}, N::Int)
    X = Array{Float64,2}(m,N+1)
    for i = 0:N
        X[:,i+1] = x.^i
    end
    X
end
