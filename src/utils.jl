### Utilities

# Functions to construct q-matrix

function qmatrix!(q::Array{Float64}, labels::Vector{Int})
    @check_argdims size(q, 1) == length(labels)
    fill!(q, 0.0)
    for i = 1 : size(q, 1)
        q[i, labels[i]] = 1.0
    end
    q
end

function qmatrix!(q::Array{Float64}; p=0.0) # n x k matrix, each row sum to 1
    if !(0. <= p <= 1.)
        throw(ArgumentError("The value of p must be in [0, 1]"))
    end

    n = size(q, 1)
    K = size(q, 2)

    if p == 0.0
        rand!(q)
        bmultiply!(q, 1.0 ./ sum(q, 2), 1)
    elseif p == 1.0
        fill!(q, 0.0)
        rg = 1:K
        for i = 1:n
            k = rand(rg)
            q[i, k] = 1.0
        end 
    else 
        rand!(q)
        bmultiply!(q, (1 - p) ./ sum(q, 2), 1)
        rg = 1:K
        for i = 1:n
            k = rand(rg)
            q[i,k] += p
        end
    end
    q
end

qmatrix(labels::Vector{Int}, K::Int) = qmatrix!(Array(Float64, length(labels), K), labels)

qmatrix(n::Int, K::Int; p=0.0) = qmatrix!(Array(Float64, n, K); p=p)
