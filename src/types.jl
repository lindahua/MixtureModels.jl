# Types

import Distributions.fit_mle
import Distributions.fit

type Mixture{C}  # C is the component type
	K::Int
	components::Vector{C}
	pi::Vector{Float64}
end

function Mixture{C}(components::Vector{C}, pi::Vector{Float64})
	K = length(components)
	@check_argdims length(pi) == K
	Mixture{C}(K, components, pi)
end

ncomponents(m::Mixture) = m.K

abstract AbstractModelEstimator{C}


immutable MLE_Estimator{C} <: AbstractModelEstimator{C}
end

MLE_Estimator{C}(::Type{C}) = MLE_Estimator{C}()

fit{C}(est::MLE_Estimator{C}, data, w::Vector{Float64}) = fit_mle(C, data; weights=w)
logpri{C}(est::MLE_Estimator{C}, m::C) = 0.


# TODO: Enable this when Distributions.jl supports MAP estimation
#
# immutable MAP_Estimator{C, Pri} <: AbstractModelEstimator{C}
# 	prior::Pri
# end

# MAP_Estimator{C, Pri}(::Type{C}, pri::Pri) = MAP_Estimator{C, Pri}(pri)

# fit(est::MAP_Estimator, data, w::Vector{Float64}) = fit_map(C, est.prior, data; weights=w)
