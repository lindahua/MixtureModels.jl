# Types

type Mixture{C}  # C is the component type
	K::Int
	components::Vector{C}
	pi::Vector{Float64}
end

function Mixture{C}(components::Vector{C}, pi::Vector{Float64})
	K = length(components)
	@check_argdims K == length(pi)
	Mixture{C}(K, components, pi)
end

ncomponents(m::Mixture) = m.K

