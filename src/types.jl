# Types

type Mixture{G}
	K::Int
	components::Vector{G}
	pi::Vector{Float64}
end

function Mixture{G}(components::Vector{G}, pi::Vector{Float64})
	K = length(components)
	if length(pi) != K
		throw(ArgumentError("Inconsistent argument lengths."))
	end
	Mixture{G}(K, components, pi)
end

ncomponents(m::Mixture) = m.K

