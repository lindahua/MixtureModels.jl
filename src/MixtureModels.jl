module MixtureModels
	using NumericExtensions
	using MLBase
	using Distributions

	export

	# types
	Mixture, ncomponents, 
	
	# utils
	qmatrix, qmatrix!,

	# fmm
	FiniteMixtureEM, FiniteMixtureEMResults, fmm_em,
	fit_fmm!, fit_fmm

	include("types.jl")
	include("utils.jl")
	include("fmm_em.jl")
end
