module MixtureModels
	using NumericExtensions
	using MLBase
	using Distributions

	export

	# types
	Mixture, ncomponents, 
	AbstractModelEstimator, MLE_Estimator, MAP_Estimator, 
	
	# utils
	qmatrix, qmatrix!

	include("types.jl")
	include("utils.jl")
end
