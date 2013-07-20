# Estimation of Gaussian mixture models using EM

println("Importing modules ...")

using NumericExtensions
using Distributions
using MixtureModels
using Winston

### config

theta = [0,3,5,7] * (pi/4)
K = length(theta)

println("Generating data ...")

function gen_cluster(t, n)
	ct = cos(t)
	st = sin(t)

	x0 = randn(n) + 5.0
	y0 = randn(n) * 0.3

	x = x0 * ct - y0 * st
	y = x0 * st + y0 * ct

	[x y]'
end

obs = hcat([gen_cluster(t, 500) for t in theta]...)

### fit model

println("Run 1 ......")
r = fit_fmm(MvNormal{PDMat}, obs, K, fmm_em(;display=:iter))

for t in 2 : 5
	println("Run $t ......")
	rt = fit_fmm(MvNormal{PDMat}, obs, K, fmm_em(;display=:iter))
	if rt.objective > r.objective
		r = rt
	end
	println()
end

comps = r.mixture.components

### visualization

function gauss_circle(g::MvNormal, r::Float64)
	μ = g.μ

	t = linspace(0, 2pi, 300)
	x = [cos(t)'; sin(t)'] * r
	unwhiten!(g.Σ, x)

	Curve(x[1,:] + μ[1], x[2,:] + μ[2], "color", "red")
end


p = FramedPlot()
add(p, Points(obs[1,:], obs[2,:], "type", "dot", "color", "blue"))
for k in 1 : K
	add(p, gauss_circle(comps[k], 2.0))
end
Winston.display(p)

# block & wait for the signal to exit

println("Press enter to exit.")
readline(STDIN)

