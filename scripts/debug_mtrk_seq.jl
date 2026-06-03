using KomaMRI
using NPZ
using JSON
using LinearAlgebra
using HDF5
using Logging

using KomaInterface

mtrk = with_logger(ConsoleLogger(stdout,Logging.Debug;show_limited=false)) do
	KomaInterface.read_seq(joinpath(@__DIR__,"../makeitKOMA/data/mtrk_spoiled_gre.mtrk"))
end
##
pulseq = KomaInterface.read_seq(joinpath(@__DIR__,"../makeitKOMA/data/mtrk_spoiled_gre.seq"))

##
n = 128
p1=plot_seq(pulseq[8(n-1)+1:8(n-1)+7]) # 1:7,9:15,17:23 -- 8(n-1)+1:8(n-1)+7
p2=plot_seq(  mtrk[3(n-1)+1:3(n-1)+3]) # 1:3,4:6,7:9    -- 3(n-1)+1:3(n-1)+3
[p1;p2]

##
using LaTeXStrings,Plots
Plots.pythonplot()
function plot_discretized_seq(seq;Δt::Real=1//10^6,bounds::Tuple{Real,Real}=(-Inf,Inf),label::String="",plot_grad::Bool=true,plot_phase::Bool=true)
	(issorted(bounds) && !any(isnan,bounds)) || throw(ArgumentError("bounds must be specified in non-decreasing order"))
	plot_grad || plot_phase || throw(ArgumentError("plot_grad and plot_phase must not be false simultaneously"))
	seqd = discretize(seq;sampling_params=Dict("Δt" => Δt,"Δt_rf" => Δt))
	t = 1000 .* seqd.t
	bounds = 1000 .* bounds
	G = stack((seqd.Gx,seqd.Gy,seqd.Gz);dims=2)
	if any(isfinite,bounds)
		m = bounds[1] .≤ t .≤ bounds[2]
		t = t[m]
		G = G[m,:]
	end
	if plot_phase
		ϕ = similar(G)
		cumsum!(ϕ,G;dims=1)
		ϕ .*= Δt
	end
	if plot_grad
		title = LaTeXString("$label \$\\vec{G}(t)\$")
		labels = LaTeXString["\$G_$ax(t)\$" for ax in ['x' 'y' 'z']]
		p1=Plots.plot(t,G;title,labels)
		!plot_phase && return p1
	end
	if plot_phase
		title = LaTeXString("$label \$\\int_{0}^{t} \\vec{G}(\\tau)\\,\\mathrm{d}\\tau\$")
		labels = LaTeXString["\$\\int_{0}^{t}G_$ax(\\tau)\\,\\mathrm{d}\\tau\$" for ax in reshape('x':'z',1,3)]
		p2=Plots.plot(t,ϕ;title,labels,legend_position=:topleft,ylim=(-1e-3,1e-3))
		!plot_grad && return p2
	end
	return Plots.plot(p1,p2;layout=(1,2))
end
bounds = (0,50e-3)
p1 = plot_discretized_seq(mtrk;label="mtrk",plot_grad=true,plot_phase=false,bounds)
p2 = plot_discretized_seq(pulseq;label="pulseq",plot_grad=true,plot_phase=false,bounds)
Plots.plot(p1,p2;layout=(2,1),size=(1600,600),xlabel="t [ms]")

##
using Interpolations,StatsBase
Δt = 1//10^6
mtrkd = discretize(mtrk;sampling_params=Dict("Δt" => Δt,"Δt_rf" => Δt))
pulseqd = discretize(pulseq;sampling_params=Dict("Δt" => Δt,"Δt_rf" => Δt))
interp_mtrk = linear_interpolation(mtrkd.t, mtrkd.Gz)
t = 0:Δt:13//1000
mtrkGz = interp_mtrk.(t)
interp_pulseq = linear_interpolation(pulseqd.t, pulseqd.Gz)
pulseqGz = interp_pulseq.(t)
lags = vcat(0:10)
lags = vcat(.-lags[end:-1:2],lags)
cor = StatsBase.crosscor(mtrkGz,pulseqGz,lags)
Plots.plot(lags,cor,markershape=:x,title=L"Covariance of mtrk $G_z$ and pulseq $G_z$",labels=nothing,xlabel="Shift")