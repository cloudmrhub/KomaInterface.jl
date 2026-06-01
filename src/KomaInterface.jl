module KomaInterface

using KomaMRI
using JSON
using StructArrays
using NPZ
using LinearAlgebra
using HDF5
using Logging
using ArgParse

include("./mtrk.jl")
include("./simulate.jl")

function read_seq(filename::AbstractString)
    endswith(filename,".seq")  && return KomaMRI.read_seq(filename)
    endswith(filename,".mtrk") && return read_seq_mtrk(filename)
    throw(ArgumentError("File at \"$filename\" does not end in \".mtrk\" or \".seq\""))
end
function tryparsedef(::Type{T},s::AbstractString,default;kwarg...) where {T}
    x = tryparse(T,s;kwarg...)
    return isnothing(x) ? default : x
end
function filter_seq(seq::Sequence,tmin::Float64,tmax::Float64)
    t = cumsum(sum(s.DUR) for s in seq)
    ib = searchsortedfirst(t,tmin)
    ie = searchsortedlast(t,tmax)
    return (seq[ib:ie], t[ib:ie])
end

end