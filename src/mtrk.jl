"""
    seq = read_seq(filename)

Returns the Sequence struct from a Pulseq file with `.seq` extension.

# Arguments
- `filename`: (`::String`) absolute or relative path of the sequence file `.seq`

# Returns
- `seq`: (`::Sequence`) Sequence struct

# Examples
```julia-repl
julia> seq_file = joinpath(dirname(pathof(KomaNYU)), "../examples/1.sequences/spiral.mtrk")

julia> seq = read_seq_mtrk(mtrk_file)

julia> plot_seq(seq)
```
"""

# Helper function: returns a function that gets the given key from a collection
_getindex(key::Key) where {Key} = Base.Fix2(Base.getindex,key)
# Helper function: returns a function that gets the given field from a struct
_getfield(s::Symbol) = Base.Fix2(Base.getfield,s)


# Truncates the amplitude or samples of a step to fit within [start, stop]
function pad(x::AbstractArray{T,N},p::NTuple{N,Int}) where {T,N}
    y = similar(x,p)
    y .= zero(T)
    return cat(x,zeros(p...);dims=Val(Base.OneTo(N)))
end
function pad!(x::AbstractVector{T},p::Int) where {T}
    iszero(p) && return x
    return append!(x,similar(x,p) .= zero(T))
end
padto!(x::AbstractVector,n::Int) = pad!(x,max(0,n-length(x)))
function padded!(X::Vararg{AbstractVector,N}) where {N}
    M = maximum(length,X)
    for x in X
        append!(x,zeros(M-length(x)))
    end
    return X
end
padded(X::Vararg{AbstractVector,N}) where {N} = padded!(copy.(X)...)

function truncate_step!(step::Dict{String,Any},start::Int,stop::Int)
    step_start = step["time"]
    step_stop = step["stop"]
    step_dur = step_stop-step_start
    
    if haskey(step,"samples")
        Na = step["samples"]
        isone(Na) && return step
        !iszero(step_dur % Na) && throw(DomainError((;step_duration=step_dur,samples=Na),"step duration must be a multiple of samples"))
        Δt = step_dur÷Na
    else
        amp = get(step,"amplitude",ComplexF64[])
        Na = length(amp)
        Δt = step_dur÷Na
    end
    istart =  1+max((start-step_start)÷Δt,0)
    istop  = Na+min(( stop-step_stop )÷Δt,0)
    if start < 1000
        @debug "Truncating with Na = $Na" start stop step_start step_stop Na istart istop
    end
    if haskey(step,"samples")
        step["samples"] = istop-istart+1
    elseif !isempty(amp)
        keepat!(amp,istart:istop)
    end
    step["time"] = start
    step["stop"] = stop
    return step
end

# Overloads for truncate_step! to accept tuples
truncate_step!(step,t::NTuple{2,Int}) = truncate_step!(step,t...)
truncate_step!(t::NTuple{2,Int}) = Base.Fix2(truncate_step!,t)

# Initial implementation by Anaïs Artiges (Anais.Artiges@nyulangone.org)
# Edited and improved by José E. Cruz Serrallés (Jose.CruzSerralles@nyulangone.org)

# Main function to read and process an mtrk sequence file
function getamp(steps,::Type{T}=Float64) where {T}
    isempty(steps) && return T[]
    isone(length(steps)) && return first(steps)["amplitude"]
    amps = padded(map(_getindex("amplitude"),steps)...)
    return sum(amps)
end
function read_seq_mtrk(filename)
    isfile(filename) || throw(ArgumentError("unable to find \"$filename\""))
    @info "Loading mtrk sequence $(basename(filename)) ..."

    ## read the SDL file
    raw_dict = JSON.parse(read(filename,String))
    dict = (;((Symbol(k),raw_dict[k]) for k in ("instructions","objects","arrays","equations","infos","settings"))...)
    haskey(dict.instructions,"main") || throw(ArgumentError("\"main\" block was not defined in \"instructions\" block"))
    
    # Artificially adding a "mark" at the end of any block that does not end with a "mark"
    @debug "Checking for missing \"mark\"s..."
    for block in values(dict.instructions)
        steps = block["steps"]
        if !isempty(last(steps)["action"]) && last(steps)["action"] == "submit" && steps[end-1]["action"]!="mark" && steps[end-1]["action"]!="run_block" && steps[end-1]["action"]!="loop"
            end_time = 0.0
            for event in steps
                if haskey(event,"time")
                    start_time = event["time"]
                    duration = dict.objects[event["object"]]["duration"]
                    end_time = max(end_time,start_time + duration)
                end
            end
            push!(steps,Dict{String,Any}("action" => "mark","time" => end_time))
            # Add a mark event with its time set to the duration of the previous block. 
        end
    end

   # assign mark if missing
    @debug "Assigning missing \"mark\"s..."
    for block in values(dict.instructions)
        steps = block["steps"]
        any(step["action"] ∈ (("loop","run_block")) for step in steps) && continue
        filter!(∈(("rf","grad","adc","mark"))∘_getindex("action"),steps)
        if last(steps)["action"] != "mark"
            push!(steps,Dict{String,Any}("action" => "mark","time" => maximum(s -> step["time"]+step["duration"],steps)))
        end
    end

    ## Interpreting time equations
    for block in values(dict.instructions)
        steps = block["steps"]
        for step in steps
            if haskey(step,"time") && typeof(step["time"]) != Int
                equation_name = step["time"]["equation"]
                # Replace set(<variable>) with its value from dict.settings
                equation_str = string(dict.equations[equation_name]["equation"])
                pattern = r"set\((\w+)\)"
                replaced_eq = equation_str
                for m in eachmatch(pattern, equation_str)
                    varname = m.captures[1]
                    value = string(dict.settings[varname])
                    replaced_eq = replace(replaced_eq, m.match => value)
                end
                step["time"] = Int(eval(Meta.parse(replaced_eq)))
            end
        end
    end

    # Flatten instructions: unroll loops and run_block actions into a flat step list
    @debug "Unrolling instructions..."
    steps = deepcopy(dict.instructions["main"]["steps"])
    idx = 1
    while idx ≤ length(steps)
        step = steps[idx]
        action = step["action"]
        if action == "run_block"
            block_steps = deepcopy(dict.instructions[step["block"]]["steps"])
            if haskey(step,"counters")
                # Inline the steps from the referenced block
                block_counter = step["counters"]
                for n in eachindex(block_steps)
                    block_steps[n]["counters"] = block_counter
                end
            end
            splice!(steps,idx,block_steps)
        elseif action == "loop"
            # Unroll the loop into repeated steps, updating counters
            loop_range = step["range"]
            loop_id = step["counter"]
            loop_counters = get(step,"counters",Pair{Int,Int}[])
            loop_steps = [deepcopy(s) for s in step["steps"],_ in Base.OneTo(loop_range)]
            for n in 1:loop_range
                new_counters = copy(loop_counters)
                pushfirst!(new_counters,loop_id => n-1)
                for m in axes(loop_steps,1)
                    loop_steps[m,n]["counters"] = new_counters
                end
            end
            splice!(steps,idx,view(loop_steps,:))
        elseif !haskey(step,"time") # prune steps without a time (such as init)
            deleteat!(steps,idx)
        else
            idx += 1
        end
    end

    # update times to reflect global timing
    offset = zero(first(steps)["time"])
    for step in steps
        new_time = step["time"]+offset
        step["time"] = new_time
        if step["action"] == "mark"
            offset = new_time
        end
    end


    # keep relevant events and sort by start time
    filter!(∈(("rf","grad","adc"))∘_getindex("action"),steps)
    sort!(steps,by=_getindex("time"))

    # get amplitudes, durations, and stop times
    @debug "Processing individual steps..."
    gammabar = 42.58e6 # Hz/T
    gamma = 2π*gammabar
    for step in steps
        action = step["action"]
        obj = dict.objects[step["object"]]
        step["duration"] = obj["duration"]
        step["stop"] = step["time"]+step["duration"]
        if action == "rf"
            # Calculate RF pulse amplitude array
            data = map(Float64,dict.arrays[obj["array"]]["data"])
            @views mag, phase = data[1:2:end], data[2:2:end]
            dt = 1e-6*step["duration"]/length(mag)
            amplitude = deg2rad(obj["flipangle"]).*(mag./(sum(mag)*gamma*dt)).*cis.(phase)
            step["amplitude"] = amplitude
        elseif action == "grad"
            # Calculate gradient amplitude array, possibly evaluating equations
            if haskey(step,"amplitude")
                amplitude = step["amplitude"]
                if amplitude == "flip"
                    amplitude = -obj["amplitude"]
                elseif haskey(amplitude,"type") && amplitude["type"] == "equation"
                    haskey(step,"counters") || throw(KeyError("counters"))
                    eq = dict.equations[amplitude["equation"]]["equation"]
                    eq = replace(eq, ("ctr($id)" => string(idx) for (id,idx) in step["counters"])...)
                    amplitude = eval(Meta.parse(eq))
                end
            else
                amplitude = obj["amplitude"]
            end
            array = (1e-3*amplitude).*dict.arrays[obj["array"]]["data"]
            step["action"] = step["axis"]
            step["amplitude"] = array

            Δt = step["duration"]÷length(array)
            if step["time"] ≤ 10
                @debug "" step["duration"] length(array) Δt
            end
            iszero(Δt) && throw(ArgumentError("gradient array length exceeds step length (duration in μs)"))
        elseif action == "adc"
            # Set number of ADC samples
            step["samples"] = obj["samples"]
        end
    end

    # bin steps into overlapping and non-overlapping groups
    @debug "Binning into groups and generating sequence..." numsteps=length(steps)
    all_times = unique!(sort!(reshape(stack([step["time"];step["stop"]] for step in steps),:)))
    !isempty(all_times) && !iszero(first(all_times)) && pushfirst!(all_times,zero(eltype(all_times)))
    @debug "Sorted, unique times" extrema(all_times)
    
    # Initialize empty sequence containers for each channel
    seq = StructVector{@NamedTuple{read::Grad,phase::Grad,slice::Grad,rf::RF,adc::ADC,dur::Float64}}(undef,0)
    sizehint!(seq,length(all_times)-1)
    @debug "Iterating over all event times"
    i = firstindex(all_times)
    actions = ("read","phase","slice","rf","adc")
    while i < lastindex(all_times)
        start = all_times[i]
        filter!(>(start)∘_getindex("stop"),steps)
        j = i
        counts = (0,0,0,0,0)
        group = Vector{Any}[]
        while j < lastindex(all_times)
            stop = all_times[j]
            upper = searchsortedlast(steps,Dict{String,Any}("time"=>stop-1);by=_getindex("time"))
            lower = _findfirst(≤(start)∘_getindex("time"),view(steps,Base.OneTo(upper)))
            group = deepcopy(view(steps,lower:upper))
            counts = Tuple(count(==(l)∘_getindex("action"),group) for l in actions)
            if any(>(1),counts)
                break
            else
                j += 1
            end
        end
        while j ≥ i && any(>(1),counts)
            j -= 1
            stop = all_times[j]
            upper = searchsortedlast(steps,Dict{String,Any}("time"=>stop-1);by=_getindex("time"))
            lower = _findfirst(≤(start)∘_getindex("time"),view(steps,Base.OneTo(upper)))
            group = deepcopy(view(steps,lower:upper))
            counts = Tuple(count(==(l)∘_getindex("action"),group) for l in actions)
        end
        stop = all_times[j]
        if i < 50
        @debug "Processing events for t ∈ [$start,$stop) μs" i:j counts
        end
        channels = Tuple(filter(==(l)∘_getindex("action"),group) for l in actions)
        A = (getamp.(channels[1:3])..., getamp(channels[4],ComplexF64), isempty(channels[5]) ? 0 : only(channels[5])["samples"])
        durs = map(c->isempty(c) ? stop-start : only(c)["duration"],channels)
        delays = map(c->isempty(c) ? 0 : only(c)["time"]-start,channels)
        # If any channel is active, create a new sequence step; otherwise, accumulate delay
        dur_us = stop-start
        dur = dur_us*1e-6
        if any(!iszero,A)
            new_step = (;
                 read=Grad(A[1],1e-6durs[1]),
                phase=Grad(A[2],1e-6durs[2]),
                slice=Grad(A[3],1e-6durs[3]),
                   rf=  RF(A[4],1e-6durs[4]),
                  adc= ADC(A[5],1e-6durs[5]),
                  dur
            )
            if isempty(seq)
                foreach((k,delay)->setfield!(getfield(new_step,k),:delay,1e-6*delay),keys(A),delays)
            end
            push!(seq,new_step)
        end
        i = max(j,i+1)
    end
    grad = stack((seq.read,seq.phase,seq.slice);dims=1)
    sequence = Sequence(grad,reshape(seq.rf,1,:),seq.adc,seq.dur)
    
    ## Fill sequence header with metadata from the file
    fov = dict.infos["fov"] * 1e-3
    sliceThickness = dict.objects["rf_excitation"]["thickness"] * 1e-3
    sequence.DEF["FOV"] = [fov, fov, sliceThickness]
    sequence.DEF["Name"] = dict.infos["seqstring"]
    sequence.DEF["Nz"] = dict.infos["slices"]
    sequence.DEF["Nx"] = dict.infos["pelines"]
    sequence.DEF["Ny"] = dict.infos["pelines"]
    sequence.DEF["FileName"] = filename
    sequence.DEF["GradientRasterTime"] = 1.0e-5
    sequence.DEF["AdcRasterTime"] = 3.0e-5
    sequence.DEF["RadiofrequencyRasterTime"] = 2.0e-5
    sequence.DEF["BlockDurationRaster"] = 1.0e-5
    sequence.DEF["TotalDuration"] = sum(sequence.DUR)

    ## Print sequence info and return
    @info "$sequence"

    return sequence
end

# Helper: findfirst, but throw if not found
function _findfirstorthrow(f::Function,itr::A) where {A}
    i = findfirst(f,itr)
    isnothing(i) && throw(ArgumentError("findfirst returned nothing"))
    return i
end


function _findfirst(f::Function,itr::A) where {A}
    i = findfirst(f,itr)
    return isnothing(i) ? nextind(itr,lastindex(itr)) : i
end
