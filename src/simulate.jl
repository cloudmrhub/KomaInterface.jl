function banner_print(msg::Vararg{Any,N})::Nothing where {N}
    banner = '\n' * '=' ^ 60 * '\n'
    print(banner,msg...,banner)
    return nothing
end

# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

function load_rotation_matrix(json_path::String)
    data = JSON.parsefile(json_path)
    Rrows = @something(
        get(data,"R_body_to_seq",nothing),
        get(data,"rotation_matrix",nothing),
        get(data,"R",nothing),
        throw(ArgumentError("No rotation matrix found in \"$json_path\"."))
    )
    return stack(Float64.(row) for row in Rrows;dims=1)
end

const VecF64 = Vector{Float64}
function load_phantom(path::String)
    isfile(path) || throw(ArgumentError("unable to find file at \"$path\""))
    if endswith(path, ".h5") || endswith(path, ".hdf5")
        HDF5_AVAILABLE || error("HDF5.jl not installed. Run: `] add HDF5`")
        f = h5open(path,"r")
        data = Dict(k => collect(f[k]) for k in keys(f))
    else
        data = JSON.parsefile(path)
    end
    ρ = Vector{Float64}(collect(data["rho"]))
    mask = .!iszero.(ρ)
    keepat!(ρ,mask)
    obj = Phantom{Float64}(;
        name = get(data, "name", "phantom"),ρ,
        (s => keepat!(Vector{Float64}(collect(get(()->zero(ρ),data,l))),mask) for (s,l) in ((:x,"x"),(:y,"y"),(:z,"z"),(:T1,"t1"),(:T2,"t2"),(:T2s,"t2s"),(:Δw,"dw")))...
    )
    return obj
end

function rescale_fov!(seq::Sequence, scale::Vararg{Float64,N}) where {N}
    for j in axes(sec.GR,2),i in 1:N
        g = seq.GR[i,j]
        all(iszero,g.A) && continue
        seq.GR[i,j] = typeof(g)(g.A .* scale[i], g.T, g.rise, g.fall, g.delay)
    end
end

function apply_fov_rescaling!(seq::Sequence, fov_json_path::String)
    fov_data = JSON.parsefile(fov_json_path)
    seq_fov = Float64.(fov_data["seq_fov_m"])
    tgt_fov = Float64.(fov_data["target_fov_m"])
    N = minimum(length,(seq_fov,tgt_fov))
    scale = seq_fov[1:N] ./ tgt_fov[1:N]
    if any(!isapprox(one(eltype(scale))),scale)
        println("    FOV rescale: ", 1000seq_fov[1], "×", 1000seq_fov[2],
                " → ", 1000tgt_fov[1], "×", 1000tgt_fov[2], " mm")
        rescale_fov!(seq, scale...)
    end
end

function save_results(raw, directory::String, B0::Float64, R::AbstractMatrix, fov_rescaled::Bool,
                      phantom_path::String, n_spins::Int)
    Np = size(raw.profiles)[1]
    Nf = size(raw.profiles[1].data)[1]
    K = stack(ComplexF32.(p.data[:]) for p in raw.profiles;dims=1)

    mkpath(directory)
    filename = joinpath(directory, "k.npz")
    npzwrite(filename, K)

    R_nested = collect(eachrow(R))

    info = Dict(
        "version"          => "v3_batch_final",
        "KS"               => filename,
        "B0"               => B0,
        "Np"               => Np,
        "Nf"               => Nf,
        "n_spins"          => n_spins,
        "rotated"          => R != I,
        "rotation_matrix"  => R_nested,
        "fov_rescaled"     => fov_rescaled,
        "phantom_format"   => endswith(phantom_path, ".h5") ? "hdf5" : "json",
    )
    write(joinpath(directory, "info.json"),JSON.json(info))
    return Np, Nf
end

function main()
    aps = ArgParseSettings()
    @add_arg_table! aps begin
        "--B0"
            help = "static field strength [T]"
            arg_type = Float64
            required = true
        "--seq"
            help = "path of sequence file (.seq for pulseq, .mtrk for mtrk)"
            arg_type = String
            required = true
        "--batch"
            help = "path of json file for batch processing"
            arg_type = String
            required = true
        "--threads"
            help = "number of threads to use in KomaMRI"
            arg_type = Int
            default = Threads.nthreads()
        "--rotation"
            help = "path of json containing global 3x3 rotation matrix"
            arg_type = String
            default = ""
        "--fov"
            help = "path of json containing FOV specification"
            arg_type = String
            default = ""
        "--gpu"
            help = "enables use of GPU, and if able to use GPU library, sets number of threads to 1"
            action = :store_true
    end
    args = parse_args(ARGS,aps)

    B0         = args["B0"]
    seq_file   = args["seq"]
    batch_json = args["batch"]

    banner_print("KomaMRI Batch Simulation (KomaInterface.jl $(pkgversion(@__MODULE__)))")
    println("Sequence : ", seq_file)
    println("Batch    : ", batch_json)
    println("B0       : ", B0, " T")

    # GPU
    GPU::Bool = args["gpu"]
    if GPU
        println("GPU mode requested — loading CUDA backend...")
        try
            @eval using CUDA
            println("CUDA loaded: $(CUDA.name(CUDA.device()))")
        catch e
            @warn "CUDA.jl not available or no GPU found: $e — falling back to CPU"
            println(stderr, "\n[ERROR] GPU was requested but CUDA failed to load: $e")
            println(stderr, "  Install KomaMRIGPU.jl and ensure CUDA drivers are visible.\n")
            global GPU = false
        end
    end

    # Threads
    NT = ifelse(GPU,1,args["threads"])
    println("Threads  : ", NT)

    # Global default rotation / FOV rescaling
    global_rotation_json = nothing
    if !isempty(args["rotation"]) && args["rotation"] != "none"
        global_rotation_json = args["rotation"]
    end

    global_fov_json = nothing
    if !isempty(args["fov"]) && args["fov"] != "none"
        global_fov_json = args["fov"]
    end

    # ═══════════════════════════════════════════════════════════════════════════════
    # Load sequence ONCE
    # ═══════════════════════════════════════════════════════════════════════════════

    println("\nLoading sequence...")
    @time seq_original = KomaInterface.read_seq(seq_file)
    println("  Duration: ", sum(seq_original.DUR), " s")

    # Apply global FOV rescaling to the template (if any)
    fov_was_rescaled = !isnothing(global_fov_json) && isfile(global_fov_json)
    fov_was_rescaled && apply_fov_rescaling!(seq_original, global_fov_json)

    # Load global rotation (if any)
    R_global = Matrix{Float64}(I, 3, 3)
    if !isnothing(global_rotation_json) && isfile(global_rotation_json)
        R_global = load_rotation_matrix(global_rotation_json)
        println("Global rotation loaded from: ",global_rotation_json)
        display(R_global)
    end

    # ═══════════════════════════════════════════════════════════════════════════════
    # Simulation parameters
    # ═══════════════════════════════════════════════════════════════════════════════

    sim_params = KomaMRICore.default_sim_params()
    sim_params["gpu"]      = GPU
    sim_params["Nthreads"] = NT
    sys = Scanner(;B0)

    # ═══════════════════════════════════════════════════════════════════════════════
    # Load batch manifest and simulate
    # ═══════════════════════════════════════════════════════════════════════════════

    batch = JSON.parsefile(batch_json)
    n_entries = length(batch)
    banner_print("Batch: $n_entries entries")

    for (entry_idx, entry) in enumerate(batch)
        phantom_path = entry["phantom"]
        out_dir      = entry["output"]

        @info "Entry $entry_idx/$n_entries" Phantom=phantom_path Output=out_dir

        obj = load_phantom(phantom_path)
        println("  Spins   : ", length(obj.x))

        # Rotation
        rot_path = get(entry, "rotation", nothing)
        if isnothing(rot_path) && !isnothing(global_rotation_json)
            R = R_global
        elseif !isnothing(rot_path) && isfile(rot_path)
            R = load_rotation_matrix(rot_path)
        else
            R = Matrix{Float64}(I, 3, 3)
        end

        # Deep copy sequence (so per-entry FOV changes don't accumulate)
        seq = deepcopy(seq_original)

        entry_fov = get(entry, "fov_rescale", nothing)
        entry_fov_rescaled = !isnothing(entry_fov) && isfile(entry_fov)
        entry_fov_rescaled && apply_fov_rescaling!(seq, entry_fov)

        println("  Simulating...")
        @time raw = simulate(obj, seq, sys; sim_params)

        Np, Nf = save_results(raw, out_dir, B0, R, entry_fov_rescaled,
                            phantom_path, length(obj.x))
        println("  K-space: $Np × $Nf → $out_dir")
    end

    banner_print("Batch complete: $n_entries entries.")
end