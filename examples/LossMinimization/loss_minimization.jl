using Distributions
using LinearAlgebra
using Random
using Plots

struct EnsembleKalmanInversion{FT}
    obs_mean::Vector{FT} #vector of the observed vector size [N_obs]
    obs_noise_cov::Array{FT,2} #covariance matrix of the observational noise, of size [N_obs × N_obs]
end

# Update follows eqns. (4) and (5) of Schillings and Stuart (2017)
function update_ensemble(
    ekp::EnsembleKalmanInversion,
    u::AbstractArray,
    g::AbstractArray;
    cov_threshold = 0.01,
    Δt = 1.0,
    deterministic_forward_map=true
)
    #catch works when g non-square
    # if !(size(g)[2] == ekp.N_ens) 
    #      throw(DimensionMismatch("ensemble size in EnsembleKalmanProcess and g do not match, try transposing g or check ensemble size"))
    # end

    # u: N_par × N_ens, g: N_obs × N_ens
    N_ens = size(u, 2)
    N_obs = size(g, 1)

    cov_init = cov(u, dims=2)
    cov_ug = cov(u, g, dims = 2, corrected=false) # [N_par × N_obs]
    cov_gg = cov(g, g, dims=2, corrected=false) # [N_obs × N_obs]

    # Scale noise using Δt
    scaled_obs_noise_cov = ekp.obs_noise_cov / Δt # [N_obs × N_obs]
    noise = rand(MvNormal(zeros(N_obs), scaled_obs_noise_cov), N_ens)

    # Add obs_mean (N_obs) to each column of noise (N_obs × N_ens) if
    # G is deterministic
    y = deterministic_forward_map ? (ekp.obs_mean .+ noise) : (ekp.obs_mean .+ zero(noise))

    # N_obs × N_obs \ [N_obs × N_ens]
    # --> tmp is [N_obs × N_ens]
    tmp = (cov_gg + scaled_obs_noise_cov) \ (y - g)
    u_updated = u + cov_ug * tmp # [N_par × N_ens]

    # Calculate error
    mean_g = dropdims(mean(g, dims=2), dims=2)
    diff = ekp.obs_mean - mean_g
    err = dot(diff, ekp.obs_noise_cov \ diff)

    # Check convergence
    # cov_new = cov(get_u_final(ekp), dims=2)
    # cov_ratio = det(cov_new) / det(cov_init)
    # if cov_ratio < cov_threshold
    #     @warn string("New ensemble covariance determinant is less than ",
    #                  cov_threshold, " times its previous value.",
    #                  "\nConsider reducing the EK time step.")
    # end

    return u_updated, err
end

let
    # Seed for pseudo-random number generator for reproducibility
    rng_seed = 41
    Random.seed!(rng_seed)

    # Set up observational noise 
    n_obs = 1 # Number of synthetic observations from G(u)
    noise_level = 1e-8 # Defining the observation noise level
    Γy = noise_level * Matrix(I, n_obs, n_obs) # Independent noise for synthetic observations       
    noise = MvNormal(zeros(n_obs), Γy)

    # Set up the loss function (unique minimum)
    function G(u)
        return [sqrt((u[1]-1)^2 + (u[2]+1)^2)]
    end
    u_star = [1.0, -1.0] # Loss Function Minimum
    y_obs  = G(u_star) .+ rand(noise) 

    # Set up prior
    prior_mean = zeros(2)
    prior_cov = Matrix(I, length(prior_mean), length(prior_mean))
    prior = MvNormal(prior_mean, prior_cov)

    # Set up optimizer
    ekiobj = EnsembleKalmanInversion{Float64}(y_obs, Γy)

    # Do optimization loop
    N_ens = 50  # number of ensemble members
    N_iter = 20 # number of EKI iterations
    ensemble = rand(prior, N_ens)
    storage_g = []
    storage_u = [copy(ensemble)]
    storage_e = []
    for i in 1:N_iter
        evaluations = hcat(map(G, eachcol(ensemble))...)
        ensemble, err = update_ensemble(ekiobj, ensemble, evaluations)
        push!(storage_u, copy(ensemble))
        push!(storage_g, copy(evaluations))
        push!(storage_e, err)
    end

    # Do plotting
    u_init = storage_u[1]
    u1_min = minimum(minimum(u[1,:]) for u in storage_u)
    u1_max = maximum(maximum(u[1,:]) for u in storage_u)
    u2_min = minimum(minimum(u[2,:]) for u in storage_u)
    u2_max = maximum(maximum(u[2,:]) for u in storage_u)
    xlims = (u1_min, u1_max)
    ylims = (u2_min, u2_max)
    for (i, u) in enumerate(storage_u)
        p = plot(u[1,:], u[2,:], seriestype=:scatter, xlims = xlims, ylims = ylims)
        plot!([u_star[1]], xaxis="u1", yaxis="u2", seriestype="vline",
            linestyle=:dash, linecolor=:red, label = false,
            title = "EKI iteration = " * string(i)
            )
        plot!([u_star[2]], seriestype="hline", linestyle=:dash, linecolor=:red, label = "optimum")
        display(p)
        sleep(0.1)
    end
end