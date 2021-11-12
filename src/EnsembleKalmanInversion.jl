#Ensemble Kalman Inversion: specific structures and function definitions

"""
    Inversion <: Process

An ensemble Kalman Inversion process
"""
struct Inversion <: Process end

"""
    find_ekp_stepsize(ekp::EnsembleKalmanProcess{FT, IT, Inversion}, g::Array{FT, 2}; cov_threshold::FT=0.01) where {FT}

Find largest stepsize for the EK solver that leads to a reduction of the determinant of the sample
covariance matrix no greater than cov_threshold. 
"""
function find_ekp_stepsize(
    ekp::EnsembleKalmanProcess{FT, IT, Inversion},
    g::Array{FT, 2};
    cov_threshold::FT = 0.01,
) where {FT, IT}
    accept_stepsize = false
    if !isempty(ekp.Δt)
        Δt = deepcopy(ekp.Δt[end])
    else
        Δt = FT(1)
    end
    # final_params [N_par × N_ens]
    cov_init = cov(get_u_final(ekp), dims = 2)
    while accept_stepsize == false
        ekp_copy = deepcopy(ekp)
        update_ensemble!(ekp_copy, g, Δt_new = Δt)
        cov_new = cov(get_u_final(ekp_copy), dims = 2)
        if det(cov_new) > cov_threshold * det(cov_init)
            accept_stepsize = true
        else
            Δt = Δt / 2
        end
    end

    return Δt

end

"""
    update_ensemble!(ekp::EnsembleKalmanProcess{FT, IT, <:Inversion}, g::Array{FT,2} cov_threshold::FT=0.01, Δt_new=nothing) where {FT, IT}

Updates the ensemble according to which type of Process we have. Model outputs `g` need to be a `N_obs × N_ens` array (i.e data are columms).
"""
function update_ensemble!(
    ekp::EnsembleKalmanProcess{FT, IT, Inversion},
    g::Array{FT, 2};
    cov_threshold::FT = 0.01,
    Δt_new = nothing,
    deterministic_forward_map = true,
) where {FT, IT}

    # Update follows eqns. (4) and (5) of Schillings and Stuart (2017)

    #catch works when g non-square
    if !(size(g)[2] == ekp.N_ens)
        throw(DimensionMismatch("ensemble size in EnsembleKalmanProcess and g do not match, try transposing g or check ensemble size"))
    end

    # u: N_par × N_ens 
    # g: N_obs × N_ens
    u = copy(get_u_final(ekp))
    u_old = copy(u)
    N_obs = size(g, 1)
   
    cov_init = cov(u, dims=2)
    cov_ug = cov(u,g, dims = 2, corrected=false) # [N_par × N_obs]
    cov_gg = cov(g,g, dims = 2, corrected=false) # [N_obs × N_obs]

    if !isnothing(Δt_new)
        push!(ekp.Δt, Δt_new)
    elseif isnothing(Δt_new) && isempty(ekp.Δt)
        push!(ekp.Δt, FT(1))
    else
        push!(ekp.Δt, ekp.Δt[end])
    end
    
    # Scale noise using Δt
    scaled_obs_noise_cov = ekp.obs_noise_cov / ekp.Δt[end]
    noise = rand(MvNormal(zeros(N_obs), scaled_obs_noise_cov), ekp.N_ens)

    # Add obs_mean (N_obs) to each column of noise (N_obs × N_ens) if
    # G is deterministic
    y = deterministic_forward_map ? (ekp.obs_mean .+ noise) : (ekp.obs_mean .+ zero(noise))

    #batch particle failures
    failed_particles = [i for i = 1:size(g,2) if isnan(g[1,i])]
    if length(failed_particles) == 0
        # N_obs × N_obs \ [N_obs × N_ens]
        # --> tmp is [N_obs × N_ens]
        tmp = (cov_gg + scaled_obs_noise_cov) \ (y - g)
        u += (cov_ug * tmp) # [N_par × N_ens]
    else

        successful_particles = filter(x-> !(x in failed_particles), collect(1:size(g,2)))
        u_succ = u[:,successful_particles]
        g_succ = g[:,successful_particles]
        y_succ = y[:,successful_particles]

        if length(failed_particles) > length(successful_particles)
            @warn string("More than 50% of runs produced NaN.",
                         "\nIterating... \nbut consider increasing model stability.",
                         "\nThis will effect optimization result.")
        end
        #update successful ones
        cov_ug_succ = cov(u_succ, g_succ, dims=2, corrected=false)
        cov_gg_succ = cov(g_succ, g_succ, dims=2, corrected=false)
        
        tmp = (cov_gg_succ + scaled_obs_noise_cov) \ (y_succ - g_succ)
        
        u[:,successful_particles] += (cov_ug_succ * tmp) # [N_par × N_ens]
        
        cov_u_new = cov(u[:,successful_particles],u[:,successful_particles], dims=2)
        inv_cov_change = cov_init \ cov_u_new 
        
        u[:,failed_particles] =  mean(u[:,successful_particles]) .+ inv_cov_change*(u_old[:,failed_particles] .- mean(u_old,dims=2))
        
    end
        
    # store new parameters (and model outputs)
    push!(ekp.u, DataContainer(u, data_are_columns = true))
    push!(ekp.g, DataContainer(g, data_are_columns = true))

    # Store error
    compute_error!(ekp)

    # Check convergence
    cov_new = cov(get_u_final(ekp), dims = 2)
    cov_ratio = det(cov_new) / det(cov_init)
    if cov_ratio < cov_threshold
        @warn string(
            "New ensemble covariance determinant is less than ",
            cov_threshold,
            " times its previous value.",
            "\nConsider reducing the EK time step.",
        )
    end
end
