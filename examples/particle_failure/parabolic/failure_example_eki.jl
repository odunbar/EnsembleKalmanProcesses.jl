include("failure_model.jl")

###
###  Calibrate: Ensemble Kalman Inversion
###
N_ens = 50 
N_iter = 20 # number of UKI iterations
initial_par = construct_initial_ensemble(priors, N_ens; rng_seed)
ekiobj = EnsembleKalmanProcess(initial_par, truth_sample, truth.obs_noise_cov, Inversion(), Δt = 0.1)


# UKI iterations
err = zeros(N_iter)
for i in 1:N_iter

    params_i = get_u_final(ekiobj)

    g_ens = run_G_ensemble(params_i)


    # analysis step 
    EnsembleKalmanProcessModule.update_ensemble!(ekiobj, g_ens)

    err[i] = get_error(ekiobj)[end] #mean((params_true - mean(params_i,dims=2)).^2)
    println(
        "Iteration: " *
        string(i) *
        ", Error: " *
        string(err[i]),
    )
end



println("True parameters: ")
println(params_true)

println("\nEKI results:")
println(mean(get_u_final(ekiobj), dims = 2))


####
N_θ = length(param_names)
θ_mean_arr = zeros(Float64, (N_θ, N_iter + 1))
θθ_std_arr = zeros(Float64, (N_θ, N_iter + 1))
for i in 1:(N_iter + 1)
    θ_mean_arr[:, i] = mean(get_u(ekiobj, i), dims = 2)
    θ_cov = cov(get_u(ekiobj, i), dims = 2)
    for j in 1:N_θ
        θθ_std_arr[j, i] = sqrt(θ_cov[j, j])
    end
end

ites = Array(LinRange(1, N_iter + 1, N_iter + 1))
plot(ites, grid = false, θ_mean_arr[1, :], yerror = 3.0 * θθ_std_arr[1, :], label = "θ1")
plot!(ites, fill(params_true[1], N_iter + 1), linestyle = :dash, linecolor = :grey, label = nothing)

plot!(ites, grid = false, θ_mean_arr[2, :], yerror = 3.0 * θθ_std_arr[2, :], label = "θ2", xaxis = "Iterations")
plot!(ites, fill(params_true[2], N_iter + 1), linestyle = :dash, linecolor = :grey, label = nothing)


#plots
gr(size = (600, 600))
xlims = extrema(append!(hcat([get_u(ekiobj, i) for i=1:N_iter+1]...)[1,:], params_true[1]))
ylims = extrema(append!(hcat([get_u(ekiobj, i) for i=1:N_iter+1]...)[2,:], params_true[2]))
u_init = get_u_prior(ekiobj)
for i in 1:N_iter
    u_i = get_u(ekiobj, i)

    p1 = plot(u_i[1, :], u_i[2, :], seriestype = :scatter, xlims = xlims, ylims = ylims)
    plot!(
        p1,
        [params_true[1]],
        xaxis = "θ1",
        yaxis = "θ2",
        seriestype = "vline",
        linestyle = :dash,
        linecolor = :red,
        label = false,
        title = "EKI iteration = " * string(i),
    )
    plot!(p1, [params_true[2]], seriestype = "hline", linestyle = :dash, linecolor = :red, label = "optimum")
    display(p1)
    sleep(0.5)
end


