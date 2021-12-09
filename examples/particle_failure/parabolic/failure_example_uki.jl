include("failure_model.jl")

N_iter = 20 # number of UKI iterations
# need to choose regularization factor α ∈ (0,1],  
# when you have enough observation data α=1: no regularization
α_reg = 1.0
# update_freq 1 : approximate posterior covariance matrix with an uninformative prior
#             0 : weighted average between posterior covariance matrix with an uninformative prior and prior
update_freq = 0

process = Unscented(prior_mean, prior_cov, α_reg, update_freq)
ukiobj = EnsembleKalmanProcessModule.EnsembleKalmanProcess(truth_sample, truth.obs_noise_cov, process)


# UKI iterations
err = zeros(N_iter)
for i in 1:N_iter

    params_i = get_u_final(ukiobj)

    g_ens = run_G_ensemble(params_i)


    # analysis step 
    EnsembleKalmanProcessModule.update_ensemble!(ukiobj, g_ens)

    err[i] = get_error(ukiobj)[end] #mean((params_true - mean(params_i,dims=2)).^2)
    println(
        "Iteration: " *
        string(i) *
        ", Error: " *
        string(err[i]) *
        " norm(Cov): " *
        string(norm(ukiobj.process.uu_cov[i])),
    )
end



println("True parameters: ")
println(params_true)

println("\nUKI results:")
println(get_u_mean_final(ukiobj))





####
θ_mean_arr = hcat(ukiobj.process.u_mean...)
N_θ = length(ukiobj.process.u_mean[1])
θθ_std_arr = zeros(Float64, (N_θ, N_iter + 1))
for i in 1:(N_iter + 1)
    for j in 1:N_θ
        θθ_std_arr[j, i] = sqrt(ukiobj.process.uu_cov[i][j, j])
    end
end

ites = Array(LinRange(1, N_iter + 1, N_iter + 1))
plot(ites, grid = false, θ_mean_arr[1, :], yerror = 3.0 * θθ_std_arr[1, :], label = "θ1")
plot!(ites, fill(params_true[1], N_iter + 1), linestyle = :dash, linecolor = :grey, label = nothing)

plot!(ites, grid = false, θ_mean_arr[2, :], yerror = 3.0 * θθ_std_arr[2, :], label = "θ2", xaxis = "Iterations")
plot!(ites, fill(params_true[2], N_iter + 1), linestyle = :dash, linecolor = :grey, label = nothing)



#plots
gr(size = (600, 600))
dθ = 1.0
for i in 1:N_iter
    θ_mean, θθ_cov = ukiobj.process.u_mean[i], ukiobj.process.uu_cov[i]
    xx = Array(LinRange(params_true[1] - dθ, params_true[1] + dθ, 100))
    yy = Array(LinRange(params_true[2] - dθ, params_true[2] + dθ, 100))
    xx, yy, Z = Gaussian_2d(θ_mean, θθ_cov, 100, 100, xx = xx, yy = yy)

    p = contour(xx, yy, Z)
    plot!(
        [params_true[1]],
        xaxis = "θ₁", 
        yaxis = "θ₂", 
        seriestype = "vline",
        linestyle = :dash,
        linecolor = :red,
        label = false,
        title = "UKI iteration = " * string(i),
    )
    plot!([params_true[2]], seriestype = "hline", linestyle = :dash, linecolor = :red, label = "optimum")
    display(p)
    sleep(0.5)
end
