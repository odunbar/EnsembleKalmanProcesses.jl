# Import modules
using Distributions  # probability distributions and associated functions
using LinearAlgebra
using StatsPlots

using Random
using JLD2

using Plots
# CES 
using EnsembleKalmanProcesses.EnsembleKalmanProcessModule
using EnsembleKalmanProcesses.Observations
using EnsembleKalmanProcesses.ParameterDistributionStorage

rng_seed = 4137
Random.seed!(rng_seed)

# Output figure save directory
homedir = pwd()
println(homedir)
figure_save_directory = homedir * "/output/"
data_save_directory = homedir * "/output/"
if ~isdir(figure_save_directory)
    mkdir(figure_save_directory)
end
if ~isdir(data_save_directory)
    mkdir(data_save_directory)
end


###
###  model settings
###


###  Generate (artificial) truth samples
###  Note: The observables y are related to the parameters θ by:
###        y = G(θ) + η
###
# G(u) = [u[1] ; u[2]]
G(u) = [sqrt(u[1] - u[2]^2) ; u[2]]
θ1_true, θ2_true = 2.0, 1.0
prior_mean = [θ1_true + 1.0; θ2_true + 0.5]
prior_cov = Array(Diagonal(fill(0.1^2,  length(prior_mean))))
# prior_cov = Array(Diagonal(fill(1.0^2,  length(prior_mean))))




params_true = [θ1_true, θ2_true]
param_names = ["θ1", "θ2"]
n_param = length(param_names)
params_true = reshape(params_true, (n_param, 1))

println(n_param)
println(params_true)




# prior_stds = [2.0, 0.5*A_true]

prior_distributions = [Parameterized(Normal(prior_mean[1], sqrt(prior_cov[1,1]))), Parameterized(Normal(prior_mean[2], sqrt(prior_cov[2,2])))]
constraints = [[no_constraint()], [no_constraint()]]


priors = ParameterDistribution(prior_distributions, constraints, param_names)


###
###  Define the data from which we want to learn the parameters
###
data_names = ["y0", "y1"]


function run_G_ensemble(params)
    N_ensemble = size(params, 2)
    return hcat([G(params[:, i]) for i in 1:N_ensemble]...)
end
# Lorenz forward
# Input: params: [N_params, N_ens]
# Output: gt: [N_data, N_ens]
# Dropdims of the output since the forward model is only being run with N_ens=1 
# corresponding to the truth construction
n_samples = 100
G_t = G(params_true)
y_t = zeros(length(G_t), n_samples)
Γy = Array(Diagonal(fill(0.1^2,  length(G_t))))

# Add noise
for i in 1:n_samples
    y_t[:, i] = G_t .+ rand(MvNormal(zeros(size(G_t)), Γy))
end

# Construct observation object
truth = Observations.Obs(y_t, Γy, data_names)
truth_sample = truth.mean