# # Minimization of simple loss functions
#
# First we load the required packages.

using
    Distributions,
    LinearAlgebra,
    Random,
    Plots

using
    EnsembleKalmanProcesses.EnsembleKalmanProcessModule,
    EnsembleKalmanProcesses.ParameterDistributionStorage

# ## Loss function with single minimum
#
# Here, we minimize the loss function
# ```math
# G₁(u) = \|u - u_*\| ,
# ```
# where ``u`` is a 2-vector of parameters and ``u_*`` is given; here ``u_* = (-1, 1)``. 
u_min = [0.05, 0.05] #minimizer is 0,0
function G₁(u)

    if (u[1]>0) && (u[2] > 0)
        return sqrt((u[1]-u_min[1])^2 + (u[2]-u_min[2])^2)
    else 
        return NaN
    end
end

# We set the seed for pseudo-random number generator for reproducibility.
rng_seed = 41
Random.seed!(rng_seed)

# We set a stabilization level, which can aid the algorithm convergence
dim_output = 1
stabilization_level = 1e-3
Γ_stabilization = stabilization_level * Matrix(I, dim_output, dim_output) 

# The functional is positive so to minimize it we may set the target to be 0,
G_target  = [0]

# ### Prior distributions
#
# As we work with a Bayesian method, we define a prior. This will behave like an "initial guess"
# for the likely region of parameter space we expect the solution to live in.
prior_distributions = [Parameterized(Normal(0, 1)), Parameterized(Normal(0, 1))]
                
constraints = [[no_constraint()], [no_constraint()]]

parameter_names = ["u1", "u2"]

prior = ParameterDistribution(prior_distributions, constraints, parameter_names)

# ### Calibration
#
# We choose the number of ensemble members and the number of iterations of the algorithm
N_ensemble   = 40
N_iterations = 10
nothing # hide

# The initial ensemble is constructed by sampling the prior
initial_ensemble =
    EnsembleKalmanProcessModule.construct_initial_ensemble(prior, N_ensemble;
                                                           rng_seed=rng_seed)
# We then initialize the Ensemble Kalman Process algorithm, with the initial ensemble, the
# target, the stabilization and the process type (for EKI this is `Inversion`, initialized 
# with `Inversion()`). 
ensemble_kalman_process = 
    EnsembleKalmanProcessModule.EnsembleKalmanProcess(initial_ensemble, G_target,
                                                      Γ_stabilization, Inversion())

# Then we calibrate by *(i)* obtaining the parameters, *(ii)* calculate the loss function on
# the parameters (and concatenate), and last *(iii)* generate a new set of parameters using
# the model outputs:
for i in 1:N_iterations
    params_i = get_u_final(ensemble_kalman_process)

    g_ens = hcat([G₁(params_i[:, i]) for i in 1:N_ensemble]...)
    println("Iteration: ",i," number of particle failures: ", sum([1 for i = 1:size(g_ens,2) if isnan(g_ens[1,i])]), "/", size(g_ens,2))

    EnsembleKalmanProcessModule.update_ensemble!(ensemble_kalman_process, g_ens)

end

# and visualize the results:
u_init = get_u_prior(ensemble_kalman_process)
        
anim_unique_minimum = @animate for i in 1:N_iterations
    u_i = get_u(ensemble_kalman_process, i)
    g_i = get_g(ensemble_kalman_process, i)
    failed_particles = [i for i = 1:size(g_i,2) if isnan(g_i[1,i])]
    successful_particles = filter(x-> !(x in failed_particles), collect(1:size(g_i,2)))
    u_i_succ = u_i[:,successful_particles]
    u_i_fail = u_i[:,failed_particles]
    
    plot([u_min[1]], [u_min[2]],
          seriestype = :scatter,
         markershape = :star5,
          markersize = 11,
         markercolor = :green,
               label = "optimum u⋆"
         )
    
    plot!(u_i_succ[1, :], u_i_succ[2, :],
             seriestype = :scatter,
                  xlims = extrema(u_init[1, :]/2),
                  ylims = extrema(u_init[2, :]/2),
                 xlabel = "u₁",
                 ylabel = "u₂",
             markersize = 5,
            markeralpha = 0.6,
            markercolor = :blue,
                  label = "successes",
                  title = "EKI iteration = " * string(i)
          )
        plot!(u_i_fail[1, :], u_i_fail[2, :],
             seriestype = :scatter,
                  xlims = extrema(u_init[1, :]/2),
                  ylims = extrema(u_init[2, :]/2),
             markersize = 5,
            markeralpha = 0.6,
              markercolor = :red,
              markershape = :x,
                  label = "failures",
         )    

end
# The results show that the minimizer of ``G_1`` is ``u=u_*``. 

gif(anim_unique_minimum, "unique_minimum.gif", fps = 1) # hide

