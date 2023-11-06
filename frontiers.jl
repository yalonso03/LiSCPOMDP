#=
Modeling the US Path to Lithium Self Sufficiency Using POMDPs
Autumn 2023
Yasmine Alonso, Mansur Arief, Anthony Corso, Jef Caers, and Mykel Kochenderfer

File: frontiers.jl
----------------
This file contains the code that runs and evaluates all of our policies, producing frontiers 
=#
using Random
using POMDPs
using POMCPOW
using POMDPModels
using POMDPTools
using MCTS
using Serialization
using Plots

Random.set_global_seed!(0)

include("LiPOMDP.jl")
include("utils.jl")
include("policies.jl")


# Initializing the POMDP, Belief Updater, and initial state, as well as the MDP version of the POMDP for MCTS
pomdps = [LiPOMDP() for i in 1:10]  # Array of 10 LiPOMDPs
for i in 1:10
    pomdps[i].obj_weights = [0, i / 10, 1 - i / 10]  # [r1, r2, r3] for time and volume, volume, CO2
end
ups = [LiBeliefUpdater(pomdps[i]) for i in 1:10]  # Updater for each of the pomdps
s0 = pomdps[1].init_state
#b0 = initialize_belief(up, s0)
mdps = [GenerativeBeliefMDP(pomdps[i], ups[i]) for i in 1:10]  # MDP version of each of the POMDPs

#println([pomdp.obj_weights for pomdp in pomdps])
# benchmark planners (from policies.jl)
random_planner = RandomPolicy(pomdps[1]) #! Just one time
strong_planner = EfficiencyPolicy(pomdps[1], [true, true, true, true]) #! Just one time
robust_planner = EfficiencyPolicyWithUncertainty(pomdps[1], 1., [true, true, true, true]) #! Just one time
eco_planner = EmissionAwarePolicy(pomdps[1], [true, true, true, true]) #! Just one time


# #MCTS Solver -- uses mdp version of pomdp #!10 times
mcts_solver = DPWSolver(
    depth=8,
    n_iterations = 100,
    estimate_value=RolloutEstimator(robust_planner, max_depth=100),
    enable_action_pw=false,
    enable_state_pw=true,
    k_state = 4.,
    alpha_state = 0.1,
)
mcts_planners = [solve(mcts_solver, mdp) for mdp in mdps]  # Will have 10 MCTS planners


# POMCPOW Solver #! 10 times
#println("POMCPOW Solver")
solver = POMCPOWSolver(
    tree_queries=1000, 
    estimate_value = estimate_value,#RolloutEstimator(RandomPolicy(pomdp)), #estimate_value,
    k_observation=4., 
    alpha_observation=0.1, 
    max_depth=15, 
    enable_action_pw=false,
    init_N=10  
) # Estimate value should fix the previous problem with action functions
pomcpow_planners = [solve(solver, pomdp) for pomdp in pomdps]

simple_planners = [random_planner, strong_planner, robust_planner, eco_planner] # just the singletons


n_reps=20
max_steps=15

# # Compares all the policies and prints out relevant information
# # Read in utils.jl
# println("************** Evaluating the single policies **************")
# evaluate_policies(pomdps[1], simple_planners, n_reps, max_steps)

# # Evaluate the 10 pomcpow planners
# println("************** Evaluating 10 POMCPOW Planners **************")
# for i in 1:10
#     evaluate_policy(pomdps[i], pomcpow_planners[i], n_reps, max_steps)
#     print("--finished evaluating POMCPOW Policy ", i)
# end

# # Evaluate the 10 MCTSplanners
# println("************** Evaluating 10 MCTS Planners **************")
# for i in 1:10
#     evaluate_policy(pomdps[i], mcts_planners[i], n_reps, max_steps)
#     print("--finished evaluating MCTS Policy ", i)
# end

