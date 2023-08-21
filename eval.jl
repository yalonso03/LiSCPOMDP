using POMDPs
using POMCPOW
using POMDPModels
using POMDPTools
using MCTS
using Serialization

include("LiPOMDP.jl")
include("utils.jl")
include("policies.jl")


# Load the POMDP problem, belief updater, and initial states/beliefs
pomdp = LiPOMDP()
up = LiBeliefUpdater(pomdp)
s0 = pomdp.init_state
b0 = initialize_belief(up, s0)

n_reps = 20
max_steps = 20

#planners
random_planner = RandomPolicy(pomdp)
strong_planner = EfficiencyPolicy(pomdp, [true, true, true, true])
robust_planner = EfficiencyPolicyWithUncertainty(pomdp, 5., [true, true, true, true])
eco_planner = EmissionAwarePolicy(pomdp, [true, true, true, true])
pomcpow_planner = deserialize("planners/pomcpow_planner.jld2")
planners = [random_planner, strong_planner, robust_planner, eco_planner, pomcpow_planner]

evaluate_policies(pomdp, planners, n_reps, max_steps)