using POMDPs
using POMCPOW
using POMDPModels
using POMDPTools
using MCTS
using Serialization

Random.set_global_seed!(0)

include("LiPOMDP.jl")
include("utils.jl")
include("policies.jl")



# POMCPOW Solver
pomdp = LiPOMDP() #always use continous and use POMCPOW obs widening params to control the discretization
up = LiBeliefUpdater(pomdp)
s0 = pomdp.init_state
#b0 = initialize_belief(up, s0)

mdp = GenerativeBeliefMDP(pomdp, up)

# benchmarks
random_planner = RandomPolicy(pomdp)
strong_planner = EfficiencyPolicy(pomdp, [true, true, true, true])
robust_planner = EfficiencyPolicyWithUncertainty(pomdp, 1., [true, true, true, true])
eco_planner = EmissionAwarePolicy(pomdp, [true, true, true, true])


#MCTS Solver
mcts_solver = DPWSolver(
    depth=8,
    n_iterations = 100,
    estimate_value=RolloutEstimator(robust_planner, max_depth=11),
    enable_action_pw=false,
    enable_state_pw=true,
    k_state = 4.,
    alpha_state = 0.1,
)
mcts_planner = solve(mcts_solver, mdp)

# POMCP Solver
println("POMCPOW Solver")
solver = POMCPOWSolver(
    tree_queries=1000, 
    estimate_value = estimate_value,#RolloutEstimator(RandomPolicy(pomdp)), #estimate_value,
    k_observation=4., 
    alpha_observation=0.1, 
    max_depth=15, 
    enable_action_pw=false,
    init_N=10  
) # Estimate value should fix the previous problem with action functions
pomcpow_planner = solve(solver, pomdp)
# ap, info = action_info(pomcpow_planner, b0, tree_in_info=true)
# tree = D3Tree(info[:tree], init_expand=1)
# inchrome(tree)

# compare mcts and pomcpow
# simulate(RolloutSimulator(max_steps=11), mdp, mcts_planner, b0)
# simulate(RolloutSimulator(max_steps=11), pomdp, pomcpow_planner, up, b0)
# simulate(RolloutSimulator(max_steps=11), pomdp, eco_planner, up, b0)


planners = [random_planner, strong_planner, robust_planner, eco_planner, pomcpow_planner, mcts_planner] #

n_reps=20
max_steps=15

evaluate_policies(pomdp, planners, n_reps, max_steps)

#save planner
# println("pomcpow policy saved as planners/pomcpow_planner.jld2")
# serialize("planners/pomcpow_planner.jld2", pomcpow_planner)
