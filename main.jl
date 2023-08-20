using POMDPs
using POMCPOW
using POMDPModels
using POMDPTools
using MCTS

include("LiPOMDP.jl")
include("utils.jl")


# POMCPOW Solver
pomdp = LiPOMDP() #always use continous and use POMCPOW obs widening params to control the discretization
up = LiBeliefUpdater(pomdp)
s0 = pomdp.init_state
b0 = initialize_belief(up, s0)

# test model
# a= action(planner, b0)
# rng = MersenneTwister(1)
# (sp, o, r) = gen(pomdp, s0, a, rng)
# b1 =update(up, b0, a, o) 


# POMCP Solver
println("POMCPOW Solver")
solver = POMCPOWSolver(tree_queries=1000, max_depth=20, estimate_value = 0., k_observation=0.1, alpha_observation=0.1) # Estimate value should fix the previous problem with action functions
planner = solve(solver, pomdp)
ap, info = action_info(planner, b0, tree_in_info=true)
tree = D3Tree(info[:tree], init_expand=1)
#inchrome(tree)

# random planner
random_planner = RandomPolicy(pomdp) #! should I seed this

println("==========start simulations==========")
# using manual simulation to extract the metrics of interest
println("simulating random policy")
sim_random = simulate_policy(pomdp, random_planner, up, b0, s0, max_steps=15)
println("simulating pomcpow policy")
sim_pomcpow = simulate_policy(pomdp, planner, up, b0, s0, max_steps=15)

println("""
    Evaluation (for 1 simulation)        
        POMCPOW: $(sim_pomcpow),  #! check why it always explores
        RANDOM: $(sim_random).
    """)