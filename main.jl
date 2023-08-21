using POMDPs
using POMCPOW
using POMDPModels
using POMDPTools
using MCTS
using Serialization


include("LiPOMDP.jl")
include("utils.jl")
include("policies.jl")


# POMCPOW Solver
pomdp = LiPOMDP() #always use continous and use POMCPOW obs widening params to control the discretization
up = LiBeliefUpdater(pomdp)
s0 = pomdp.init_state
b0 = initialize_belief(up, s0)


# POMCP Solver
println("POMCPOW Solver")
solver = POMCPOWSolver(
    tree_queries=2000, 
    estimate_value = estimate_value, 
    k_observation=3., 
    alpha_observation=0.1, 
    max_depth=50, 
    enable_action_pw=false,
    init_N=10  
) # Estimate value should fix the previous problem with action functions


pomcpow_planner = solve(solver, pomdp)
ap, info = action_info(pomcpow_planner, b0, tree_in_info=true)
tree = D3Tree(info[:tree], init_expand=1)
inchrome(tree)

# benchmarks
random_planner = RandomPolicy(pomdp)
strong_planner = EfficiencyPolicy(pomdp, [true, true, true, true])
robust_planner = EfficiencyPolicyWithUncertainty(pomdp, 5., [true, true, true, true])
eco_planner = EmissionAwarePolicy(pomdp, [true, true, true, true])
planners = [random_planner, strong_planner, robust_planner, eco_planner, pomcpow_planner]

n_reps=10
max_steps=15

evaluate_policies(pomdp, planners, n_reps, max_steps)

#save planner
println("pomcpow policy saved as planners/pomcpow_planner.jld2")
serialize("planners/pomcpow_planner.jld2", pomcpow_planner)

# test model
# s = deepcopy(s0)
# b = deepcopy(b0)

# println("\n==============simulate==============\n")
# a= action(random_planner, b)
# rng = MersenneTwister(1)
# (sp, o, r) = gen(pomdp, s, a, rng)
# b1 = update(up, b, a, o) 
# println("b0: ", b)
# println("s0: ", s)
# println("a: ", a)
# println("o: ", o)
# println("sp: ", sp)
# println("b1: ", b1)
# b = deepcopy(b1)
# s = deepcopy(sp);
# println("\n==============simulate==============\n")
# a= EXPLORE2#action(random_planner, b)
# rng = MersenneTwister(1)
# (sp, o, r) = gen(pomdp, s, a, rng)
# b1 = update(up, b, a, o) 
# println("b: ", b)
# println("s: ", s)
# println("a: ", a)
# println("o: ", o)
# println("sp: ", sp)
# println("b1: ", b1)
# b = b1
# s = sp;
# println("\n==============simulate==============\n")
# a= action(random_planner, b)
# rng = MersenneTwister(1)
# (sp, o, r) = gen(pomdp, s, a, rng)
# b1 = update(up, b, a, o) 
# println("b0: ", b)
# println("s0: ", s)
# println("a: ", a)
# println("o: ", o)
# println("sp: ", sp)
# println("b1: ", b1)
# b = deepcopy(b1)
# s = deepcopy(sp);
# println("\n==============simulate==============\n")
# a= action(random_planner, b)
# rng = MersenneTwister(1)
# (sp, o, r) = gen(pomdp, s, a, rng)
# b1 = update(up, b, a, o) 
# println("b0: ", b)
# println("s0: ", s)
# println("a: ", a)
# println("o: ", o)
# println("sp: ", sp)
# println("b1: ", b1)
# b = deepcopy(b1)
# s = deepcopy(sp);

