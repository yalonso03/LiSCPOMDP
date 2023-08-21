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

max_steps = 15

planner = deserialize("planners/pomcpow_planner.jld2")
#planner = strong_planner = EfficiencyPolicy(pomdp, [true, true, true, true])

# test model
s = deepcopy(s0)
b = deepcopy(b0)
println("s0: ", s)
println("b0: ", b)
rng = MersenneTwister(1)

println("\n==============simulate==============\n")
a= EXPLORE1#action(planner, b)
(sp, o, r) = gen(pomdp, s, a, rng)
b1 = update(up, b, a, o) 
println("b0: ", b)
println("s0: ", s)
println("a: ", a)
println("o: ", o)
println("sp: ", sp)
println("b1: ", b1)
b = deepcopy(b1)
s = deepcopy(sp);
println("\n==============simulate==============\n")
a= MINE2#action(random_planner, b)
(sp, o, r) = gen(pomdp, s, a, rng)
println("b: ", b)
b1 = update(up, b, a, o) 
println("s: ", s)
println("a: ", a)
println("o: ", o)
println("sp: ", sp)
println("b1: ", b1)
b = deepcopy(b1)
s = deepcopy(sp);
println("\n==============simulate==============\n")
a= MINE1#action(planner, b)
(sp, o, r) = gen(pomdp, s, a, rng)
b1 = update(up, b, a, o) 
println("b0: ", b)
println("s0: ", s)
println("a: ", a)
println("o: ", o)
println("sp: ", sp)
println("b1: ", b1)
b = deepcopy(b1)
s = deepcopy(sp);
println("\n==============simulate==============\n")
a= EXPLORE2#action(planner, b)
(sp, o, r) = gen(pomdp, s, a, rng)
b1 = update(up, b, a, o) 
println("b0: ", b)
println("s0: ", s)
println("a: ", a)
println("o: ", o)
println("sp: ", sp)
println("b1: ", b1)
b = deepcopy(b1)
s = deepcopy(sp);
println("\n==============simulate==============\n")
a= action(planner, b)
(sp, o, r) = gen(pomdp, s, a, rng)
b1 = update(up, b, a, o) 
println("b0: ", b)
println("s0: ", s)
println("a: ", a)
println("o: ", o)
println("sp: ", sp)
println("b1: ", b1)
b = deepcopy(b1)
s = deepcopy(sp);