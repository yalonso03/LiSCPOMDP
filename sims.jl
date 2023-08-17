# All of the imports
using POMDPs
using POMDPModelTools
using POMDPPolicies
using QuickPOMDPs
using Parameters
using Random
using DiscreteValueIteration
using Distributions
using Plots
using POMCPOW
using LinearAlgebra
using Statistics
using QMDP
using D3Trees
#using ConjugatePriors: posterior

include("LiPOMDP.jl")
include("main.jl")
include("utils.jl")


using Plots
P = LiPOMDP()
init_belief = LiBelief([Normal(9, 0.2), Normal(5, 2), Normal(2, 0.2), Normal(9, 4)], 0.0, 0.0, [false, false, false, false])
init_state = P.init_state


# Deposit 1 stuff
# dep_1_actions = [EXPLORE1, MINE1]
#action_sequence = [EXPLORE1, EXPLORE1, EXPLORE1, EXPLORE1, MINE1, MINE1]#[rand(dep_1_actions) for x in 1:20]
action_sequence = [EXPLORE1, EXPLORE2, EXPLORE1, EXPLORE2, MINE1, MINE2]#[rand(dep_1_actions) for x in 1:20]

# Change MersenneTwister(1) to rng
belief_history, state_history = run_sims(P, init_belief, init_state, action_sequence, MersenneTwister(7))
times = [b.t for b in belief_history]
# μs = [mean(b.deposit_dists[1]) for b in belief_history]
# σs = [std(b.deposit_dists[1]) for b in belief_history]
# true_v1s = [s.deposits[1] for s in state_history] # actual amount of Li

# plot(times, μs, grid=false, ribbon=σs, fillalpha=0.5, title="Deposit 1 Belief vs. time", xlabel="Time (t)", ylabel="Amount Li in deposit 1", label="μ1", linecolor=:orange, fillcolor=:orange)
# plot!(times, true_v1s, label="Actual V₁", color=:blue)

belief_history, state_history = run_sims(P, init_belief, init_state, action_sequence, rng)
times = [b.t for b in belief_history]  # Goes up at top like an iteration counter
d1_normals = [b.deposit_dists[2] for b in belief_history]


@gif for i in 1:length(times)
    normal = d1_normals[i]
    if i < 7
        a = action_sequence[i]
    else
        a = "DONE"
    end    

    plot(3:0.01:10, (x) -> pdf(normal, x), title = "Iter. $i, a: $a", ylim = (0, 20), xlim = (3, 10), xlabel = "V belief", label= "V belief", legend=:topright, color=:purple)
end fps = 1