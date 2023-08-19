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

function run_sims(P::LiPOMDP, b0::LiBelief, s0::State, action_sequence::Vector{Action}, rng::AbstractRNG)
    b = b0
    s = s0
    belief_history = []
    state_history = []
    push!(belief_history, b)
    push!(state_history, s)
    
    for a in action_sequence
        sp, o, r = gen(P,s,a, rng)  # from anthony
        o = Float64.(o)
        # Maybe here instead of passing in the vector i need to select the right entry and pass that into update
        new_belief = update(P, b, a, o)
        b = new_belief
        s = sp
        # Changed this from s -> sp
        if isterminal(P, sp)
            break
        end
        push!(belief_history, new_belief)
        push!(state_history, sp)
    end 
    return belief_history, state_history
end

P = LiPOMDP()
init_belief = LiBelief([Normal(9, 0.2), Normal(5, 2), Normal(2, 0.2), Normal(9, 4)], 0.0, 0.0, [false, false, false, false])
init_state = P.init_state

a = EXPLORE3
dist = observation(P, a, init_state)  # Product distribution
o = rand(dist) # Vector of floats, index in to proper index
new_belief = update(P, init_belief, a, o)

using Plots
P = LiPOMDP()
init_belief = LiBelief([Normal(9, 0.2), Normal(5, 2), Normal(2, 0.2), Normal(9, 4)], 0.0, 0.0, [false, false, false, false])
init_state = P.init_state


# Deposit 1 stuff
dep_1_actions = [EXPLORE1, MINE1]
action_sequence = [EXPLORE1, EXPLORE1, MINE1, EXPLORE1, MINE1, EXPLORE1]#[rand(dep_1_actions) for x in 1:20]

# Change MersenneTwister(1) to rng
belief_history, state_history = run_sims(P, init_belief, init_state, action_sequence, MersenneTwister(7))
times = [b.t for b in belief_history]
μs = [mean(b.deposit_dists[1]) for b in belief_history]
σs = [std(b.deposit_dists[1]) for b in belief_history]
true_v1s = [s.deposits[1] for s in state_history] # actual amount of Li

plot(times, μs, grid=false, ribbon=σs, fillalpha=0.5, title="Deposit 1 Belief vs. time", xlabel="Time (t)", ylabel="Amount Li in deposit 1", label="μ1", linecolor=:orange, fillcolor=:orange)
plot!(times, true_v1s, label="Actual V₁", color=:blue)

belief_history, state_history = run_sims(P, init_belief, init_state, action_sequence, rng)
times = [b.t for b in belief_history]  # Goes up at top like an iteration counter
d1_normals = [b.deposit_dists[1] for b in belief_history]


@gif for i in 1:length(times)
    normal = d1_normals[i]
    if i < 7
        a = action_sequence[i]
    else
        a = "DONE"
    end    
    
    plot(5:0.01:10, (x) -> pdf(normal, x), title = "Iter. $i, a: $a", ylim = (0, 20), xlim = (5, 10), xlabel = "V₁ belief", label= "V₁ belief", legend=:topright, color=:purple)
end fps = 2

