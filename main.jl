using POMDPs
using POMCPOW
using POMDPModels
using POMDPTools
using BasicPOMCP

include("LiPOMDP.jl")
include("utils.jl")


# POMCPOW Solver
pomdp = LiPOMDP(obs_type="discrete")
s0 = pomdp.init_state


solver = POMCPOWSolver(tree_queries=100, next_action=next_action) #use our own random next_action function
planner = solve(solver, pomdp)
b0 = LiBelief([Normal(9, 0.2), Normal(3, 0.2), Normal(3, 0.2), Normal(9, 2.)], 0.0, 0.0, [false, false, false, false])

a= action(planner, b0)
ap, info = action_info(planner, b0, tree_in_info=true)
tree = D3Tree(info[:tree], init_expand=1)
inchrome(tree)


pomcp_planner = solve(solver, pomdp)

println("solved, going to simulate")
pomdp = LiPOMDP(obs_type="continous")

hr = HistoryRecorder(max_steps=11)
hist = simulate(hr, pomdp, planner)
rhist = simulate(hr, pomdp, RandomPolicy(pomdp))  #! should I seed this
bhist = simulate(hr, pomdp, pomcp_planner)
println("""
    Cumulative Discounted Reward (for 1 simulation)
        Random: $(eval_policy(pomdp, rhist)),
        POMCPOW: $(eval_policy(pomdp, hist)),
        POMCP: $(eval_policy(pomdp, bhist)).
    """)
