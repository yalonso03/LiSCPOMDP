using POMDPs
using POMCPOW
using POMDPModels
using POMDPTools
using BasicPOMCP

include("LiPOMDP.jl")
include("utils.jl")


# POMCPOW Solver
# pomdp = LiPOMDP(obs_type="discrete")
pomdp = LiPOMDP(obs_type="continous")

s0 = pomdp.init_state


solver = POMCPOWSolver(tree_queries=100, estimate_value = 0) # Estimate value should fix the previous problem with action functions
planner = solve(solver, pomdp)
b0 = LiBelief([Normal(9, 0.2), Normal(3, 0.2), Normal(3, 0.2), Normal(9, 2.0)], 0.0, 0.0, [false, false, false, false])
up = LiBeliefUpdater(pomdp)

a= action(planner, b0)
ap, info = action_info(planner, b0, tree_in_info=true)
tree = D3Tree(info[:tree], init_expand=1)
inchrome(tree)


pomcp_planner = solve(solver, pomdp)

println("solved, going to simulate")
# pomdp = LiPOMDP(obs_type="continous")

# updater(planner)

hr = HistoryRecorder(max_steps=11)
hist = simulate(hr, pomdp, planner, up, b0)
rhist = simulate(hr, pomdp, FunctionPolicy((b) -> rand(actions(pomdp, b))), up, b0)  #! should I seed this
bhist = simulate(hr, pomdp, pomcp_planner, up, b0)
println("""
    Cumulative Discounted Reward (for 1 simulation)
        Random: $(eval_policy(pomdp, rhist)
        POMCPOW: $(eval_policy(pomdp, hist)),
        POMCP: $(eval_policy(pomdp, bhist)).
    """)
