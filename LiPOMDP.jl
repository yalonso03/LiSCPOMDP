#=
Modeling the US Path to Lithium Self Sufficiency Using POMDPs
Summer 2023
By: Yasmine Alonso with assistance from Mykel Kochenderfer, Mansur Arief, and Anthony Corso
=#

#= 
Updates since last meeting
--------------------------
    - Made state space continuous
        - QUESTION: how to add null_state to continuous state space, could i like union it or smth
    - Added an initialstate, product_distribution of 4 normals 1 for each deposit
        - "If it is difficult to define the probability density or mass function explicitly, consider using 
            POMDPModelTools.ImplicitDistribution to define a model for sampling."
    - Made a copy of the state instead, as per Mykel's suggestion
   
    - QUESTION: What is the dimension of the state space, do we have 4? Was thinking 1 dimension for each
        deposit, and then V_t and t don't add dimensions because they're certain?
    - QUESTION: coming up with these covariance matrices, where should we get the numbers from lol i'm making stuff up 
    - QUESTION: o = Float64.(rand(dist)) Is this a bad way to turn them all into floats? does rand(Dist) only give back ints?
=#

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

@with_kw mutable struct State
    deposits::Vector{Float64} # [v₁, v₂, v₃, v₄]
    CO2_emissions::Vector{Float64}  #[C₁, C₂, C₃, C₄] amount of CO2 each site emits
    t::Float64 = 0  # current time
    Vₜ::Float64 = 0  # current amt of Li mined up until now
    have_mined::Vector{Bool} = [false, false, false, false]  # Boolean value to represent whether or not we have taken a mine action
end

# To make the struct iterable (potentially for value iteration?) Was experiencing errors
function Base.iterate(state::State, index=1)
    if index <= 5  # I should get rid of magic numbers later
        
        # If on a valid field index, get the field name and then the thing at that field
        field = fieldnames(State)[index]
        value = getfield(state, field)
        # Return value and the next index for iteration
        return (value, index + 1)
    else
        # If we've gone through all fields, return nothing to signify that we're done
        return nothing
    end
end

function Base.deepcopy(s::State)
   return State(deepcopy(s.deposits), deepcopy(s.CO2_emissions), s.t, s.Vₜ, s.have_mined)  # don't have to copy t and Vₜ cuz theyre immutable i think
end

@enum Action MINE1 MINE2 MINE3 MINE4 EXPLORE1 EXPLORE2 EXPLORE3 EXPLORE4
rng = MersenneTwister(1)

@with_kw struct LiPOMDP <: POMDP{State, Action, Any} 
    t_goal = 10
    σ_obs = 0.1
    Vₜ_goal = 5
    γ = 0.9
    n_deposits = 4 
    bin_edges = [0.25, 0.5, 0.75]  # Used to discretize observations
    cdf_threshold = 0.95  # threshold allowing us to mine or not
    min_n_units = 3  # minimum number of units required to mine. So long as cdf_threshold portion of the probability
                    # is 
    obj_weights = [0.33, 0.33, 0.33]  # how we want to weight each component of the reward
    
    null_state::State = State([-1, -1, -1, -1], [-1, -1, -1, -1], -1, -1, [false, false, false, false])
    init_state::State = State([8.9, 7, 1.8, 5], [1, 2, 3, 4], 0, 0, [false, false, false, false])  # For convenience for me rn when I want to do a quick test and pass in some state
end

struct LiBelief
    deposit_dists::Vector{Normal}
    CO2_emissions::Vector{Float64}
    t::Float64
    V_tot::Float64
    have_mined::Vector{Bool} 
end

function Base.rand(rng::AbstractRNG, b::LiBelief)
    deposit_samples = rand.(rng, b.deposit_dists)
    CO2_emissions = b.CO2_emissions
    t = b.t
    V_tot = b.V_tot
    have_mined = b.have_mined
    return State(deposit_samples, CO2_emissions, t, V_tot, have_mined)
end

# In: an action (enum)
# Out: the last digit of the action as an int. This is helpful to clean up some functions so that I can just apply
# whatever I need to the desired index of the deposits vector
function get_site_number(a::Action)
    action_str = string(a)
    len = length(action_str)
    deposit_number = Int(action_str[len]) - 48  # -48 because Int() gives me the ascii code
end


function splice(begin_index, end_index, str)
    result = ""
    for i = begin_index:end_index
        result = result * str[i]
    end
    return result
end


# Takes in an action, a, and returns whether the action is a MINE action
# or an EXPLORE action, as a string.
function get_action_type(a::Action)
    action_str = string(a)
    len = length(action_str)
    action_type = splice(1, len - 1, action_str)
    return action_type
end

# Unsure if the right way to do this is to have a product distribution over the 4 deposits? 
# Pass in an RNG?
function POMDPs.initialstate(P::LiPOMDP)
    init_state = State([8.9, 7, 1.8, 5], [1, 2, 3, 4], 0, 0, [false, false, false, false])
    return Deterministic(init_state)
end

# Continuous state space
# NEED TO uPDATE THIS TO ACCOUNT FOR NEW ADDITIONS TO THE STATE
function POMDPs.states(P::LiPOMDP)
    # Min and max amount per singular deposit
    V_deposit_min = 0
    V_deposit_max = 10
    
    # Min and max amount total mined, can be at smallest the deposit_min * 4, and at largest, the deposit_max * 4
    V_tot_min = V_deposit_min * P.n_deposits  # 0
    V_tot_max = V_deposit_max * P.n_deposits  # 40
    
    deposit_vec_bounds = [(V_deposit_min, V_deposit_max) for x in 1:n_deposits]  # Make a length-4 vector, one for each deposit
    V_tot_bounds = Interval(V_tot_min, V_tot_max)
    time_bounds = 0:10  # This can be discrete since we're only going a year at a time
    
    𝒮 = product_state_space(deposit_vec_bounds, V_tot_bounds, time_bounds)  # Cartesian product 
    # QUESTION: how could I add the null state into the space?
    return 𝒮
    
end

# function POMDPs.actions(P::LiPOMDP)
#     return [MINE1, MINE2, MINE3, MINE4, EXPLORE1, EXPLORE2, EXPLORE3, EXPLORE4]
# end

function can_explore_here(a::Action, b::LiBelief)
    action_type = get_action_type(a)
    site_number = get_site_number(a)
    if action_type == "MINE"
        return true
    end
    
    return !b.have_mined[site_number]
end

function POMDPs.actions(P::LiPOMDP, b::LiBelief)
    potential_actions = [MINE1, MINE2, MINE3, MINE4, EXPLORE1, EXPLORE2, EXPLORE3, EXPLORE4]
    
    for i = 1:4
        potential_actions = filter(a -> can_explore_here(a, b), potential_actions)
    end
    
    for i = 1:4
        dist = b.deposit_dists[i]
        portion_below_threshold = cdf(dist, P.min_n_units)
        if portion_below_threshold > P.cdf_threshold  # BAD!
            bad_action = "MINE" * string(i)
            # Ensure that this bad_action is not in potential_actions
            potential_actions = filter(a -> a != bad_action, potential_actions)
        end   
    end 
    return potential_actions
    
end

function tester(P::LiPOMDP, b::LiBelief)
    
    
   for i = 1:4
        dist = b.deposit_dists[i]
        portion_below_threshold = cdf(dist, P.min_n_units)
        if portion_below_threshold > P.cdf_threshold  # BAD!
            bad_action = "MINE" * string(i)
            # Ensure that this bad_action is not in potential_actions
            potential_actions = filter(a -> a != bad_action, potential_actions)
        end   
    end  
end



P = LiPOMDP()
b0 = LiBelief([Normal(1, .1), Normal(5, 2), Normal(2, 0.2), Normal(9, 4)], [1, 2, 3, 4], 0.0, 0.0, [false, false, false, false])
actions(P, b0)

# function POMDPs.reward(P::LiPOMDP, s::State, a::Action)
#     if s.t >= P.t_goal && s.Vₜ >= P.Vₜ_goal
#         return 1
#     else
#         return 0
#     end
# end

# Updated reward fn
function POMDPs.reward(P::LiPOMDP, s::State, a::Action)
    r = 0
    
    # Not sure how we want to account for time
    r1 = (s.t >= P.t_goal && s.Vₜ >= P.Vₜ_goal) ? 1 : 0

    # Do we have to discount this manually? or will that be done with the solver
    r2 = s.Vₜ
    
    action_type = get_action_type(a)
    action_number = get_site_number(a)
    
    # r3 is the co2 emission part
    r3 = (action_type == "MINE") ? s.CO2_emissions[action_number] * -1 : 0
    
    reward = dot([r1, r2, r3], P.obj_weights)
    
    return reward
end

function POMDPs.gen(P::LiPOMDP, s::State, a::Action, rng::AbstractRNG)
    next_state::State = deepcopy(s) # Make a copy!!! need to be wary of this in Julia deepcopy might be slow
    next_state.t = s.t + 1  # Increase time by 1 in all cases
    
    if s.t >= P.t_goal && s.Vₜ >= P.Vₜ_goal  # If we've reached all our goals, we can terminate
        next_state = P.null_state
    end
    
    action_type = get_action_type(a)
    site_number = get_site_number(a)
    
    # If we choose to MINE, so long as there is Li available to us, decrease amount in deposit by one unit
    # and increase total amount mined Vₜ by 1 unit. We do not have any transitions for EXPLORE actions because 
    # exploring does not affect state
    if action_type == "MINE" && s.deposits[site_number] >= 1
        next_state.deposits[site_number] = s.deposits[site_number] - 1
        next_state.Vₜ = s.Vₜ + 1
    end
    
    # If we're mining, update state to reflect that we now have mined and can no longer explore
    if action_type == "MINE"
        next_state.have_mined[site_number] = true
    end
    # Now sample an observation and get the reward as well
    
    # Here now in gen we're using the discrete observation.
    cts_dist, o = disc_and_ctns_obs(P, a, next_state) #rand(rng, observation(P, a, next_state))  # Vector of floats
    r = reward(P, s, a)
    return (sp=next_state, o=o, r=r)  
end

function POMDPs.observation(P::LiPOMDP, a::Action, sp::State)
    # When we take an action to EXPLORE one of the four sites, we only really gain an observation on said
    # state. So, the other remaining three states have this kinda sentinel distribution thing of -1 to represent
    # that it's not really important/relevant

    site_number = get_site_number(a)  #1, 2, 3, or 4, basically last character of    
    action_type = get_action_type(a)  # "EXPLORE" or "MINE"
    temp::Vector{UnivariateDistribution} = [DiscreteNonParametric([-1], [1]), DiscreteNonParametric([-1], [1]), DiscreteNonParametric([-1], [1]), DiscreteNonParametric([-1], [1])]
    
    if action_type == "MINE"
       return product_distribution(temp) 
        
    else  # action_type must be "EXPLORE"
#         temp[site_number] = Normal(sp.deposits[site_number], P.σ_obs)
#         return product_distribution(temp)
        cts_dist, _ = disc_and_ctns_obs(P, a, sp)  # Here we don't need the discretized obs, only cts. We will use discretized in gen
        return cts_dist
    end
end

# This function allows us to get both the distribution (continuous, like before) and then also a discretized
# Version of the observation (returns 1 of 5 possible observations). It returns the distribution (continuous)
# and also a vector of floats with the proper site index holding the bucketed observation
function disc_and_ctns_obs(P::LiPOMDP, a::Action, sp::State)
    site_number = get_site_number(a)  #1, 2, 3, or 4, basically last character of    
    temp::Vector{UnivariateDistribution} = [DiscreteNonParametric([-1], [1]), DiscreteNonParametric([-1], [1]), DiscreteNonParametric([-1], [1]), DiscreteNonParametric([-1], [1])]
    site_dist = Normal(sp.deposits[site_number], P.σ_obs)
    
    
    # Bin converting, snapping rand obs -> 1 of 5 things
    sample_point = rand(site_dist)  # Step 1: get us a random sample on that distribution 
    quantile_vols = quantile(site_dist, P.bin_edges)  # Step 2: get the Li Volume amounts that correspond to each quantile
    index = argmin(abs.(sample_point .- quantile_vols))  # Step 3: get the index of the quantiled volume that is closest to our sample_point
    snapped_obs = quantile_vols[index]
    
    # Continuous observation
    temp[site_number] = site_dist
    cts_dist = product_distribution(temp)  # Continuous stuff, the usual
    
    # Discretized observation
    discretized_obs = [-1.0, -1.0, -1.0, -1.0]  
    discretized_obs[site_number] = snapped_obs  # Replace proper index with relevant observation
   
    return cts_dist, discretized_obs
end

# Define == operator to use in the termination thing, just compares two states
Base.:(==)(s1::State, s2::State) = (s1.deposits == s2.deposits) && (s1.t == s2.t) && (s1.Vₜ == s2.Vₜ) && (s1.CO2_emissions == s2.CO2_emissions) && (s1.have_mined == s2.have_mined)

POMDPs.discount(P::LiPOMDP) = P.γ
POMDPs.isterminal(P::LiPOMDP, s::State) = s == P.null_state

#P = LiPOMDP()
function kalman_step(P::LiPOMDP, μ::Float64, σ::Float64, z::Float64)
    k = σ / (σ + P.σ_obs)  # Kalman gain
    μ_prime = μ + k * (z - μ)  # Estimate new mean
    σ_prime = (1 - k) * σ   # Estimate new uncertainty
    return μ_prime, σ_prime
end

function update(P::LiPOMDP, b::LiBelief, a::Action, o::Vector{Float64})
    # EXPLORE actions: Adjust mean of the distribution corresponding to the proper deposit, using the Kalman
    # predict/update step (see kalman_step function above). Time increases by 1 in the belief.
    # Return new belief, with everything else untouched (EXPLORE only allows us to gain info about one site) 
    action_type = get_action_type(a)
    site_number = get_site_number(a)
    
    if action_type == "EXPLORE"
        bi = b.deposit_dists[site_number]  # This is a normal distribution
        μi = mean(bi)
        σi = std(bi)
        zi = o[site_number]
        μ_prime, σ_prime = kalman_step(P, μi, σi, zi)
        bi_prime = Normal(μ_prime, σ_prime)
        
        # Default, not including updated belief
        belief = LiBelief([b.deposit_dists[1], b.deposit_dists[2], b.deposit_dists[3], b.deposit_dists[4]], b.CO2_emissions, b.t + 1, b.V_tot, b.have_mined)
        
        # Now, at the proper site number, update to contain the updated belief
        belief.deposit_dists[site_number] = bi_prime
        
        return belief

    # MINE actions: Shifts our mean of the distribution corresponding to the proper deposit down by 1 (since we
    # have just mined one unit deterministically). Does not affect certainty at all. 
    else # a must be a MINE action
        bi = b.deposit_dists[1]
        μi = mean(bi)
        σi = std(bi)
        
        if μi >= 1
            μi_prime = μi - 1
            n_units_mined = 1  # we were able to mine a unit
        else 
            μi_prime = μi
            n_units_mined = 0  # we did NOT mine a unit
        end
        
        # Default, not including updated belief
        belief = LiBelief([b.deposit_dists[1], b.deposit_dists[2], b.deposit_dists[3], b.deposit_dists[4]], b.CO2_emissions, b.t + 1, b.V_tot + n_units_mined, b.have_mined)
        # Now, at the proper site number, update to contain the updated belief
        belief.deposit_dists[site_number] = Normal(μi_prime, σi)
        return belief
    end 
end

# using QMDP
# pomdp = LiPOMDP() # initialize POMDP

# # initialize the solver
# # key-word args are the maximum number of iterations the solver will run for, and the Bellman tolerance
# solver = QMDPSolver(max_iterations=20,
#                     belres=1e-3,
#                     verbose=true
#                    ) 

# # run the solver
# policy = solve(solver, pomdp)

# POMCPOW Solver
solver = POMCPOWSolver()
pomdp = LiPOMDP()
planner = solve(solver, pomdp)
b0 = LiBelief([Normal(9, 0.2), Normal(5, 2), Normal(2, 0.2), Normal(9, 4)], [1, 2, 3, 4], 0.0, 0.0, [false, false, false, false])

ap, info = action_info(planner, b0, tree_in_info=true)
tree = D3Tree(info[:tree], init_expand=1)

# inputs: pomddp, an initial belief, and a sequence of actions. 
# Runs the updater on said sequence, keeping track of the belief at each time step in a history vector
# returns: the history vector of all of the beliefs, and all true states 

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
init_belief = LiBelief([Normal(9, 0.2), Normal(5, 2), Normal(2, 0.2), Normal(9, 4)], [1, 2, 3, 4], 0, 0, [false, false, false, false])
init_state = P.init_state

a = EXPLORE3
dist = observation(P, a, init_state)  # Product distribution
o = rand(dist) # Vector of floats, index in to proper index
new_belief = update(P, init_belief, a, o)

using Plots
P = LiPOMDP()
init_belief = LiBelief([Normal(9, 0.2), Normal(5, 2), Normal(2, 0.2), Normal(9, 4)], [1, 2, 3, 4], 0.0, 0.0, [false, false, false, false])
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



# using D3Trees
# solver = POMCPOWSolver()
# pomdp = LiPOMDP()
# planner = solve(solver, pomdp)

# # Query an action from the planner and some initial belief
# b0 = LiBelief([Normal(9, 0.2), Normal(5, 2), Normal(2, 0.2), Normal(9, 4)], [1, 2, 3, 4], 0.0, 0.0, false)
# ap, info = action_info(planner, b0)
# ap

# random_policy = RandomPolicy(pomdp)

# # Query an action  
# b0 = LiBelief([Normal(9, 0.2), Normal(5, 2), Normal(2, 0.2), Normal(9, 4)], [1, 2, 3, 4], 0.0, 0.0, false)
# ap, info = action_info(planner, b0)
# ap

# # random_policy = RandomPolicy(pomdp)  # Not passing in a belief updater right now (or an rng but I feel like i should for reproducibility)
# # #show_requirements(get_requirements(POMDPs.solve, (solver, pomdp)))
# # ap, info = action_info(planner, initialstate(pomdp), tree_in_info=true)
# # tree = D3Tree(info[:tree], init_expand=3)

# # a = action(random_policy, initialstate(pomdp))
# # a

# function run_simulation_new(P::LiPOMDP, b0::LiBelief, s0::State, rng::AbstractRNG)
#     b = b0
#     s = s0
#     total_reward = 0
#     belief_history = []
#     state_history = []
#     action_history = []
#     reward_history = []
#     push!(belief_history, b)
#     push!(state_history, s)
#     i = 0
    
#     while !isterminal(P, s)
#         a, ai = action_info(planner, b)  # Probably should pass in the planner
#         push!(action_history, a)
        
#         sp, o, r = gen(P, s, a, rng)
#         push!(reward_history, r)
#         total_reward += r
#         o = Float64.(o)
#         b = update(P, b, a, o) 
#        push!(belief_history, b)

#         s = deepcopy(sp)
#         push!(state_history, s)
#         i=i+1
#         if i > 50
#             break
#         end
#     end
#     return belief_history, state_history, action_history, reward_history, total_reward
# end


# function run_simulation_random_policy(P::LiPOMDP, b0::LiBelief, s0::State, rng::AbstractRNG)
#     b = b0
#     s = s0
#     total_reward = 0
#     belief_history = []
#     state_history = []
#     action_history = []
#     reward_history = []
#     push!(belief_history, b)
#     push!(state_history, s)
#     i = 0
    
#     while !isterminal(P, s)
#         a = action(random_policy, b)
#         push!(action_history, a)
        
#         sp, o, r = gen(pomdp, s, a, rng)
#         push!(reward_history, r)
#         total_reward += r
#         o = Float64.(o)
#         b = update(P, b, a, o) 
#        push!(belief_history, b)

#         s = deepcopy(sp)
#         push!(state_history, s)
#         i=i+1
#         if i > 50
#             break
#         end
#     end
#     return belief_history, state_history, action_history, reward_history, total_reward
# end

# b0 = LiBelief([Normal(9, 0.2), Normal(5, 2), Normal(2, 0.2), Normal(9, 4)], [1, 2, 3, 4], 0.0, 0.0, false)
# s0 = State([8.9, 7, 1.8, 5], [1, 2, 3, 4], 0, 0, false)

# belief_history_rand, state_history_rand, action_history_rand, reward_history_rand, total_reward_rand = run_simulation_random_policy(pomdp, b0, s0, rng)
# belief_history, state_history, action_history, reward_history, total_reward = run_simulation_new(pomdp, b0, s0, rng)
mynumber = 5