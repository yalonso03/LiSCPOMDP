#=
Modeling the US Path to Lithium Self Sufficiency Using POMDPs
Summer 2023
Yasmine Alonso, Mansur Arief, Anthony Corso, Jef Caers, and Mykel Kochenderfer

File: DiscreteLiPOMDP.jl
------------------------
Discrete version of the LiPOMDP defined in LiPOMDP.jl that we use for planning. Then, we can use
the version of the pomdp (LiPOMDP.jl) that has continuous observations for simulations.
=#

include("LiPOMDP.jl")

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

@with_kw struct DiscreteLiPOMDP <: POMDP{State, Action, Any}
    lipomdp = LiPOMDP()
    bin_edges = [0.25, 0.5, 0.75]  # will be used to discretize observations
end

POMDPs.initialstate_distribution(::DiscreteLiPOMDP) = DiscreteNonParametric([State([8.9, 7, 1.8, 5], [1, 2, 3, 4], 0, 0, [false, false, false, false])], [1.0])


function POMDPs.states(P::DiscreteLiPOMDP)
    # Min and max amount per singular deposit
    V_deposit_min = 0
    V_deposit_max = 10
    
    # Min and max amount total mined, can be at smallest the deposit_min * 4, and at largest, the deposit_max * 4
    V_tot_min = V_deposit_min * P.lipomdp.n_deposits  # 0
    V_tot_max = V_deposit_max * P.lipomdp.n_deposits  # 40
    
    deposit_vec_bounds = [(V_deposit_min, V_deposit_max) for x in 1:P.lipomdp.n_deposits]  # Make a length-4 vector, one for each deposit
    V_tot_bounds = Interval(V_tot_min, V_tot_max)
    time_bounds = 0:10  # This can be discrete since we're only going a year at a time
    
    𝒮 = product_state_space(deposit_vec_bounds, V_tot_bounds, time_bounds)  # Cartesian product 
    # QUESTION: how could I add the null state into the space?
    return 𝒮
end

function POMDPs.actions(P::DiscreteLiPOMDP, b::LiBelief)
    potential_actions = [MINE1, MINE2, MINE3, MINE4, EXPLORE1, EXPLORE2, EXPLORE3, EXPLORE4]
    
    # Checks to ensure that we aren't trying to explore at a site we have already mined at
    potential_actions = filter(a -> can_explore_here(a, b), potential_actions)
    
    # Ensures that there is <= 10% (or P.cdf_threshold) of the belief distribution below the P.min_n_units
    for i = 1:4
        dist = b.deposit_dists[i]
        portion_below_threshold = cdf(dist, P.lipomdp.min_n_units)
        if portion_below_threshold > P.lipomdp.cdf_threshold  # BAD!

            bad_action_str = "MINE" * string(i)
            bad_action = str_to_action(bad_action_str)
            # Ensure that this bad_action is not in potential_actions
            potential_actions = filter(a -> a != bad_action, potential_actions)
        end   
    end 
    return potential_actions
end


function POMDPs.discount(P::DiscreteLiPOMDP)
    return P.lipomdp.γ
end


# Reward function: returns the reward for being in state s and taking action a
# Reward is comprised of three parts:
#       1. Whether or not we have reached our time delay + volume goal (1 if yes, 0 if no)
#       2. The amount of volume we have mined
#       3. The amount of CO2 emissions we have produced if taking a mine action (negative)
# The three parts of the reward are then weighted by the obj_weights vector and returned.
function POMDPs.reward(P::DiscreteLiPOMDP, s::State, a::Action)
    r = 0
    
    # Not sure how we want to account for time
    r1 = (s.t >= P.lipomdp.t_goal && s.Vₜ >= P.lipomdp.Vₜ_goal) ? 1 : 0

    # Do we have to discount this manually? or will that be done with the solver
    r2 = s.Vₜ
    
    action_type = get_action_type(a)
    action_number = get_site_number(a)
    
    # r3 is the co2 emission part
    r3 = (action_type == "MINE") ? P.lipomdp.CO2_emissions[action_number] * -1 : 0
    
    reward = dot([r1, r2, r3], P.lipomdp.obj_weights)
    
    return reward
end



function POMDPs.transition(P::DiscreteLiPOMDP, s::State, a::Action, rng::AbstractRNG)
    next_state::State = deepcopy(s) # Make a copy!!! need to be wary of this in Julia deepcopy might be slow
    next_state.t = s.t + 1  # Increase time by 1 in all cases
    
    if s.t >= P.lipomdp.t_goal && s.Vₜ >= P.lipomdp.Vₜ_goal  # If we've reached all our goals, we can terminate
        next_state = P.lipomdp.null_state
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

    return next_state
end


function compute_chunk_boundaries(quantile_vols::Vector{Float64})
    chunk_boundaries::Vector{Float64} = []
    for i = 1:length(quantile_vols)
        if i > 1
            boundary = (quantile_vols[i] + quantile_vols[i-1]) / 2
            push!(chunk_boundaries, boundary)
        end
    end

    return chunk_boundaries
end


function compute_chunk_probs(chunk_boundaries::Vector{Float64}, site_dist::Normal)
    chunk_probs::Vector{Float64} = []
    len = length(chunk_boundaries)
    for i = 1:len
        if i == 1
            prob = cdf(site_dist, chunk_boundaries[i])
        # elseif i == len
        #     prob = 1 - cdf(site_dist, chunk_boundaries[i])
        else
            prob = cdf(site_dist, chunk_boundaries[i]) - cdf(site_dist, chunk_boundaries[i-1])
        end
        push!(chunk_probs, prob)
    end
    p_last = 1 - cdf(site_dist, chunk_boundaries[len])
    push!(chunk_probs, p_last)
    return chunk_probs
end


P = DiscreteLiPOMDP()
dist = Normal(10, 1)
quantile_vols = quantile(dist, P.bin_edges)
chunk_boundaries = compute_chunk_boundaries(quantile_vols)
chunk_probs = compute_chunk_probs(chunk_boundaries, dist)


# Discrete version of observation function
function POMDPs.observation(P::DiscreteLiPOMDP, a::Action, sp::State)
    temp::Vector{UnivariateDistribution} = [DiscreteNonParametric([-1], [1]), DiscreteNonParametric([-1], [1]), DiscreteNonParametric([-1], [1]), DiscreteNonParametric([-1], [1])]
    action_type = get_action_type(a)

    # If we're taking a mine action we don't gain an observation
    if action_type == "MINE"
        return product_distribution(temp)
    end

    site_number = get_site_number(a)  #1, 2, 3, or 4, basically last character of    
    site_dist = Normal(sp.deposits[site_number], P.lipomdp.σ_obs)


    # sample_point = rand(site_dist)  # Step 1: get us a random sample on that distribution 
    quantile_vols = quantile(site_dist, P.bin_edges)  # Step 2: get the Li Volume amounts that correspond to each quantile

    # Now get the chunk boundaries (Dashed lines in my drawings)
    chunk_boundaries = compute_chunk_boundaries(quantile_vols)

    # Now compute the probabilities of each chunk
    probs = compute_chunk_probs(chunk_boundaries, site_dist)
    
    # I believe the idea was that with other solvers, we need an observation fn that returns an explicit
    # distribution, not just a sample. So, I decided to use a sparsecat here, but I'm unsure, since all of this doesn't
    # really seem to be working properly :(
    return SparseCat(chunk_boundaries, probs)

    
    # # Discretized observation
    # discretized_obs = [-1.0, -1.0, -1.0, -1.0]  
    # discretized_obs[site_number] = snapped_obs  # Replace proper index with relevant observation
    
    # temp[site_number] = DiscreteNonParametric([discretized_obs[site_number]], [1.0])  # Replace proper index with relevant observation
    # return product_distribution(temp)
end


pomdp = DiscreteLiPOMDP()
observation(pomdp, EXPLORE1, State([1, 1, 1, 1], 0, 0, [false, false, false, false]))

# POMCPOW Solver
solver = POMCPOWSolver()
pomdp = DiscreteLiPOMDP()
planner = solve(solver, pomdp)
b0 = LiBelief([Normal(9, 0.2), Normal(1, 2), Normal(3, 0.2), Normal(9, 4)], 0.0, 0.0, [false, false, false, false])
actions(pomdp, b0)
ap, info = action_info(planner, b0, tree_in_info=true)
tree = D3Tree(info[:tree], init_expand=1)
inchrome(tree)