#=
Modeling the US Path to Lithium Self Sufficiency Using POMDPs
Summer 2023
Yasmine Alonso, Mansur Arief, Anthony Corso, Jef Caers, and Mykel Kochenderfer

File: LiPOMDP.jl
----------------
This file contains the continuous version of the observation function. We use the DiscreteLiPOMDP.jl file
for planning, and this file for running simulations.
=#

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


@with_kw mutable struct State
    deposits::Vector{Float64} # [v₁, v₂, v₃, v₄]
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

# Make a copy of the state
function Base.deepcopy(s::State)
   return State(deepcopy(s.deposits), s.t, s.Vₜ, deepcopy(s.have_mined))  # don't have to copy t and Vₜ cuz theyre immutable i think
end

# All potential actions
@enum Action MINE1 MINE2 MINE3 MINE4 EXPLORE1 EXPLORE2 EXPLORE3 EXPLORE4
rng = MersenneTwister(1)

@with_kw struct LiPOMDP <: POMDP{State, Action, Any} 
    t_goal = 10
    σ_obs = 0.1
    Vₜ_goal = 5
    γ = 0.9
    n_deposits = 4 
    bin_edges = [0.25, 0.5, 0.75]  # Used to discretize observations
    obs_type = "continuous"
    cdf_threshold = 0.1  # threshold allowing us to mine or not
    min_n_units = 3  # minimum number of units required to mine. So long as cdf_threshold portion of the probability
    obj_weights = [0.33, 0.33, 0.33]  # how we want to weight each component of the reward 
    CO2_emissions::Vector{Float64} = [5, 7, 2, 5]  #[C₁, C₂, C₃, C₄] amount of CO2 each site emits
    null_state::State = State([-1, -1, -1, -1], -1, -1, [false, false, false, false])
    init_state::State = State([8.9, 7, 1.8, 5], 0, 0, [false, false, false, false])  # For convenience for me rn when I want to do a quick test and pass in some state
end

# Belief struct
struct LiBelief{T<:UnivariateDistribution}
    deposit_dists::Vector{T}
    t::Float64
    V_tot::Float64
    have_mined::Vector{Bool} 
end


# Input a belief and randomly produce a state from it 
function Base.rand(rng::AbstractRNG, b::LiBelief)
    deposit_samples = rand.(rng, b.deposit_dists)
    t = b.t
    V_tot = b.V_tot
    have_mined = b.have_mined
    return State(deposit_samples, t, V_tot, have_mined)
end


# Unsure if the right way to do this is to have a product distribution over the 4 deposits? 
# Pass in an RNG?
function POMDPs.initialstate(P::LiPOMDP)
    init_state = State([8.9, 7, 1.8, 5], 0, 0, [false, false, false, false])
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
    
    deposit_vec_bounds = [(V_deposit_min, V_deposit_max) for x in 1:P.n_deposits]  # Make a length-4 vector, one for each deposit
    V_tot_bounds = Interval(V_tot_min, V_tot_max)
    time_bounds = 0:10  # This can be discrete since we're only going a year at a time
    
    𝒮 = product_state_space(deposit_vec_bounds, V_tot_bounds, time_bounds)  # Cartesian product 
    # QUESTION: how could I add the null state into the space?
    return 𝒮
    
end



function POMDPs.actions(P::LiPOMDP)
    return [MINE1, MINE2, MINE3, MINE4, EXPLORE1, EXPLORE2, EXPLORE3, EXPLORE4]
end
# Action function: now dependent on belief state
function POMDPs.actions(P::LiPOMDP, b::LiBelief)
    potential_actions = [MINE1, MINE2, MINE3, MINE4, EXPLORE1, EXPLORE2, EXPLORE3, EXPLORE4]#actions(P)
    
    # Checks to ensure that we aren't trying to explore at a site we have already mined at
    potential_actions = filter(a -> can_explore_here(a, b), potential_actions)
    
    # Ensures that there is <= 10% (or P.cdf_threshold) of the belief distribution below the P.min_n_units
    for i = 1:4
        dist = b.deposit_dists[i]
        portion_below_threshold = cdf(dist, P.min_n_units)
        if portion_below_threshold > P.cdf_threshold  # BAD!

            bad_action_str = "MINE" * string(i)
            bad_action = str_to_action(bad_action_str)
            # Ensure that this bad_action is not in potential_actions
            potential_actions = filter(a -> a != bad_action, potential_actions)
        end   
    end 
    return potential_actions
    
end


# Reward function: returns the reward for being in state s and taking action a
# Reward is comprised of three parts:
#       1. Whether or not we have reached our time delay + volume goal (1 if yes, 0 if no)
#       2. The amount of volume we have mined
#       3. The amount of CO2 emissions we have produced if taking a mine action (negative)
# The three parts of the reward are then weighted by the obj_weights vector and returned.

function POMDPs.reward(P::LiPOMDP, s::State, a::Action)
    # Time and volume delay goal
    r1 = (s.t >= P.t_goal && s.Vₜ >= P.Vₜ_goal) ? 100 : 0

    # Total amount mined thus far
    r2 = s.Vₜ
    
    r3 = get_action_emission(P, a)
    
    reward = dot([r1, r2, r3], P.obj_weights)
    
    return reward
end

# Gen function
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
    
    # o is continuous
    o = rand(rng, observation(P, a, next_state))  # Vector of floats
    r = reward(P, s, a)

    out = (sp=next_state, o=o, r=r)  
    #println(out)
    return out
end

# Observation function
function POMDPs.observation(P::LiPOMDP, a::Action, sp::State)
    # When we take an action to EXPLORE one of the four sites, we only really gain an observation on said
    # state. So, the other remaining three states have this kinda sentinel distribution thing of -1 to represent
    # that it's not really important/relevant
    site_number = get_site_number(a)  #1, 2, 3, or 4, basically last character of    
    action_type = get_action_type(a)  # "EXPLORE" or "MINE"

    sentinel_dist = DiscreteNonParametric([-1], [1])
    temp::Vector{UnivariateDistribution} = fill(sentinel_dist, 4)

    # handle degenerate case where we have no more Li at this site
    if sp.deposits[site_number] <= 0
        site_dist = sentinel_dist
        return product_distribution(temp)        
    end
    
    if action_type == "MINE"
        return product_distribution(temp) 
    end

    #handles EXPLORE
    if P.obs_type == "continuous"        
        temp[site_number] = Normal(sp.deposits[site_number], P.σ_obs)
        #println("returning cts obs type")
        return product_distribution(temp)
    else    
        site_dist = Normal(sp.deposits[site_number], P.σ_obs)
        # sample_point = rand(site_dist)  # Step 1: get us a random sample on that distribution 
        quantile_vols = quantile(site_dist, P.bin_edges)  # Step 2: get the Li Volume amounts that correspond to each quantile
        quantile_vols = [round(x, digits=1) for x in quantile_vols]  # Round to 1 decimal place

        # Now get the chunk boundaries (Dashed lines in my drawings)
        chunk_boundaries = compute_chunk_boundaries(quantile_vols)

        # Now compute the probabilities of each chunk
        probs = compute_chunk_probs(chunk_boundaries, site_dist)
        #println("sp: ", sp, "q :", quantile_vols)
        
        # I believe the idea was that with other solvers, we need an observation fn that returns an explicit
        # distribution, not just a sample. So, I decided to use a sparsecat here, but I'm unsure, since all of this doesn't
        # really seem to be working properly :(
        #println("site dist: ", site_dist)
        #println("returning discrete obs type: ", quantile_vols)
        return SparseCat(quantile_vols, probs)
    end
end

# Define == operator to use in the termination thing, just compares two states
Base.:(==)(s1::State, s2::State) = (s1.deposits == s2.deposits) && (s1.t == s2.t) && (s1.Vₜ == s2.Vₜ) && (s1.have_mined == s2.have_mined)

POMDPs.discount(P::LiPOMDP) = P.γ

POMDPs.isterminal(P::LiPOMDP, s::State) = s == P.null_state

# kalman_step is used in the belief updater update function
#function kalman_step(P::LiPOMDP, μ::Float64, σ::Float64, z::Float64)
#    k = σ / (σ + P.σ_obs)  # Kalman gain
#    μ_prime = μ + k * (z - μ)  # Estimate new mean
#    σ_prime = (1 - k) * σ   # Estimate new uncertainty
#    return μ_prime, σx_prime
#end

# takes in a belief, action, and observation and uses it to update the belief
function update(P::LiPOMDP, b::LiBelief, a::Action, o::Vector{Float64})
    action_type = get_action_type(a)
    site_number = get_site_number(a)
    
    if action_type == "EXPLORE"
        bi = b.deposit_dists[site_number]  
        zi = o[site_number]
        prior = (bi, std(bi))
        bi_prime = Normal(kalman_step(P, mean(bi), std(bi), zi))#posterior(prior, Normal, zi)
        
        # Default, not including updated belief
        belief = LiBelief(copy(b.deposit_dists), b.t + 1, b.V_tot, copy(b.have_mined))
        
        # Now, at the proper site number, update to contain the updated belief
        belief.deposit_dists[site_number] = bi_prime
        
        return belief

    else # a must be a MINE action
        bi = b.deposit_dists[site_number]
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
        belief = LiBelief(copy(b.deposit_dists), b.t + 1, b.V_tot + n_units_mined, copy(b.have_mined))
        
        # Now, at the proper site number, update to contain the updated belief
        belief.deposit_dists[site_number] = Normal(μi_prime, σi)
        belief.have_mined[site_number] = true  # Update the mining history
        return belief
    end 
end


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
