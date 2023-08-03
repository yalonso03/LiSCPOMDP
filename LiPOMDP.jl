#=
Modeling the US Path to Lithium Self Sufficiency Using POMDPs
July 2023
By: Yasmine Alonso with assistance from Mykel Kochenderfer, Mansur Arief, and Anthony Corso
=#

#= 
Updates since last meeting
--------------------------
    
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
using D3Trees

@with_kw mutable struct State
    deposits::Vector{Float64} # [v₁, v₂, v₃, v₄] amount of Li in each deposit in each country, for now we're going to say we know this
    t::Float64 = 0  # current time
    Vₜ::Float64 = 0  # current amt of Li mined up until now
end

# To make the struct iterable (potentially for value iteration?) Was experiencing errors
function Base.iterate(state::State, index=1)
    if index <= 4  # I should get rid of magic numbers later
        
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
   return State(deepcopy(s.deposits), s.t, s.Vₜ)  # don't have to copy t and Vₜ cuz theyre immutable i think
end

@enum Action MINE1 MINE2 MINE3 MINE4 EXPLORE1 EXPLORE2 EXPLORE3 EXPLORE4
rng = MersenneTwister(1)

@with_kw struct LiPOMDP <: POMDP{State, Action, Any} 
    t_goal = 10
    σ_obs = 0.1
    Vₜ_goal = 15
    γ = 0.9
    n_sites = 4  # number of deposits 
    
    null_state::State = State([-1, -1, -1, -1], -1, -1)  # Null state we go to once we've reached goal
end

# QUESTION: should this be the initial TRUE state (so we don't actually know this)? 
function POMDPs.initialstate(P::LiPOMDP)
#     dist1 = Normal(2.0, P.σ_obs) 
#     dist2 = Normal(1.0, P.σ_obs)
#     dist3 = Normal(1.0, P.σ_obs)
#     dist4 = Normal(2.0, P.σ_obs)
#     init_state = product_distribution(dist1, dist2, dist3, dist4)
    init_state = State([7, 2, 3, 4], 0, 0)
    return Deterministic(init_state)
end

function POMDPs.actions(P::LiPOMDP)
    return [MINE1, MINE2, MINE3, MINE4, EXPLORE1, EXPLORE2, EXPLORE3, EXPLORE4]
end

function POMDPs.reward(P::LiPOMDP, s::State, a::Action)
    if s.t >= P.t_goal && s.Vₜ >= P.Vₜ_goal
        return 1
    else
        return 0
    end
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

function POMDPs.gen(P::LiPOMDP, s::State, a::Action, rng::AbstractRNG)  # needed to take in a rng
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
    
    # Now sample an observation and get the reward as well
    o = rand(rng, observation(P, a, next_state))  # Vector of floats
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
        temp[site_number] = Normal(sp.deposits[site_number], P.σ_obs)
        return product_distribution(temp)
    end
end

# Define == operator to use in the termination thing, just compares two states
Base.:(==)(s1::State, s2::State) = (s1.deposits == s2.deposits) && (s1.t == s2.t) && (s1.Vₜ == s2.Vₜ)  

POMDPs.discount(P::LiPOMDP) = P.γ
POMDPs.isterminal(P::LiPOMDP, s::State) = s == P.null_state

struct LiBelief
    deposits::Vector{Normal}
    t::Float64
    V_tot::Float64
end

function Base.rand(b::LiBelief)
    deposit_samples = rand.(b.deposits)
    t = b.t
    V_tot = b.V_tot
    return State(deposit_samples, t, V_tot)
end

p = LiPOMDP()
σ_obs = p.σ_obs
function kalman_step(μ::Float64, σ::Float64, z::Float64)
    k = σ / (σ + σ_obs)  # Kalman gain
    μ_prime = μ + k * (z - μ)  # Estimate new mean
    σ_prime = (1 - k) * σ   # Estimate new uncertainty
    return μ_prime, σ_prime
end

function update(b::LiBelief, a::Action, o::Vector{Float64})
    # EXPLORE actions: Adjust mean of the distribution corresponding to the proper deposit, using the Kalman
    # predict/update step (see kalman_step function above). Time increases by 1 in the belief.
    # Return new belief, with everything else untouched (EXPLORE only allows us to gain info about one site) 
    action_type = get_action_type(a)
    site_number = get_site_number(a)
    
    if action_type == "EXPLORE"
        bi = b.deposits[site_number]  # This is a normal distribution
        μi = mean(bi)
        σi = std(bi)
        zi = o[site_number]
        μ_prime, σ_prime = kalman_step(μi, σi, zi)
        bi_prime = Normal(μ_prime, σ_prime)
        
        # Default, not including updated belief
        belief = LiBelief([b.deposits[1], b.deposits[2], b.deposits[3], b.deposits[4]], b.t + 1, b.V_tot)
        
        # Now, at the proper site number, update to contain the updated belief
        belief.deposits[site_number] = bi_prime
        
        return belief

    # MINE actions: Shifts our mean of the distribution corresponding to the proper deposit down by 1 (since we
    # have just mined one unit deterministically). Does not affect certainty at all. 
    else # a must be a MINE action
        bi = b.deposits[1]
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
        belief = LiBelief([b.deposits[1], b.deposits[2], b.deposits[3], b.deposits[4]], b.t + 1, b.V_tot + n_units_mined)
        # Now, at the proper site number, update to contain the updated belief
        belief.deposits[site_number] = Normal(μi_prime, σi)
        return belief
    end 
end

P = LiPOMDP()
init_belief = LiBelief([Normal(9, 0.2), Normal(5, 2), Normal(2, 0.2), Normal(9, 4)], 0, 0)
init_state = State([8.9, 7, 1.8, 5], 0, 0)
a = EXPLORE3
dist = observation(P, a, init_state)  # Product distribution
o = rand(dist) # Vector of floats, index in to proper index
new_belief = update(init_belief, a, o)
new_belief

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
        sp, o, r = gen(P, s, a, rng)
        o = Float64.(o)
        new_belief = update(b, a, o)
        b = new_belief
        s = sp
       
        if sp == P.null_state
            break
        end
        push!(belief_history, new_belief)
        push!(state_history, sp)
    end 
    return belief_history, state_history
end

#LiBelief([Normal(9, 0.2), Normal(5, 2), Normal(2, 0.2), Normal(9, 4)], 0, 0)
using Plots
P = LiPOMDP()
init_belief = LiBelief([Normal(9, 0.2), Normal(5, 2), Normal(2, 0.2), Normal(9, 4)], 0.0, 0.0)
init_state = State([8.9, 7, 1.8, 5], 0, 0)

# Deposit 1 stuff
dep_1_actions = [EXPLORE1, MINE1]
action_sequence = [EXPLORE1, EXPLORE1, MINE1, EXPLORE1, MINE1, EXPLORE1]#[rand(dep_1_actions) for x in 1:20]

belief_history, state_history = run_sims(P, init_belief, init_state, action_sequence, rng)
times = [b.t for b in belief_history]
μs = [mean(b.deposits[1]) for b in belief_history]
σs = [std(b.deposits[1]) for b in belief_history]
true_v1s = [s.deposits[1] for s in state_history] # actual amount of Li

plot(times, μs, grid=false, ribbon=σs, fillalpha=0.5, title="Deposit 1 Belief vs. time", xlabel="Time (t)", ylabel="Amount Li in deposit 1", label="μ1", linecolor=:orange, fillcolor=:orange)
plot!(times, true_v1s, label="Actual V₁", color=:blue)


belief_history, state_history = run_sims(P, init_belief, init_state, action_sequence, rng)
times = [b.t for b in belief_history]  # Goes up at top like an iteration counter
d1_normals = [b.deposits[1] for b in belief_history]


@gif for i in 1:length(times)
    normal = d1_normals[i]
    plot(6.5:0.1:10, (x) -> pdf(normal, x), title = "Iteration $i", xlabel = "V₁ belief", label= "V₁ belief", color=:purple)
end fps = 10

# Continuous state space
function POMDPs.states(P::LiPOMDP)
    # Min and max amount per singular deposit
    V_deposit_min = 0
    V_deposit_max = 10
    
    # Min and max amount total mined, can be at smallest the deposit_min * 4, and at largest, the deposit_max * 4
    V_tot_min = V_deposit_min * P.n_sites  # 0
    V_tot_max = V_deposit_max * P.n_sites  # 40
    
    deposit_vec_bounds = [(V_deposit_min, V_deposit_max) for x in 1:P.n_sites]  # Make a length-4 vector, one for each deposit
    V_tot_bounds = Interval(V_tot_min, V_Intervaltot_max)
    time_bounds = 0:10  # This can be discrete since we're only going a year at a time
    
    𝒮 = product_state_space(deposit_vec_bounds, V_tot_bounds, time_bounds)  # Cartesian product 
    # QUESTION: how could I add the null state into the space?
    return 𝒮
    
end

# POMCPOW Solver
solver = POMCPOWSolver()
pomdp = LiPOMDP()
planner = solve(solver, pomdp)

# query an action
ap, info = action_info(planner, initialstate(pomdp), tree_in_info=true)

# show the tree using D3Tree
tree = D3Tree(info[:tree], init_expand=3)
inchrome(tree)
# beliefs = []
# trees=[]
# i=1
# ret = 0
# while !isterminal(pomdp, s)
# 	a, ai = action_info(planner, b)
# 	push!(trees, ai[:tree])
# 	println("action: ", a)
# 	sp, o, r = gen(pomdp, s, a)
# 	ret += r
	
# 	println("observation: ", o)
#    	# if a[1] in [:drill, :observe]
# 	t = @elapsed b = update(up, b, a, o)
# 	println("belief update time: ", t)
# 	push!(renders, render(pomdp, sp, a, timestep=i, belief=b))
# 	# else
# 		# b = update(up_basic, b, a, o)
# 	# end
# 	s = deepcopy(sp)
# 	# push!(belief_plots, plot_belief(b, s0, title="timestep: $i"))
# 	i=i+1
# 	if i > 50
# 		break
# 	end
# end