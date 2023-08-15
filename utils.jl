using Distributions
using ParticleFilters


function eval_policy(P, hist)
    tot_r = 0.0
    tot_emission = 0.0
    for (s, a, b, o, r) in hist
        #@show s, a, r
        tot_r = r
        tot_emission -= get_action_emission(P, a)
    end

    (s, a, b, o, r)  = hist[end]
    tot_V = s.Vₜ

    return (discounted_r = round(discounted_reward(hist), digits=2), tot_r=round(tot_r, digits=2), tot_emission=tot_emission, tot_V=tot_V)
end
function compute_chunk_boundaries(quantile_vols::Vector{Float64})
    n = length(quantile_vols)
    @assert n > 0 "quantile_vols must not be empty"

    chunk_boundaries = Vector{Float64}(undef, n-1)
    for i = 2:n
        chunk_boundaries[i-1] = (quantile_vols[i] + quantile_vols[i-1]) / 2
    end
    return chunk_boundaries
end

function compute_chunk_probs(chunk_boundaries::Vector{Float64}, site_dist::Normal)
    n = length(chunk_boundaries)
    @assert n > 0 "chunk_boundaries must not be empty"

    chunk_probs = Vector{Float64}(undef, n+1)

    chunk_probs[1] = cdf(site_dist, chunk_boundaries[1])
    for i = 2:n
        chunk_probs[i] = cdf(site_dist, chunk_boundaries[i]) - cdf(site_dist, chunk_boundaries[i-1])
    end
    chunk_probs[n+1] = 1 - cdf(site_dist, chunk_boundaries[n])

    return chunk_probs
end

function get_action_emission(P, a)
    action_type = get_action_type(a)
    action_number = get_site_number(a)
    
    # Subtract carbon emissions (if relevant)
    r3 = (action_type == "MINE") ? P.CO2_emissions[action_number] * -1 : 0
    return r3
end

function get_site_number(a::Action)
    action_str = string(a)
    len = length(action_str)
    deposit_number = Int(action_str[len]) - 48  # -48 because Int() gives me the ascii code
    return deposit_number
end

# I'm sure there's some builtin for this but I couldn't find it lol
function splice(begin_index, end_index, str)
    result = ""
    for i = begin_index:end_index
        result = result * str[i]
    end
    return result
end

function get_action_type(a::Action)
    action_str = string(a)
    len = length(action_str)
    action_type = splice(1, len - 1, action_str)
    return action_type
end

function str_to_action(s::String)
    if s == "MINE1"
        return MINE1
    elseif s == "MINE2"
        return MINE2
    elseif s == "MINE3"
        return MINE3
    elseif s == "MINE4"
        return MINE4
    elseif s == "EXPLORE1"
        return EXPLORE1
    elseif s == "EXPLORE2"
        return EXPLORE2
    elseif s == "EXPLORE3"
        return EXPLORE3
    else
        return EXPLORE4
    end
end

function can_explore_here(a::Action, b::Any)
    action_type = get_action_type(a)
    site_number = get_site_number(a)

    if action_type == "MINE" || isa(b, ParticleFilters.ParticleCollection{State})
        return true
    end

    if isa(b, POMCPOW.StateBelief{POWNodeBelief{State, Action, Any, LiPOMDP}})
        b = sample_state_belief(b)
    end
    
    return !b.have_mined[site_number]        
end


function sample_state_belief(b::POMCPOW.StateBelief{POWNodeBelief{State, Action, Any, LiPOMDP}}, bin_beliefstate=true)
    
    b_dist = b.sr_belief.dist

    function bin_elements(v)
        return [x < 4 ? 2 : (x <= 8 ? 6 : 10) for x in v]
    end
    

    if bin_beliefstate
        new_state = deepcopy(b_dist.items)
        for (i, state) in enumerate(b_dist.items)
            b_dist.items[i][1].deposits = bin_elements(state[1].deposits)
        end
    end

    out =  rand(b_dist)[1]
    return out
end

function get_all_states(b::POMCPOW.StateBelief{POWNodeBelief{State, Action, Any, LiPOMDP}})
    dist = b.sr_belief.dist
    states = [x[1] for x in dist.items]
    states = [x.deposits for x in states]
    return states
end

function compute_portion_below_threshold(P, b, idx::Int64)
    if isa(b, LiBelief)
        dist = b.deposit_dists[idx]
        portion_below_threshold = cdf(dist, P.min_n_units)
    elseif isa(b, ParticleCollection{State})
        portion_below_threshold = 0.
    else
        sampled_belief = get_all_states(b)
        n_rows = length(sampled_belief)
        num_below_threshold = sum(row[idx] < P.min_n_units for row in sampled_belief)
        portion_below_threshold = num_below_threshold / n_rows
    end
    return portion_below_threshold
end

function next_action(
    P,
    b,
    h, 
    a::Union{Action, Nothing}=nothing, 
    arg::Union{Any, Nothing}=nothing,  
    b0::Union{LiBelief, Nothing}=nothing
)

    potential_actions = [MINE1, MINE2, MINE3, MINE4, EXPLORE1, EXPLORE2, EXPLORE3, EXPLORE4]

    # Checks to ensure that we aren't trying to explore at a site we have already mined at
    potential_actions = filter(a -> can_explore_here(a, b), potential_actions)

    for i = 1:4
        portion_below_threshold = compute_portion_below_threshold(P, b, i)
        if portion_below_threshold > P.cdf_threshold  # BAD!
            bad_action_str = "MINE" * string(i)
            bad_action = str_to_action(bad_action_str)
            # Ensure that this bad_action is not in potential_actions
            potential_actions = filter(a -> a != bad_action, potential_actions)
        end   
    end 
    
    return rand(potential_actions)
end
