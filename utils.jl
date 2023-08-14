function sample_state_belief(b::POMCPOW.StateBelief{POWNodeBelief{State, Action, Any, LiPOMDP}})
    b_dist = b.sr_belief.dist
    return rand(b_dist)[1]
end

function get_all_states(b::POMCPOW.StateBelief{POWNodeBelief{State, Action, Any, LiPOMDP}})
    dist = b.sr_belief.dist
    states = [x[1] for x in dist.items]
    states = [x.deposits for x in states]
    return states
end

function compute_portion_below_threshold(P::LiPOMDP, b::Union{LiBelief, POMCPOW.StateBelief{POWNodeBelief{State, Action, Any, LiPOMDP}}}, idx::Int64)
    if isa(b, LiBelief)
        dist = b.deposit_dists[idx]
        portion_below_threshold = cdf(dist, P.min_n_units)
    else            
        sampled_belief = get_all_states(b)
        n_rows = length(sampled_belief)
        num_below_threshold = sum(row[idx] < P.min_n_units for row in sampled_belief)
        portion_below_threshold = num_below_threshold / n_rows
    end
    return portion_below_threshold
end

function next_action(
    P::LiPOMDP, 
    b::Union{LiBelief, State, POMCPOW.StateBelief{POWNodeBelief{State, Action, Any, LiPOMDP}}},
    h::POWTreeObsNode{POWNodeBelief{State, Action, Any, LiPOMDP}}, 
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
