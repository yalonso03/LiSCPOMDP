#RANDOM POLICY
struct RandomPolicy <: Policy
    pomdp::LiPOMDP
end

function POMDPs.action(p::RandomPolicy, b::LiBelief)
    potential_actions = actions(p.pomdp, b)
    #println("random actions list: $potential_actions for $b")
    return rand(potential_actions)
end

#GREEDY EFFICIENCY POLICY 
@with_kw mutable struct EfficiencyPolicy <: Policy 
    pomdp::LiPOMDP
    need_explore::Vector{Bool}
end

function POMDPs.action(p::EfficiencyPolicy, b::LiBelief)

    #println("efficiency actions list: $(actions(p.pomdp, b)) for $b")

    # Explore all that needs exploring first
    for (index, to_explore) in enumerate(p.need_explore)
        if to_explore
            p.need_explore[index] = false
            return eval(Meta.parse("EXPLORE$(index)"))
        end
    end
    
    # If we have explored all deposits, greedily decide which one to mine that is allowed by the belief.
    scores = zeros(p.pomdp.n_deposits)
    for i in 1:p.pomdp.n_deposits
        if can_explore_here(eval(Meta.parse("MINE$(i)")), b)
            score = mean(b.deposit_dists[i])
        else
            score = -Inf
        end
        scores[i] = score
    end
    _, best_mine = findmax(scores)
    
    return eval(Meta.parse("MINE$(best_mine)"))
end


#GREEDY EFFICIENCY POLICY CONSIDERING UNCERTAINTY
@with_kw mutable struct EfficiencyPolicyWithUncertainty <: Policy 
    pomdp::LiPOMDP
    lambda::Float64  # Penalty factor for uncertainty
    need_explore::Vector{Bool}
end

function POMDPs.action(p::EfficiencyPolicyWithUncertainty, b::LiBelief)

    #println("EfficiencyPolicyWithUncertainty actions list: $(actions(p.pomdp, b)) for $b")

    # Explore all that needs exploring first
    for (index, to_explore) in enumerate(p.need_explore)
        if to_explore
            p.need_explore[index] = false
            return eval(Meta.parse("EXPLORE$(index)"))
        end
    end
    
    # If we have explored all deposits, decide which one to mine that is allowed by the belief.
    # We will consider both the expected Lithium and the uncertainty in our decision.    
    scores = zeros(p.pomdp.n_deposits)
    for i in 1:p.pomdp.n_deposits
        if can_explore_here(eval(Meta.parse("MINE$(i)")), b)
            score = mean(b.deposit_dists[i])  - p.lambda * std(b.deposit_dists[i])
        else
            score = -Inf
        end
        scores[i] = score
    end
    _, best_mine = findmax(scores)
    return eval(Meta.parse("MINE$(best_mine)"))
end


#EMISSION AWARE POLICY
@with_kw mutable struct EmissionAwarePolicy <: Policy 
    pomdp::LiPOMDP
    need_explore::Vector{Bool}
end

function POMDPs.action(p::EmissionAwarePolicy, b::LiBelief)

    #println("EmissionAwarePolicy actions list: $(actions(p.pomdp, b)) for $b")

    # Explore all that needs exploring first
    for (index, to_explore) in enumerate(p.need_explore)
        if to_explore
            p.need_explore[index] = false
            return eval(Meta.parse("EXPLORE$(index)"))
        end
    end
    
    # If we have explored all deposits, decide which one to mine.
    # We will prioritize mining the site with the most expected Lithium,
    # but also factor in emissions.

    scores = zeros(p.pomdp.n_deposits)
    for i in 1:p.pomdp.n_deposits
        if can_explore_here(eval(Meta.parse("MINE$(i)")), b)
            score = mean(b.deposit_dists[i])/p.pomdp.CO2_emissions[i]
        else
            score = -Inf
        end
        scores[i] = score
    end
    
    _, best_mine = findmax(scores)
    
    return eval(Meta.parse("MINE$(best_mine)"))
end