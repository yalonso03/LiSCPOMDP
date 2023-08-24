using Distributions
using ParticleFilters

function estimate_value(P::LiPOMDP, s, h, steps)
    return s.Vₜ < P.Vₜ_goal ? -100.0 : 0.0
end

function evaluate_policies(pomdp::LiPOMDP, policies::Vector, k::Int, max_steps::Int)

    up = LiBeliefUpdater(pomdp)
    
    for policy in policies
        println("==========start $k simulations for ", typeof(policy), "==========")

        sim_results = replicate_simulation(pomdp, policy, up, k=k, max_steps=max_steps)

        # Print results
        print_policy_results(string(typeof(policy)), sim_results)
    end
end

function print_policy_results(policy_name, simulation_results)
    println("\n$(policy_name) Results:")
    println("rdisc mean: ", simulation_results[:rdisc_mean], ", stdev: ", simulation_results[:rdisc_std])
    println("edisc mean: ", simulation_results[:edisc_mean], ", stdev: ", simulation_results[:edisc_std])
    println("rtot mean: ", simulation_results[:rtot_mean], ", stdev: ", simulation_results[:rtot_std])
    println("etot mean: ", simulation_results[:etot_mean], ", stdev: ", simulation_results[:etot_std])
    println("vt mean: ", simulation_results[:vt_mean], ", stdev: ", simulation_results[:vt_std])
    println("vol_tot mean:", simulation_results[:vol_tot_mean], ", stdev: ", simulation_results[:vol_tot_std])
end

function replicate_simulation(pomdp, policy, up; k=100, max_steps=10, rng=MersenneTwister(1), randomized=true)
    rdisc_values = Float64[]
    edisc_values = Float64[]
    rtot_values = Float64[]
    etot_values = Float64[]
    vt_values = Float64[]
    vol_tot_values = Float64[]

    for i in 1:k
        if policy isa MCTSPlanner
            println(" replication: $i")
        end
       
        if randomized
            s0 = random_initial_state(pomdp)
            b0 = random_initial_belief(s0)
        else
            s0 = pomdp.init_state
            b0 = initialize_belief(up, s0)
        end

        #println("====new rep====")
        result = simulate_policy(pomdp, policy, up, b0, s0, max_steps=max_steps, rng=rng)
        push!(rdisc_values, result.rdisc)
        push!(edisc_values, result.edisc)
        push!(rtot_values, result.rtot)
        push!(etot_values, result.etot)
        push!(vt_values, result.vt)
        push!(vol_tot_values, result.vol_total)
    end

    return Dict(
        :rdisc_mean => mean(rdisc_values),
        :rdisc_std => std(rdisc_values),
        :edisc_mean => mean(edisc_values),
        :edisc_std => std(edisc_values),
        :rtot_mean => mean(rtot_values),
        :rtot_std => std(rtot_values),
        :etot_mean => mean(etot_values),
        :etot_std => std(etot_values),
        :vt_mean => mean(vt_values),
        :vt_std => std(vt_values),
        :vol_tot_mean => mean(vol_tot_values),
        :vol_tot_std => std(vol_tot_values)
    )
end


function simulate_policy(pomdp, policy, up, b0, s0; max_steps=10, rng=MersenneTwister(1))
    r_total = 0.
    r_disc = 0.
    e_total = 0.
    e_disc = 0.
    vol_total = 0.
    t = 0
    d = 1.

    b = deepcopy(b0)
    s = deepcopy(s0)
    while (!isterminal(pomdp, s) && t < max_steps)
        t += 1        
        a = action(policy, b)
        (s, o, r) = gen(pomdp, s, a, rng)
        b = update(up, b, a, o)
        e = get_action_emission(pomdp, a)
        #println("action: $a, type: $(typeof(a)), emission: $e")
        r_total += r
        r_disc += r*d        
        e_total += e
        e_disc += e*d
        if get_action_type(a) == "MINE"
            vol_total += 1
        end
        d *= discount(pomdp)
        #@show(t=t, s=s, a=a, r=r, o=o)
    end
    return (rdisc=r_disc, edisc=e_disc, rtot=r_total, etot=e_total, vt=s.Vₜ, vol_total=vol_total)
end

# function simulate_mcts(pomdp, policy, s0; max_steps=10, rng=MersenneTwister(1))
#     r_total = 0.
#     r_disc = 0.
#     e_total = 0.
#     e_disc = 0.
#     t = 0
#     d = 1.
#     s = deepcopy(s0)
#     while (!isterminal(pomdp, s) && t < max_steps)
#         t += 1
#         a = action(policy, s)
#         (s, o, r) = gen(pomdp, s, a, rng)
#         e = get_action_emission(pomdp, a)
#         r_total += r
#         r_disc += r*d        
#         e_total += e
#         e_disc += e*d        

#         d *= discount(pomdp)
#         #@show(t=t, s=s, a=a, r=r, o=o)
#     end
#     return (rdisc=r_disc, edisc=e_disc, rtot=r_total, etot=e_total, vt=s.Vₜ)
# end


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
        b = convert_particle_collection_to_libelief(b)
    end
    
    return !b.have_mined[site_number]        
end

# Now unnecessary
# function sample_state_belief(b::POMCPOW.StateBelief{POWNodeBelief{State, Action, Any, LiPOMDP}}, bin_beliefstate=true)
    
#     b_dist = b.sr_belief.dist

#     out =  rand(b_dist)[1]
#     return out
# end

function get_all_states(b::POMCPOW.StateBelief{POWNodeBelief{State, Action, Any, LiPOMDP}})
    dist = b.sr_belief.dist
    states = [x[1] for x in dist.items]
    states = [x.deposits for x in states]
    return states
end

function get_all_states(s::State)
    return [s.deposits]
end


function compute_portion_below_threshold(P, b, idx::Int64)
    if isa(b, LiBelief)
        dist = b.deposit_dists[idx]
        portion_below_threshold = cdf(dist, P.min_n_units)
    elseif isa(b, ParticleCollection{State}) || isa(b, State)
        portion_below_threshold = 0.
    else
        sampled_belief = get_all_states(b)
        n_rows = length(sampled_belief)

        #! 
        num_below_threshold = sum(row[idx] < P.min_n_units for row in sampled_belief)
        portion_below_threshold = num_below_threshold / n_rows
    end
    return portion_below_threshold
end

# In: a particle collection Out: an LiBelief
function convert_particle_collection_to_libelief(part_collection::POMCPOW.StateBelief{POWNodeBelief{State, Action, Any, LiPOMDP}})
    items_vec = part_collection.sr_belief.dist.items # this is a vector of tuples (state, V). ignore the V value for now
    states_vec = [tup[1] for tup in items_vec] # vector of states
    deposits_vec = [state.deposits for state in states_vec] # vector of deposit vectors
    
    
    # Put all of the v1s, v2s, etc. into their own vectors so we can compute mean and std
    v1s = [deposits[1] for deposits in deposits_vec]
    v2s = [deposits[2] for deposits in deposits_vec]
    v3s = [deposits[3] for deposits in deposits_vec]
    v4s = [deposits[4] for deposits in deposits_vec]
    
    
    # Compute the estimated means from all the sample points
    μ1 = mean(v1s)
    μ2 = mean(v2s)
    μ3 = mean(v3s)
    μ4 = mean(v4s)
    
    
    # Compute the estimated standard deviations from all the sample points
    σ1 = std(v1s, corrected=true)
    σ2 = std(v2s, corrected=true)
    σ3 = std(v3s, corrected=true)
    σ4 = std(v4s, corrected=true)
    
    
    # Assert that all times are the same    
    # for state in states_vec
    #     if state.t != states_vec[1].t
    #         println("state.t : $(state.t), state_vec[1].t : $(states_vec[1].t)")
    #         println("state.Vₜ : $(state.Vₜ), state_vec[1].Vₜ : $(states_vec[1].Vₜ)")
    #     #println("state.t : $(state.t), state_vec[1].t : $(states_vec[1].t)")
    #     end
    # #    @assert(state.t == states_vec[1].t)
    # end
    #@assert(all([state.t == states_vec[1].t for state in states_vec]))
    
    
    # Create a new LiBelief with the mean, std of each Deposit, take the other fields (t, V_tot, have_mined) from the first particle
    return LiBelief([Normal(μ1, σ1), Normal(μ2, σ2), Normal(μ3, σ3), Normal(μ4, σ4)], states_vec[1].t, states_vec[1].Vₜ, [mine for mine in states_vec[1].have_mined])
    end
    
    