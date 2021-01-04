import pickle as pkl
import copy as copy

def population_growth(eco, populations, resources, payoff_matrix = None, lemke = True, time_step = 1/(96*365)):
    eco_temp = copy.deepcopy(eco)
    eco_temp.populations = populations
    eco_temp.water.res_counts = resources
    if payoff_matrix is None:
        payoff_matrix = total_payoff_matrix_builder(eco)
    if lemke is True:
        prior_sol = lemke_optimizer(eco, payoff_matrix=payoff_matrix)
    else:
        prior_sol = quadratic_optimizer(eco, payoff_matrix=payoff_matrix)
    x_res = (prior_sol[0:eco.populations.size * eco.layers]).reshape((eco.populations.size, -1))


    delta_pop = eco.total_growth(x_res)

    eco_temp.population_setter(delta_pop * time_step + eco.populations)
    eco_temp.strategy_setter(x_res)
    eco_temp.water.update_resources(consumed_resources=eco.consumed_resources(), time_step=time_step)

    return eco.populations, eco.water.res_counts, x_res

def fix_point_finder(eco, day_interval, x0, resources0, min_attack_rate, darkness_length):
    error = 1
    populations = np.copy(x0)
    resources = np.copy(resources0)
    i = 0
    periodic_layers = periodic_attack(eco.parameters.layered_attack, day_interval=day_interval, minimum_attack=min_attack_rate,
                                      darkness_length= darkness_length)
    reward_t, loss_t = reward_loss_time_dependent(eco, periodic_layers=periodic_layers)

    while error > 10**(-6):
        if i is 0:
            pops_start, res_start, strat_start = population_growth(eco, populations, resources)
        if i == day_interval-1:
            pops_end, res_end, strat_end = population_growth(eco, populations, resources)
            error = np.linalg.norm(pops_end-pops_start)

        current_foraging = foraging_gain_builder(eco, resources)
        current_reward = reward_t[i % day_interval]
        current_loss = loss_t[i % day_interval]

        payoff_matrix = total_payoff_matrix_builder_memory_improved(eco, populations,
                                                                    total_reward_matrix=current_reward,
                                                                    total_loss_matrix=current_loss,
                                                                    foraging_gain=current_foraging)

        populations, resources, x_res = population_growth(eco, populations, resources, payoff_matrix=payoff_matrix)
        if error > 10**(-6):
            i = (i + 1) % day_interval


def simulator(eco, params, filename, h_k = None, lemke = True, min_attack_rate = 10**(-4), total_days = 365, day_interval = 96, darkness_length = 2):
    population_list = []
    resource_list = []
    strategy_list = []

    time_step = (1 / 365) * (1 / day_interval)

    if h_k is None:
        h_k = heat_kernel(eco.spectral, time_step, 90000)

    periodic_layers = periodic_attack(params.layered_attack, day_interval=day_interval, minimum_attack=min_attack_rate,
                                      darkness_length= darkness_length)

    reward_t, loss_t = reward_loss_time_dependent(eco, periodic_layers=periodic_layers)
    total_time_steps = int(total_days * day_interval)  # Yup
    time = 0
    for i in range(total_time_steps):
        current_reward = reward_t[i % day_interval]
        current_loss = loss_t[i % day_interval]
        current_foraging = foraging_gain_builder(eco)
        payoff_matrix = total_payoff_matrix_builder_memory_improved(eco, eco.populations,
                                                                    total_reward_matrix=current_reward,
                                                                    total_loss_matrix=current_loss,
                                                                    foraging_gain=current_foraging)

        pop_old = np.copy(eco.populations)
        population_list.append(pop_old)
        eco.parameters.layered_attack = periodic_layers[i % day_interval]
        if lemke is True:
            prior_sol = lemke_optimizer(eco, payoff_matrix=payoff_matrix)
        else:
            prior_sol = quadratic_optimizer(eco, payoff_matrix=payoff_matrix)
        x_res = (prior_sol[0:eco.populations.size * eco.layers]).reshape((eco.populations.size, -1))
        strategy_list.append(x_res)

        delta_pop = eco.total_growth(x_res)
        new_pop = delta_pop * time_step + eco.populations
        error = np.linalg.norm(new_pop - pop_old)

        eco.population_setter(eco.total_growth(x_res) * time_step + eco.populations)
        eco.strategy_setter(x_res)
        r_c = np.copy(eco.water.res_counts)
        resource_list.append(r_c)

        eco.water.update_resources(consumed_resources=eco.consumed_resources(), time_step=time_step)
        eco.water.res_counts = eco.water.res_counts @ h_k

        print(error, eco.populations, np.sum(eco.water.res_counts), time_step, new_pop - pop_old,
              np.cos(i * 2 * np.pi / day_interval), i / total_time_steps)
        time += time_step

    with open('eco'+filename+'.pkl', 'wb') as f:
        pkl.dump(eco, f, pkl.HIGHEST_PROTOCOL)

    with open('strategies' + filename + '.pkl', 'wb') as f:
        pkl.dump(strategy_list, f, pkl.HIGHEST_PROTOCOL)

    with open('population' + filename + '.pkl', 'wb') as f:
        pkl.dump(population_list, f, pkl.HIGHEST_PROTOCOL)

    with open('resource' + filename + '.pkl', 'wb') as f:
        pkl.dump(resource_list, f, pkl.HIGHEST_PROTOCOL)

    with open('rewards' + filename + '.pkl', 'wb') as f:
        pkl.dump(reward_t, f, pkl.HIGHEST_PROTOCOL)

    with open('losses'  + filename + '.pkl', 'wb') as f:
        pkl.dump(loss_t, f, pkl.HIGHEST_PROTOCOL)

    with open('periodic'  + filename + '.pkl', 'wb') as f:
        pkl.dump(periodic_layers, f, pkl.HIGHEST_PROTOCOL)

