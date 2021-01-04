from size_based_ecosystem import *

class simulator: #Expand this class
    def __init__(self, time_step, end_time, step_by_step, animals = None, landscape = None, spectral = None, verbose = False):
        self.end_time = end_time
        self.time_step = time_step
        self.step_by_step = step_by_step
        self.verbose = verbose

        self.animals = animals
        self.landscape = landscape
        self.spectral = spectral
        self.animals.landscape = landscape

        self.graph = np.zeros((self.animals.populations.shape[0]+1, self.animals.populations.shape[0]))


    def simulate_iterative(self):
        frozen_ecos = []
        size_classes = 1
        stability = False
        list_of_sizes = self.animals.parameters.mass_vector

        while size_classes < len(list_of_sizes) + 1:

            x_res = self.sequential_nash(self.animals, verbose=self.verbose)
            pop_old = np.copy(self.animals.populations)
            new_pop = self.animals.total_growth(x_res) * self.time_step + self.animals.populations
            error = np.linalg.norm(pop_old - new_pop)

            if error > 0.01:
                self.time_step = max(0.75 * self.time_step, 10 ** (-12))
            else:
                self.time_step = min(5 / 4 * self.time_step, 10 ** (-5))
                self.animals.population_setter(eco.total_growth(x_res) * self.time_step + eco.populations)
                self.animals.strategy_setter(x_res)
                self.landscape.update_resources(consumed_resources=eco.consumed_resources(), time_step=self.time_step)

            if error / self.time_step < min(1 / 10, np.min(self.animals.populations / 2)):
                stability = True

            if stability is True:
                old_eco = copy.deepcopy(eco)
                strat_old = np.copy(eco.strategy_matrix)

                print("New regime")
                frozen_ecos.append(old_eco)
                pops = np.copy(old_eco.populations)
                size_classes += 1
                m_v_t = np.copy(list_of_sizes[0:size_classes])

                params = ecosystem_parameters(m_v_t, self.spectral)
                print(params.forager_or_not)
                eco = ecosystem_optimization(m_v_t, self.landscape.layers, params, self.spectral, self.landscape.start_value, l2=self.animals.l2)
                eco.strategy_matrix[0:size_classes - 1] = strat_old
                eco.populations[0:size_classes - 1] = pops
                eco.populations[-1] = 10 ** (-10)
                eco.parameters.handling_times = np.array([0, 0])

                stability = False

        with open('eco_systems.pkl', 'wb') as f:
            pkl.dump(frozen_ecos, f, pkl.HIGHEST_PROTOCOL)


def sequential_nash(eco, verbose = False, l2 = False, max_its_seq = None, time_step = 10**(-4), max_mem = None):
    x_temp = np.copy(eco.strategy_matrix)
    smooth_xtemp = np.copy(x_temp)
    x_temp2 = np.copy(eco.strategy_matrix)
    smooth_xtemp2 = np.copy(x_temp2)
    error = 1
    iterations = 0

    if max_mem is None:
        max_mem = eco.populations.shape[0]**eco.populations.shape[0]+eco.populations.shape[0]
    if max_its_seq is None:
        max_its_seq = 4*max_mem

    memory_keeper = np.zeros((max_mem, eco.strategy_matrix.shape[0], eco.strategy_matrix.shape[1]))
    mem = 0
    while error > 10 ** (-6) and iterations <= max_its_seq:
        for k in range(eco.mass_vector.shape[0]):
            result = eco.one_actor_growth(k, x_temp_i=x_temp, time_step=time_step)
            x_temp2[k] = np.array(result).flatten()
            if verbose is True:
                print(np.sum(np.dot(eco.ones, np.dot(eco.spectral.M, result))), k, "1 Norm of strategy")
                print(np.sum(np.dot(eco.ones, np.dot(eco.spectral.M, smooth_xtemp[k]))), k, "1 Norm of old smooth strategy")

        if l2 is False:
            for k in range(eco.populations.shape[0]):
                smooth_xtemp[k] = x_temp[k] @ eco.heat_kernels[k]  # Going smooth.
                smooth_xtemp2[k] = x_temp2[k] @ eco.heat_kernels[k]  # Going smooth.

            errors = np.sum(np.dot(np.abs(smooth_xtemp2 - smooth_xtemp), eco.spectral.M), axis = 1)
            error = np.max(errors)
            print(errors.shape)

        else:
            print(np.dot((x_temp2 - x_temp), eco.spectral.M).shape)
            error = np.max(np.dot((smooth_xtemp - smooth_xtemp2), eco.spectral.M) @ (x_temp2 - x_temp).T)

        if verbose is True:
            print("Error: ", error, errors)
        good_mem = 0
        mem_err = 1
        if error > 10**(-6):
            for k in range(memory_keeper.shape[0]):
                mem_err = np.sum(np.abs(memory_keeper[k] - x_temp2))
                if mem_err < 10**(-6):
                    good_mem = k

        if mem_err<10**(-6):
            bottom = min(mem, good_mem)
            top = max(mem, good_mem)
            if top == bottom:
                top = max_mem
                bottom = 0
            else:
                top += 1
            x_temp = np.mean(memory_keeper[bottom: top], axis = 0)
            print(x_temp, "Memory used for mean", memory_keeper[bottom: top], "Total memory",  memory_keeper)
            print("Memory working", top, bottom)
            error = mem_err
        else:
            x_temp = np.copy(x_temp2)

         #(error_col) * x_temp2 + (1-1/error_col)*x_temp #np.copy(x_temp2)
        memory_keeper[mem] = np.copy(x_temp2)
        mem += 1
        mem = int(mem % max_mem)
        #print(mem)
        iterations += 1
    iterations_newt = 0
    if iterations > max_its_seq and error > 10**(-6):
        print("Entering newton-phase for strategy mixing")


    return x_temp


def total_payoff_matrix_builder(eco):
    total_payoff_matrix = np.zeros((eco.populations.size*eco.layers, eco.populations.size*eco.layers))
    for i in range(eco.populations.size):
        for j in range(eco.populations.size):
            if i != j:
                i_vs_j = payoff_matrix_builder(eco, i, j)
            elif i == j:
                i_vs_j = np.zeros((eco.layers, eco.layers))
            #if i == 1:
            #    total_payoff_matrix[i*eco.layers:(i+1)*eco.layers, j*eco.layers: (j+1)*eco.layers] = i_vs_j.T
            #else:

            total_payoff_matrix[i * eco.layers:(i + 1) * eco.layers, j * eco.layers: (j + 1) * eco.layers] = i_vs_j

    total_payoff_matrix = total_payoff_matrix - np.max(total_payoff_matrix) - 0.00001 #Making sure everything is negative
    return total_payoff_matrix

def payoff_matrix_builder(eco, i, j):
    payoff_i = np.zeros((eco.layers, eco.layers))

    for k in range(eco.layers):
        one_k_vec = np.zeros(eco.layers)
        one_k_vec[k] = 1
        for n in range(eco.layers):
            one_n_vec = np.zeros(eco.layers)
            one_n_vec[n] = 1
            strat_mat = np.vstack([one_k_vec, one_n_vec])
            payoff_i[k, n] = eco.lin_growth(i, j, strat_mat)

    return payoff_i



def quadratic_optimizer(eco, payoff_matrix = None, prior_sol=None):

    A=np.zeros((eco.populations.size, eco.populations.size*eco.layers))
    for k in range(eco.populations.size):
        A[k,k*eco.layers:(k+1)*eco.layers] = -1

    q = np.zeros(eco.populations.size+eco.populations.size*eco.layers)
    q[eco.populations.size*eco.layers:] = -1
    q = q.reshape(-1, 1)
    if payoff_matrix is None:
        payoff_matrix = total_payoff_matrix_builder(eco)
        print("IM HERE!!!")

    p = ca.SX.sym('p', eco.populations.size*eco.layers)
    y = ca.SX.sym('y', eco.populations.size)
    u = ca.SX.sym('u', eco.populations.size*eco.layers)
    v = ca.SX.sym('v', eco.populations.size)

    cont_conds = []
    if eco.spectral.segments>1:
        for j in range(eco.spectral.segments-1):
            cont_conds.append(p[j*eco.spectral.n] - p[(j+1)*eco.spectral.n])
            cont_conds.append(u[j*eco.spectral.n] - u[(j+1)*eco.spectral.n])


    z = ca.vertcat(*[p, y])
    w = ca.vertcat(*[u, v])
#    print(np.block([-A]).shape)
    H = np.block([[-payoff_matrix, A.T], [-A, np.zeros((A.shape[0], eco.populations.size))]])
#    print(H.shape)

    f = ca.dot(w, z)
    if eco.spectral.segments > 1:
        g = ca.vertcat(*[*cont_conds, w - H @ z - q])

    else:
        g = w - H @ z - q #ca.norm_2()



    x = ca.vertcat(z, w)
    lbx = np.zeros(x.size())
    ubg = np.zeros(g.size())
    lbg = np.zeros(g.size())


    s_opts = {'ipopt': {'print_level': 5}}
    prob = {'x': x, 'f': f, 'g': g}
    solver = ca.nlpsol('solver', 'ipopt', prob, s_opts)
    #prior_sol = False
    if prior_sol is None:
        sol = solver(lbx=lbx, ubg=ubg, lbg=lbg)

    #    x0 = np.zeros(x.size())
    #    x0 = x0.flatten()
    #    x0[0:eco.layers * eco.populations.size] = eco.strategy_matrix.flatten()
    #    sol = solver(x0 = x0, lbx=lbx, ubg=ubg, lbg=lbg)
    #    print(prior_sol)

    else:
        sol = solver(x0=prior_sol, lbx=lbx, ubg=ubg, lbg=lbg)



    x_out = np.array(sol['x']).flatten()
    print(np.min(x_out), np.dot(x_out[0:eco.populations.size*(eco.layers+1)], x_out[eco.populations.size*(eco.layers+1):]))
    #print(x_out[0:eco.layers*eco.populations.size])
    return x_out


def graph_builder_old(eco):  # Move into ecosystem class
    strat_mat = eco.strategy_matrix

    outflows = np.zeros((eco.mass_vector.shape[0]+1, eco.mass_vector.shape[0]))
    for i in range(eco.mass_vector.shape[0]):
        for k in range(eco.parameters.who_eats_who.shape[0]):
            interaction_term = eco.parameters.who_eats_who[k] * eco.populations * eco.parameters.handling_times[k] * \
                               eco.parameters.clearance_rate[k]
            layer_action = strat_mat[i, :].reshape((eco.layers, 1)) * eco.parameters.layered_attack[:, i, :] * (
                        interaction_term.reshape((eco.populations.shape[0], 1)) * strat_mat).T

            foraging_term = eco.water.res_counts * eco.parameters.forager_or_not[k] * \
                            eco.parameters.handling_times[k] \
                            * eco.parameters.clearance_rate[k] * eco.parameters.layered_foraging[:, k]
            # print(eco.parameters.layered_attack[:,k,i], k, i, strat_mat[i])
            outflows[1+i, k] = np.dot(strat_mat[k], np.dot(eco.spectral.M, strat_mat[i] * eco.populations[
                k] * eco.parameters.layered_attack[:, k, i] * eco.parameters.clearance_rate[k] * eco.populations[
                                                             i])) / \
                             (1 + np.sum(np.dot(eco.ones, np.dot(eco.spectral.M,
                                                                 np.sum(layer_action, axis=1) + foraging_term))))
            outflows[0,k] =  np.sum(np.dot(eco.spectral.M, foraging_term)/(1 + np.sum(np.dot(eco.ones, np.dot(eco.spectral.M,
                                                                 np.sum(layer_action, axis=1) + foraging_term)))))
    return outflows


def graph_builder(eco, layered_attack=None, populations=None, resources=None, strategies=None):  # Move into ecosystem class
    if layered_attack is None:
        layered_attack = eco.parameters.layered_attack
    if populations is None:
        populations = eco.populations
    if resources is None:
        resources = eco.water.res_counts
    if strategies is None:
        strategies = np.copy(eco.strategy_matrix)

    classes = populations.size
    inflows = np.zeros((classes + 1, classes + 1))
    x_temp = np.zeros((2, eco.layers))

    for i in range(classes):
        x_temp[0] = strategies[i] @ eco.heat_kernels[i]  # Going smooth.
        inflows[i + 1, 0] = (eco.parameters.forager_or_not[i] * eco.parameters.clearance_rate[
            i] * eco.parameters.layered_foraging[:, i] * resources) @ (eco.spectral.M @ x_temp[0])

        for j in range(classes):
            x_temp[1] = strategies[j] @ eco.heat_kernels[j]  # Going smooth.
            x = x_temp[0].reshape(-1, 1)
            interaction_term = eco.parameters.who_eats_who[i, j] * eco.parameters.clearance_rate[i]
            lin_growth = interaction_term * (x_temp[1] * layered_attack[:, i, j].T) @ eco.spectral.M @ x

            actual_growth = eco.parameters.efficiency * lin_growth

            inflows[i + 1, j + 1] = actual_growth * populations[i] * populations[j]

    return inflows


def periodic_attack(layered_attack, day_interval = 96, darkness_length = 0, minimum_attack = 0):
    OG_layered_attack = np.copy(layered_attack)
    periodic_layers = np.zeros((day_interval,*layered_attack.shape))
    for i in range(day_interval):
        periodic_layers[i] = 1/2*(1+minimum_attack+min(max((darkness_length+1)*np.cos(i*2*np.pi/day_interval), -1), 1))*OG_layered_attack

    return periodic_layers

def reward_loss_time_dependent(eco, periodic_layers):
    rewards_t = np.zeros((periodic_layers.shape[0], eco.layers*eco.populations.size, eco.layers*eco.populations.size))
    losses_t = np.zeros((periodic_layers.shape[0], eco.layers*eco.populations.size, eco.layers*eco.populations.size))
    for i in range(periodic_layers.shape[0]):
        print(i)
        reward_i, loss_i = loss_and_reward_builder(eco, periodic_layers[i])
        rewards_t[i] = reward_i
        losses_t[i] = loss_i

    return rewards_t, losses_t


def total_payoff_matrix_builder_memory_improved(eco, populations, total_reward_matrix, total_loss_matrix, foraging_gain):
    total_rew_mat = np.copy(total_reward_matrix)
    total_loss_mat = np.copy(total_loss_matrix)

    for i in range(eco.populations.size):
        for j in range(eco.populations.size):
            total_rew_mat[i * eco.layers:(i + 1) * eco.layers, j * eco.layers: (j + 1) * eco.layers] = eco.parameters.efficiency*populations[j]*total_reward_matrix[i * eco.layers:(i + 1) * eco.layers, j * eco.layers: (j + 1) * eco.layers]
            total_loss_mat[i * eco.layers:(i + 1) * eco.layers, j * eco.layers: (j + 1) * eco.layers] = populations[j]*total_loss_matrix[i * eco.layers:(i + 1) * eco.layers, j * eco.layers: (j + 1) * eco.layers]

    total_payoff_matrix = total_rew_mat + foraging_gain - total_loss_mat

    return total_payoff_matrix - np.max(total_payoff_matrix) - 0.0001

def foraging_gain_builder(eco, resources = None):
    if resources is None:
        resources = eco.water.res_counts

    foragers = np.where(eco.parameters.forager_or_not == 1)
    foraging_gain = np.zeros((eco.populations.size*eco.layers, eco.populations.size*eco.layers))
    for forager in foragers:
        foraging_gain_i = eco.heat_kernels[forager[0]] @ (eco.spectral.M @ resources)/np.sum(eco.parameters.who_eats_who[:, forager[0]])
        #print(foraging_gain_i, np.sum(eco.parameters.who_eats_who[:, forager[0]]), eco.heat_kernels[forager[0]])
        np.vstack([foraging_gain_i]*eco.layers)
        eaters = np.where(eco.parameters.who_eats_who[:,forager] == 1)
        for eater in eaters:
            foraging_gain[forager[0] * eco.layers:(eater[0] + 1) * eco.layers, eater[0] * eco.layers: (eater[0] + 1) * eco.layers] = foraging_gain_i

    return foraging_gain


def loss_and_reward_builder(eco, layered_attack = None):
    if layered_attack is None:
        layered_attack = eco.parameters.layered_attack

    total_reward_matrix = np.zeros((eco.populations.size*eco.layers, eco.populations.size*eco.layers))
    total_loss_matrix = np.zeros((eco.populations.size*eco.layers, eco.populations.size*eco.layers))
    for i in range(eco.populations.size):
        for j in range(eco.populations.size):
            if i != j:
                i_vs_j = reward_matrix_builder(eco, layered_attack, i, j)
                j_vs_i = loss_matrix_builder(eco, layered_attack, i, j)
            elif i == j:
                i_vs_j = np.zeros((eco.layers, eco.layers))
                j_vs_i = np.zeros((eco.layers, eco.layers))

            total_reward_matrix[i * eco.layers:(i + 1) * eco.layers, j * eco.layers: (j + 1) * eco.layers] = i_vs_j
            total_loss_matrix[i * eco.layers:(i + 1) * eco.layers, j * eco.layers: (j + 1) * eco.layers] = j_vs_i

    return total_reward_matrix, total_loss_matrix


def reward_matrix_builder(eco, layered_attack, i, j):
    reward_i = np.zeros((eco.layers, eco.layers))

    for k in range(eco.layers):
        one_k_vec = np.zeros(eco.layers)
        one_k_vec[k] = 1
        for n in range(eco.layers):
            one_n_vec = np.zeros(eco.layers)
            one_n_vec[n] = 1
            strat_mat = np.vstack([one_k_vec, one_n_vec])
            reward_i[k, n] = lin_growth_no_pops_no_res(eco, i, j, layered_attack, strat_mat)

    return reward_i


def loss_matrix_builder(eco, layered_attack, i, j):
    loss_i = np.zeros((eco.layers, eco.layers))

    for k in range(eco.layers):
        one_k_vec = np.zeros(eco.layers)
        one_k_vec[k] = 1
        for n in range(eco.layers):
            one_n_vec = np.zeros(eco.layers)
            one_n_vec[n] = 1
            strat_mat = np.vstack([one_k_vec, one_n_vec])
            loss_i[k, n] = lin_growth_no_pops_no_res(eco, j, i, layered_attack, strat_mat)

    return loss_i