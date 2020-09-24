from size_based_ecosystem import *



def periodic_attack(layered_attack, day_interval = 96):
    OG_layered_attack = np.copy(layered_attack)
    periodic_layers = np.zeros((day_interval,*layered_attack.shape))
    for i in range(day_interval):
        periodic_layers[i] = 1/2*(1+np.cos(i*2*np.pi/day_interval))*OG_layered_attack

    return periodic_layers

def reward_loss_time_dependent(eco, periodic_layers):
    rewards = []
    losses = []
    for i in range(periodic_layers.shape[0]):
        reward_i, loss_i = loss_and_reward_builder(eco, periodic_layers[i])

        rewards.append[reward_i]
        losses.append[loss_i]

    return rewards, losses


def total_payoff_matrix_builder_memory_improved(eco, populations, total_reward_matrix, total_loss_matrix, foraging_gain):
    total_rew_mat = np.copy(total_reward_matrix)
    total_loss_mat = np.copy(total_loss_matrix)

    for i in range(eco.populations.size):
        for j in range(eco.populations.size):
            total_rew_mat[i * eco.layers:(i + 1) * eco.layers, j * eco.layers: (j + 1) * eco.layers] = eco.parameters.efficiency*total_reward_matrix[i * eco.layers:(i + 1) * eco.layers, j * eco.layers: (j + 1) * eco.layers]
            total_loss_mat[i * eco.layers:(i + 1) * eco.layers, j * eco.layers: (j + 1) * eco.layers] = populations[j]*total_loss_matrix[i * eco.layers:(i + 1) * eco.layers, j * eco.layers: (j + 1) * eco.layers]

    total_payoff_matrix = total_rew_mat + foraging_gain - total_loss_mat

    return total_payoff_matrix - np.max(total_payoff_matrix) - 0.0001

def foraging_gain_builder(eco, resources = None):
    if resources is None:
        resources = eco.water.res_counts

    foragers = np.where(eco.parameters.forager_or_not == 1)
    foraging_gain = np.zeros((eco.populations.size*eco.layers, eco.populations.size*eco.layers))
    for forager in foragers:
        foraging_gain_i = eco.heat_kernels[forager] @ eco.spectral.M @ resources/np.sum(eco.parameters.who_eats_who[:, foragers])
        np.vstack([foraging_gain_i]*eco.layers)
        eaters = np.where(eco.parameters.who_eats_who[:,forager] == 1)
        for eater in eaters:
            foraging_gain[forager * eco.layers:(eater + 1) * eco.layers, eater * eco.layers: (eater + 1) * eco.layers] = foraging_gain_i

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