def hill_tot_g(self, l, x_input):
    # temp_mat = np.zeros((self.populations.shape[0], self.populations.shape[0]))
    # temp_mat[l, l] = 1
    # x_temp_0 = np.copy(self.strategy_matrix)
    # x_temp_0[l] = 0
    x_temp = []
    x_tot = ca.SX.sym('x_tot', self.layers)
    for i in range(self.populations.shape[0]):
        if i != l:
            x_temp.append(x_input[i] @ self.heat_kernels[i])
        elif i == l:
            x_temp.append(x_tot.T @ self.heat_kernels[i])

    x_temp = ca.horzcat(*x_temp)
    x_temp = x_temp.T
    total_loss = 0
    for i in range(self.populations.shape[0]):
        temp_pops = np.copy(self.populations)
        temp_pops[i] = 0
        x_i = x_temp[i, :]

        predation_ext_food = []  # ca.zeros(self.populations.shape[0])
        predator_hunger = []  # ca.zeros((self.populations.shape[0], self.layers))
        for k in range(self.parameters.who_eats_who.shape[0]):
            interaction_term = self.parameters.who_eats_who[k] * temp_pops * self.parameters.clearance_rate[k]

            lin_g_others = ca.dot(
                (x_temp * self.parameters.layered_attack[:, k, :].T @ self.spectral.M @ x_temp[k, :].T),
                interaction_term)
            foraging_term = self.water.res_counts * self.parameters.forager_or_not[k] \
                            * self.parameters.clearance_rate[k] * self.parameters.layered_foraging[:, k] * x_temp[k,
                                                                                                           :].T

            predation_ext_food.append(ca.dot(self.spectral.M @ foraging_term, self.ones) + lin_g_others)
            predator_hunger.append(self.parameters.clearance_rate[k] * (self.spectral.M @
                                                                        self.parameters.layered_attack[:, k,
                                                                        i] * x_temp[k, :].T) *
                                   self.parameters.who_eats_who[k, i])

        predation_ext_food = ca.horzcat(*predation_ext_food)
        predator_hunger = ca.horzcat(*predator_hunger)
        interaction_term = self.parameters.who_eats_who[i] * self.populations * self.parameters.clearance_rate[i]

        lin_growth = ca.dot(interaction_term,
                            (x_temp * self.parameters.layered_attack[:, i, :].T) @ self.spectral.M @ x_i.T)

        foraging_term_self = (self.water.res_counts * self.parameters.forager_or_not[i] *
                              self.parameters.clearance_rate[i] * self.parameters.layered_foraging[:, i]).reshape(
            (1, -1)) @ (self.spectral.M @ x_i.T)

        A = np.tri(self.layers)
        cum_diff = A @ (self.spectral.M @ (self.strategy_matrix[i].reshape(x_i.size()) - x_i).T)

        actual_growth = self.parameters.efficiency * (lin_growth + foraging_term_self) \
                        / (1 + self.parameters.handling_times[i] * (
                    lin_growth + foraging_term_self)) - self.movement_cost * cum_diff.T @ self.spectral.M @ cum_diff

        pred_loss = ca.dot(((x_i @ predator_hunger) / (1 + self.parameters.handling_times.reshape((1, -1)) * (
                    self.populations[i] * (x_i @ predator_hunger) + predation_ext_food))).T,
                           self.populations)  # ca.dot((x.T @ predator_hunger.T).T, self.populations)   This snippet is for the linear case, known good

        # max_i = self.one_actor_growth(i, x_input).full().flatten()
        # strat_temporary = np.copy(x_input)
        # strat_temporary[i] = max_i
        # max_gain_i = self.one_actor_growth(i, strat_temporary)
        # regret_i_theta = self.one_actor_growth(i, strat_temporary) - self.one_actor_growth(i, x_input, solve_mode=False)

        total_loss = total_loss - (
                    (actual_growth - pred_loss) - self.one_actor_growth(i, x_input, solve_mode=False))  # Note the signs

    one_vec = np.zeros((1, self.populations.shape[0]))
    one_vec[0, l] = 1
    g = ca.dot(self.ones, self.spectral.M @ (x_tot @ self.heat_kernels[l])) - 1
    lbg = 0
    ubg = 0
    lbx = 0 * self.strategy_matrix[i]

    s_opts = {'ipopt': {'print_level': 0}}
    prob = {'x': x_tot, 'f': total_loss, 'g': g}
    solver = ca.nlpsol('solver', 'ipopt', prob, s_opts)
    sol = solver(x0=self.strategy_matrix[i], lbx=lbx, lbg=lbg, ubg=ubg)
    x_out = sol['x']

    return x_out.full(), np.array(-sol['f'].full())

def hillclimb_nash(eco, verbose = False, l2 = False):
    x_temp = np.copy(eco.strategy_matrix)  # np.zeros(size_classes*layers)
    error = 1
    iterations = 0
    best_matrices = np.zeros((eco.mass_vector.shape[0], eco.strategy_matrix.shape[0], eco.strategy_matrix.shape[1]))
    while error > 10 ** (-6) and iterations < 1000:
        gains = np.zeros(eco.mass_vector.shape[0])
        for k in range(eco.mass_vector.shape[0]):
            theta_prime, G = eco.hill_tot_g(k, x_temp)
            x_temp2 = np.copy(x_temp)
            x_temp2[k] = np.array(theta_prime).flatten()
            gains[k] = G
            best_matrices[k] = x_temp2
        print(gains, "GAINS")
        error = np.max(gains) #np.sum(np.sum(np.dot(np.abs(best_matrices[np.argmax(regrets)] - x_temp), eco.spectral.M)))
        if error>0:
            x_temp = np.copy(best_matrices[np.argmax(gains)])

        if verbose is True:
            print("Error Hillclimbing: ", error, iterations)
        iterations += 1
    if iterations >= 1000:
        x_temp = eco.strategy_matrix
        print("Hillclimbing also failed")

    return x_temp




