temp_mat = ca.zeros((self.populations.shape[0], self.populations.shape[0]))
temp_mat[l, l] = 1
x_temp_0 = ca.copy(self.strategy_matrix)
x_temp_0[l] = 0
x_tot = ca.SX.sym('x_tot', self.populations.shape[0], self.layers)
var = temp_mat @ x_tot
x_temp = var + x_temp_0
x_i = x_temp[i,:]
# x_temp = ca.sub
predation_ext_food = ca.SX.sym('predation_ext_food', self.populations.shape[0], self.layers)
predator_hunger = ca.SX.sym('predator_hunger', self.populations.shape[0], self.layers)
total_loss = 0
for i in range(self.populations.shape[0]):
    temp_pops = ca.copy(self.populations)
    temp_pops[i] = 0
    temp_pops = ca.copy(self.populations)
    temp_pops[i] = 0

    predation_ext_food = []
    predator_hunger = []
    # print(x_temp.size(), x_temp_0.shape, (x_temp*self.parameters.layered_attack[:, 1, :].T @ self.spectral.M @ x_temp[1]).size())
    for k in range(self.parameters.who_eats_who.shape[0]):
        interaction_term = self.parameters.who_eats_who[k] * temp_pops * self.parameters.clearance_rate[k]

        lin_g_others = interaction_term.reshape((1, -1)) @ (
                    x_temp * self.parameters.layered_attack[:, k, :].T @ self.spectral.M @ x_temp[k])
        foraging_term = self.water.res_counts * self.parameters.forager_or_not[k] \
                        * self.parameters.clearance_rate[k] * self.parameters.layered_foraging[:, k] * \
                        x_temp[k]
        print(ca.dot((self.spectral.M @ foraging_term), self.ones))
        predation_ext_food.append(ca.dot((self.spectral.M @ foraging_term), self.ones) + lin_g_others)
        predator_hunger.append(self.parameters.clearance_rate[k] * (
                    self.spectral.M @ self.parameters.layered_attack[:, k, i] * x_temp[k]) *
                               self.parameters.who_eats_who[k, i])

    predation_ext_food = ca.vertcat(*predation_ext_food)
    predator_hunger = ca.vertcat(*predator_hunger)

    interaction_term = self.parameters.who_eats_who[i] * self.populations * self.parameters.clearance_rate[i]

    lin_growth = interaction_term.reshape((-1, 1)) @ x_temp * (
                self.parameters.layered_attack[:, i, :].T @ self.spectral.M @ x_temp[
            i])  # ca.Function('lin_growth', [x_tot], [ca.dot(interaction_term, (x_temp * self.parameters.layered_attack[:, i, :].T) @ self.spectral.M @ x_temp[i])])

    foraging_term_self = (self.water.res_counts * self.parameters.forager_or_not[i] \
                          * self.parameters.clearance_rate[i] * self.parameters.layered_foraging[:,
                                                                i]).reshape((1, self.layers)) @ (
                                 self.spectral.M @ x_temp[i])

    actual_growth = self.parameters.efficiency * (lin_growth + foraging_term_self) \
                    / (1 + self.parameters.handling_times[i] * (
                lin_growth + foraging_term_self)) - self.movement_cost * (
                                (self.strategy_matrix[i] - x_temp[i]).T @ self.spectral.M @ (
                                    self.strategy_matrix[i] - x_temp[i]))

    pred_loss = ca.dot((predator_hunger @ x_temp[i] / (
            1 + self.parameters.handling_times.reshape((self.populations.shape[0], 1)) * (
            self.populations[i] * predator_hunger @ x_temp[i]) + predation_ext_food)),
                       self.populations)

    total_loss = total_loss - self.one_actor_growth(i, solve_mode=False) + pred_loss - actual_growth  # Note the signs



