import numpy as np
import scipy.optimize as optm
import matlab.engine
import scipy as scp
import itertools as itertools
import casadi as ca
import scipy.stats as stats
import copy as copy
import pickle as pkl

import matplotlib.pyplot as plt

eng = matlab.engine.start_matlab()

#Add interpolation logic (Convert to Lagrange, evaluate Lagrange in bigger point set, optimize again)
#Add cheeger constant calculator
#Fix water resource renewal logic.
#Implement simulation class

class ecosystem_optimization:

    def __init__(self, mass_vector, layers, parameters, spectral, water, l2 = True, output_level = 0, verbose = False, movement_cost = 0.1):
        self.layers = layers
        self.mass_vector = mass_vector
        self.spectral = spectral
        self.strategy_matrix = np.vstack([np.repeat(1/(spectral.x[-1]), layers)]*mass_vector.shape[0]) # np.zeros((mass_vector.shape[0], layers))
        self.populations = parameters.mass_vector**(-0.75)
        self.parameters = parameters
        self.ones = np.repeat(1, layers)
        self.water = water
        self.l2 = l2
        self.output_level = output_level
        self.verbose = verbose
        self.movement_cost = movement_cost

    def one_actor_growth(self, i, x_temp = None, solve_mode = True):
        temp_pops = np.copy(self.populations)
        temp_pops[i] = 0

        if x_temp is None:
            x_temp = self.strategy_matrix

        predation_ext_food = np.zeros(self.populations.shape[0])
        predator_hunger = np.zeros((self.populations.shape[0], self.layers))
        for k in range(self.parameters.who_eats_who.shape[0]):
            interaction_term = self.parameters.who_eats_who[k] * temp_pops * self.parameters.clearance_rate[k]

            lin_g_others = np.dot((x_temp*self.parameters.layered_attack[:, k, :].T @ self.spectral.M @ x_temp[k]), interaction_term)
            foraging_term = self.water.res_counts * self.parameters.forager_or_not[k] \
                                    * self.parameters.clearance_rate[k] * self.parameters.layered_foraging[:, k] * \
                                    x_temp[k]

            predation_ext_food[k] = np.sum(np.dot(self.spectral.M, foraging_term))+lin_g_others
            predator_hunger[k] = self.parameters.clearance_rate[k] * np.dot(self.spectral.M, self.parameters.layered_attack[:, k, i] * x_temp[k])*self.parameters.who_eats_who[k,i]

        x = ca.SX.sym('x', self.layers)

        interaction_term = self.parameters.who_eats_who[i] * self.populations * self.parameters.clearance_rate[i]

        lin_growth = ca.Function('lin_growth', [x], [ca.dot(interaction_term, (x_temp*self.parameters.layered_attack[:, i, :].T) @ self.spectral.M @ x)])

        foraging_term_self = ca.Function('foraging_term_self', [x], [(self.water.res_counts * self.parameters.forager_or_not[i] \
                                * self.parameters.clearance_rate[i] * self.parameters.layered_foraging[:, i]).reshape((1,self.layers)) @ (self.spectral.M @ x)])

        actual_growth = self.parameters.efficiency * (lin_growth(x) + foraging_term_self(x)) \
                                / (1 + self.parameters.handling_times[i] * (lin_growth(x)+foraging_term_self(x))) - self.movement_cost*((self.strategy_matrix[i]-x).T @ self.spectral.M @ (self.strategy_matrix[i]-x))

#        x_i_NP = x_temp[i].reshape((self.layers,1))
#        print(ca.substitute(self.movement_cost*(self.strategy_matrix[i]-x).T @ self.spectral.M @ (self.strategy_matrix[i]-x), x, x_temp[0]))
        #print(predator_hunger.shape, x_i_NP.shape)
        #pred_loss_np = np.dot(((predator_hunger @ x_i_NP) * (1 + self.parameters.handling_times.reshape((self.populations.shape[0], 1)) * (self.populations[i] * predator_hunger @ x_i_NP) + lin_g_others)**(-1)).flatten(), self.populations)

        #print(pred_loss_np, "PREDLOSSNP", i)

        pred_loss = ca.dot((predator_hunger @ x / (1 + self.parameters.handling_times.reshape((self.populations.shape[0], 1))*(self.populations[i]*predator_hunger @ x) + lin_g_others)), self.populations) #ca.dot((x.T @ predator_hunger.T).T, self.populations)   This snippet is for the linear case, known good
        #print(pred_loss.size())
        if self.verbose is True:
            print("Size Class: ", i, "Predation loss", ca.substitute(pred_loss, x, self.strategy_matrix[i]), "Actual growth", ca.substitute(actual_growth, x, self.strategy_matrix[i]))

        delta_p = ca.Function('delta_p', [x], [(actual_growth - pred_loss)], ['x'], ['r'])

        if solve_mode is True:
            pop_change = -(actual_growth - pred_loss)
            if self.l2 is True:
                g = x.T @ self.spectral.M @ x - 1
                arg = {"lbg": 0, "ubg": 0}

            else:
                g = ca.dot(self.ones, self.spectral.M @ x) - 1
                arg = {"lbg": 0, "ubg": 0}


            #print(ca.substitute(g,x,self.strategy_matrix[i]), "CONSTRAINTS")
            x_min = [0]*self.layers
            x_max = [ca.inf]*self.layers
            arg["lbx"] = x_min
            arg["ubx"] = x_max
            s_opts = {'ipopt': {'print_level' : self.output_level}}
            prob = {'x': x, 'f': pop_change, 'g': g}
            solver = ca.nlpsol('solver', 'ipopt', prob, s_opts)
            sol = solver(x0=x_temp[i], **arg)
            print(ca.substitute(pop_change, x, np.repeat(1/(self.spectral.x[-1]), self.layers)), ca.substitute(pop_change, x, sol['x']))
            x_out = sol['x']

        else:
            x_out = delta_p(x_temp[i])
        return x_out


    def casadi_total_growth(self):
        x_temp = self.strategy_matrix
        x_tot = []
        total_loss = 0 #= []
        g = []
        lbg = []
        ubg = []
        lbx = []
        ubx = []
        for i in range(self.populations.shape[0]):
            temp_pops = np.copy(self.populations)
            temp_pops[i] = 0
            x_i = ca.SX.sym('x' + str(i), self.layers)
            x_tot.append(x_i)
            print(i, x_tot[i], self.populations.shape[0])
            temp_pops = np.copy(self.populations)
            temp_pops[i] = 0

            if x_temp is None:
                x_temp = self.strategy_matrix

            predation_ext_food = np.zeros(self.populations.shape[0])
            predator_hunger = np.zeros((self.populations.shape[0], self.layers))
            for k in range(self.parameters.who_eats_who.shape[0]):
                interaction_term = self.parameters.who_eats_who[k] * temp_pops * self.parameters.clearance_rate[k]

                lin_g_others = np.dot((x_temp*self.parameters.layered_attack[:, k, :].T @ self.spectral.M @ x_temp[k]), interaction_term)
                foraging_term = self.water.res_counts * self.parameters.forager_or_not[k] \
                                * self.parameters.clearance_rate[k] * self.parameters.layered_foraging[:, k] * \
                                x_temp[k]

                predation_ext_food[k] = np.sum(np.dot(self.spectral.M, foraging_term)) + lin_g_others
                predator_hunger[k] = self.parameters.clearance_rate[k] * np.dot(self.spectral.M,
                                                                                self.parameters.layered_attack[:, k,
                                                                                i] * x_temp[k]) * \
                                     self.parameters.who_eats_who[k, i]

            interaction_term = self.parameters.who_eats_who[i] * self.populations * self.parameters.clearance_rate[i]

            lin_growth = ca.Function('lin_growth', [x_i], [
                ca.dot(interaction_term, (x_temp * self.parameters.layered_attack[:, i, :].T) @ self.spectral.M @ x_i)])

            foraging_term_self = ca.Function('foraging_term_self', [x_i],
                                             [(self.water.res_counts * self.parameters.forager_or_not[i] \
                                               * self.parameters.clearance_rate[i] * self.parameters.layered_foraging[:,
                                                                                     i]).reshape((1, self.layers)) @ (
                                                          self.spectral.M @ x_i)])

            actual_growth = self.parameters.efficiency * (lin_growth(x_i) + foraging_term_self(x_i)) \
                            / (1 + self.parameters.handling_times[i] * (lin_growth(x_i) + foraging_term_self(x_i))) - self.movement_cost*((self.strategy_matrix[i]-x_i).T @ self.spectral.M @ (self.strategy_matrix[i]-x_i))

            pred_loss = ca.dot((predator_hunger @ x_i / (
                        1 + self.parameters.handling_times.reshape((self.populations.shape[0], 1)) * (
                            self.populations[i] * predator_hunger @ x_i) + lin_g_others)),
                               self.populations)  # ca.dot((x.T @ predator_hunger.T).T, self.populations)   This snippet is for the linear case, known good

            #ca.dot((x_i.T @ predator_hunger.T).T, self.populations)  #  #This snippet works, is made for the linear case
            x_i_NP = x_temp[i].reshape((self.layers, 1))
            grad_i = ca.gradient(actual_growth - pred_loss, x_i)
            print(grad_i.size(), ca.substitute(grad_i, x_i, x_temp[i]))
            total_loss = total_loss + ca.norm_2(grad_i)

            if self.l2 is True:
                g.append(x_i.T @ self.spectral.M @ x_i - 1)
                lbg.append(0)
                ubg.append(0)

            else:
                g.append(ca.dot(self.ones, self.spectral.M @ x_i) - 1)
                lbg.append(0)
                ubg.append(0)

            lbx.append(np.zeros(self.layers))
            ubx.append([ca.inf]*self.layers)

        x_tot = ca.vertcat(*x_tot)
        g = ca.vertcat(*g)
        lbx = np.concatenate(lbx)
        ubx = np.concatenate(ubx)
        lbg = np.zeros(self.populations.shape[0])
        ubg = np.zeros(self.populations.shape[0])

        print(ca.substitute(total_loss, x_tot, x_temp.flatten()), "Extremity")
        s_opts = {'ipopt': {'print_level' : 0}}
        prob = {'x': x_tot, 'f': total_loss, 'g': g}
        solver = ca.nlpsol('solver', 'ipopt', prob, s_opts)
        sol = solver(x0=x_temp.flatten(), lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
        x_out = sol['x']
        print(ca.substitute(total_loss, x_tot, x_out), "Extremity")
        return(x_out.reshape(x_temp.shape))

    def hill_tot_g(self):
        x_temp = self.strategy_matrix
        x_tot = []
        total_loss = 0 #= []
        g = []
        lbg = []
        ubg = []
        lbx = []
        ubx = []
        for i in range(self.populations.shape[0]):
            temp_pops = np.copy(self.populations)
            temp_pops[i] = 0
            x_tot.append(x_i)
            print(i, x_tot[i], self.populations.shape[0])
            temp_pops = np.copy(self.populations)
            temp_pops[i] = 0

            if x_temp is None:
                x_temp = self.strategy_matrix

            predation_ext_food = np.zeros(self.populations.shape[0])
            predator_hunger = np.zeros((self.populations.shape[0], self.layers))
            for k in range(self.parameters.who_eats_who.shape[0]):
                interaction_term = self.parameters.who_eats_who[k] * temp_pops * self.parameters.clearance_rate[k]

                lin_g_others = np.dot((x_temp*self.parameters.layered_attack[:, k, :].T @ self.spectral.M @ x_temp[k]), interaction_term)
                foraging_term = self.water.res_counts * self.parameters.forager_or_not[k] \
                                * self.parameters.clearance_rate[k] * self.parameters.layered_foraging[:, k] * \
                                x_temp[k]

                predation_ext_food[k] = np.sum(np.dot(self.spectral.M, foraging_term)) + lin_g_others
                predator_hunger[k] = self.parameters.clearance_rate[k] * np.dot(self.spectral.M,
                                                                                self.parameters.layered_attack[:, k,
                                                                                i] * x_temp[k]) * \
                                     self.parameters.who_eats_who[k, i]

            interaction_term = self.parameters.who_eats_who[i] * self.populations * self.parameters.clearance_rate[i]

            lin_growth = ca.Function('lin_growth', [x_i], [
                ca.dot(interaction_term, (x_temp * self.parameters.layered_attack[:, i, :].T) @ self.spectral.M @ x_i)])

            foraging_term_self = ca.Function('foraging_term_self', [x_i],
                                             [(self.water.res_counts * self.parameters.forager_or_not[i] \
                                               * self.parameters.clearance_rate[i] * self.parameters.layered_foraging[:,
                                                                                     i]).reshape((1, self.layers)) @ (
                                                          self.spectral.M @ x_i)])

            actual_growth = self.parameters.efficiency * (lin_growth(x_i) + foraging_term_self(x_i)) \
                            / (1 + self.parameters.handling_times[i] * (lin_growth(x_i) + foraging_term_self(x_i))) - self.movement_cost*((self.strategy_matrix[i]-x_i).T @ self.spectral.M @ (self.strategy_matrix[i]-x_i))

            pred_loss = ca.dot((predator_hunger @ x_i / (
                        1 + self.parameters.handling_times.reshape((self.populations.shape[0], 1)) * (
                            self.populations[i] * predator_hunger @ x_i) + lin_g_others)),
                               self.populations)  # ca.dot((x.T @ predator_hunger.T).T, self.populations)   This snippet is for the linear case, known good

            #ca.dot((x_i.T @ predator_hunger.T).T, self.populations)  #  #This snippet works, is made for the linear case
            x_i_NP = x_temp[i].reshape((self.layers, 1))
            grad_i = ca.gradient(actual_growth - pred_loss, x_i)
            print(grad_i.size(), ca.substitute(grad_i, x_i, x_temp[i]))
            total_loss = total_loss + ca.norm_2(grad_i)

            if self.l2 is True:
                g.append(x_i.T @ self.spectral.M @ x_i - 1)
                lbg.append(0)
                ubg.append(0)

            else:
                g.append(ca.dot(self.ones, self.spectral.M @ x_i) - 1)
                lbg.append(0)
                ubg.append(0)

            lbx.append(np.zeros(self.layers))
            ubx.append([ca.inf]*self.layers)

        x_tot = ca.vertcat(*x_tot)
        g = ca.vertcat(*g)
        lbx = np.concatenate(lbx)
        ubx = np.concatenate(ubx)
        lbg = np.zeros(self.populations.shape[0])
        ubg = np.zeros(self.populations.shape[0])

        print(ca.substitute(total_loss, x_tot, x_temp.flatten()), "Extremity")
        s_opts = {'ipopt': {'print_level' : 0}}
        prob = {'x': x_tot, 'f': total_loss, 'g': g}
        solver = ca.nlpsol('solver', 'ipopt', prob, s_opts)
        sol = solver(x0=x_temp.flatten(), lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
        x_out = sol['x']
        print(ca.substitute(total_loss, x_tot, x_out), "Extremity")
        return(x_out.reshape(x_temp.shape))




    def total_growth(self, x_temp = None):
        if x_temp is None:
            x_temp = self.strategy_matrix

        total_growth = np.zeros((self.populations.shape[0]))
        for i in range(self.populations.shape[0]):
            total_growth[i] = self.one_actor_growth(i, x_temp=x_temp, solve_mode=False)
        print(self.parameters.loss_term, total_growth)
        return (total_growth - self.parameters.loss_term)*self.populations

    def strategy_setter(self, strat_vec): #Keep in population state class
        s_m = np.reshape(strat_vec, self.strategy_matrix.shape)
        self.strategy_matrix = s_m

    def population_setter(self, new_pop): #Keep in population state class
        self.populations = new_pop #self.total_growth()*time_step

    def consumed_resources(self): #Move to simulator class

        consumed_resources = np.zeros(self.layers)
        for i in range(len(self.mass_vector)):
            strat_mat = self.strategy_matrix
            interaction_term = self.parameters.who_eats_who[i] * self.populations
            interaction_term = interaction_term * self.parameters.clearance_rate[i]
            layer_action = np.zeros((self.layers, self.mass_vector.shape[0]))
            for j in range(self.layers):
                layer_action[j] = self.parameters.layered_attack[j, i] * strat_mat[i, j] * interaction_term * strat_mat[:,j] \
                                  * self.parameters.handling_times[i]

            foraging_term = self.water.res_counts*self.parameters.forager_or_not[i] \
                            *self.parameters.clearance_rate[i] * self.parameters.layered_foraging[:, i] * strat_mat[i]
            consumed_resources += self.populations[i]*foraging_term /(1 + self.parameters.handling_times[i]*np.sum(np.dot(self.ones, np.dot(self.spectral.M, (np.sum(layer_action, axis = 1) + foraging_term)))))
        if self.verbose is True:
            print(np.sum(np.dot(self.spectral.M, consumed_resources)), "Consumed resources")
        return consumed_resources

class spectral_method:
    def __init__(self, depth, layers):

        self.n = layers

        JacobiGL = lambda x, y, z: eng.JacobiGL(float(x), float(y), float(z), nargout=1)

        x = np.array(list(itertools.chain(JacobiGL(0, 0, layers))))
        self.x = np.reshape(x, x.shape[0])

        D_calc = lambda n: np.matmul(np.transpose(self.vandermonde_dx()),
                                                   np.linalg.inv(np.transpose(self.vandermonde_calculator())))*(depth/2)

        self.D = D_calc(layers)
        M_calc = lambda n: np.dot(np.linalg.inv(self.vandermonde_calculator()),
                                       np.linalg.inv(np.transpose(self.vandermonde_calculator())))*(depth/2)

        self.M = M_calc(layers)
        self.x = ((self.x+1)/2) * depth

        self.vandermonde = self.vandermonde_calculator().T
        self.vandermonde_inv = np.linalg.inv(self.vandermonde)
        #self.vandermonde = depth*self.vandermonde

    def JacobiP(self, x, alpha, beta, n):
        P_n = np.zeros((n, x.shape[0]))
        P_n[0] = 1
        P_n[1] = 0.5 * (alpha - beta + (alpha + beta + 2) * x)
        for i in range(1, n - 1):
            an1n = 2 * (i + alpha) * (i + beta) / ((2 * i + alpha + beta + 1) * (2 * i + alpha + beta))
            ann = (alpha ** 2 - beta ** 2) / ((2 * i + alpha + beta + 2) * (2 * i + alpha + beta))
            anp1n = 2 * (i + 1) * (i + alpha + beta + 1) / ((2 * i + alpha + beta + 2) * (2 * i + alpha + beta + 1))

            P_n[i + 1] = ((ann + x) * P_n[i] - an1n * P_n[i - 1]) / anp1n

        return P_n


    def JacobiP_n(self, x, alpha, beta, n):
        P_n = self.JacobiP(x, alpha, beta, n)
        if alpha == 1 and beta == 1:
            gamma = lambda alpha, beta, m: 2 ** (3) * (m + 1) / (m + 2) * 1 / ((2 * m + alpha + beta + 1))
        elif alpha == 0 and beta == 0:
            gamma = lambda alpha, beta, m: 2 / ((2 * m + alpha + beta + 1))
        elif alpha == -1 / 2 and beta == - 1 / 2:
            gamma = lambda alpha, beta, m: 2 * scp.math.factorial(m) / ((2 * m + alpha + beta + 1) * scp.gamma(m + 1 / 2))

        for i in range(n):
            d = np.sqrt(gamma(alpha, beta, i))
            P_n[i] = P_n[i] / d

        return P_n


    def GradJacobi_n(self, x, alpha, beta, n):
        P_diff = np.zeros((n, x.shape[0]))
        JacobiPnorma = self.JacobiP_n(x, alpha + 1, beta + 1, n)
        for i in range(1, n):
            P_diff[i] = JacobiPnorma[i - 1] * np.sqrt(i * (i + alpha + beta + 1))
        return P_diff


    def vandermonde_calculator(self):
        n = self.n
        x = self.x
        return (self.JacobiP_n(x, 0, 0, n + 1))


    def vandermonde_dx(self):
        n = self.n
        x = self.x
        return (self.GradJacobi_n(x, 0, 0, n + 1))

    def expander(self, old_spec = None, small_vec = None):

        new_vm = self.JacobiP(self.x, 0, 0, small_vec.shape[0])
        old_vm = np.linalg.inv(old_spec.vandermonde_calculator())

        #print(small_vec)
        return(np.dot(new_vm.T, np.dot(old_vm.T, small_vec)))

    def projector(self, old_spec, big_vec):
        pass

    def interpolater(old_vec, old_size, new_size, size_classes, old_spec, new_spec):
        new_vec = np.zeros(new_size * size_classes)
        for k in range(size_classes):
            new_strat = new_spec.expander(old_spec=old_spec, small_vec=old_vec[old_size * k:old_size * (k + 1)])
            new_strat = np.abs(new_strat)  # '#[new_strat < 0] = 0
            new_vec[new_size * k:new_size * (k + 1)] = new_strat / np.dot(np.repeat(1, new_size),
                                                                          np.dot(new_spec.M, new_strat))

        return new_vec


class ecosystem_parameters:
    def __init__(self, mass_vector, spectral):
        self.forage_mass = 0.05

        self.mass_vector = mass_vector
        self.spectral = spectral
        self.attack_matrix = self.attack_matrix_setter()
        self.handling_times = self.handling_times_setter()
        self.who_eats_who = self.who_eats_who_setter()
        self.clearance_rate = self.clearance_rate_setter() #330/12 * mass_vector**(3/4)
        self.layered_attack = self.layer_creator(self.attack_matrix, lam = 0.8)
        self.efficiency = 0.7 #Very good number.
        self.loss_term = self.loss_rate_setter() #mass_vector**(0.75)*0.01
        self.forager_or_not = self.forager_or_not_setter()
        self.foraging_attack_prob = self.foraging_attack_setter()
        self.layered_foraging = self.layer_creator(self.foraging_attack_prob, lam = 2)


    def forager_or_not_setter(self):
        #forage_mass =1/408 #Should probably be in the ecosystem parameters explicitly
        fo_or_not = (np.copy(self.mass_vector))/self.forage_mass
        fo_or_not[fo_or_not > 1600] = 0
        fo_or_not[fo_or_not != 0] = 1

        return fo_or_not

    def foraging_attack_setter(self):
        #forage_mass = 0.05
        sigma = 1.3
        beta = 408

        foraging_attack = np.exp(-(np.log(self.mass_vector/(beta*self.forage_mass))) ** 2 / (2 * sigma ** 2))

        return foraging_attack

    def attack_matrix_setter(self):
        sigma = 1.3
        beta = 408

        att_m = self.mass_vector*1/(beta*self.mass_vector.reshape((self.mass_vector.shape[0],1)))
        att_m = att_m.T
        att_m = np.exp(-(np.log(att_m))**2/(2*sigma**2))

        return att_m


    def who_eats_who_setter(self):
        who_eats_who = self.mass_vector * 1/self.mass_vector.reshape((self.mass_vector.shape[0], 1))
        who_eats_who[who_eats_who < 10] = 0
        who_eats_who[who_eats_who != 0] = 1

        who_eats_who = who_eats_who.T

        return who_eats_who

    def handling_times_setter(self):
        return 1/(22.3*self.mass_vector**(0.75))

    def loss_rate_setter(self):
        return 0.1*self.mass_vector**(0.75) #Used to be 1.2, but this was annoying

    def layer_creator(self, obj, lam = 0.8):
        weights = 2/(1+np.exp(lam*self.spectral.x)) #Replace with actual function
        layers = np.zeros((self.spectral.x.shape[0], *obj.shape))
       # print(layers.shape, obj.shape, self.spectral.x.shape[0])

        for i in range(self.spectral.x.shape[0]):
            layers[i] = weights[i] * obj
        return layers

    def clearance_rate_setter(self):
        return 330*self.mass_vector**(0.75) #This should probably be more sophisticated... But a good first approximation

    def layered_attack_setter(self, layered_attack):
        self.layered_attack = layered_attack


class water_column:
    def __init__(self, spectral, res_vec, advection = 1, diffusion = 0.5, resource_max = None, replacement = 1.2, layers = 2):
        self.adv = advection
        self.diff = diffusion
        if resource_max is None:
            self.resource_max = stats.norm.pdf(spectral.x, loc=2)

        self.resource_max = resource_max
        self.lam = replacement
        self.spectral = spectral
        self.res_counts = res_vec
        self.layers = layers
        self.res_top = np.zeros(layers)
        if diffusion !=0 or advection != 0:
            self.diff_op = (-self.adv*self.spectral.D+self.diff*np.linalg.matrix_power(self.spectral.D,2))+np.identity(self.layers) #missing time_step
            #diff_op[0] = self.spectral.D[0]
            self.diff_op = np.dot(spectral.M, self.diff_op)

            self.diff_op[-1] = self.spectral.D[-1]
            self.diff_op[0] = 0
            self.diff_op[0,0] = 1


    def resource_setter(self, new_res):
        self.res_counts = new_res

    def update_resources(self, consumed_resources = 0, time_step = 0.001):
        ##Chemostat step
        self.res_counts += (self.lam*(self.resource_max - self.res_counts) - consumed_resources)*time_step
        ##Advection diffusion
        if self.diff != 0 or self.adv != 0: #THIS IS HORRIBLY BROKEN ATM!!!
            sol_vec = np.copy(self.res_counts)
            #sol_vec[0] = 0
            sol_vec = np.dot(self.spectral.M, sol_vec) #Add extra and interpolation steps here.
            sol_vec[-1] = 0
            #print(self.diff, self.adv, self.time_step)
            #diff_op[-1,-1] = 1
            #print(self.res_counts, np.linalg.solve(diff_op, sol_vec))
            self.res_counts = np.abs(np.linalg.solve(self.diff_op, sol_vec))




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
    def graph_builder(self):  # Move into ecosystem class
        strat_mat = self.animals.strategy_matrix
        #outflows = np.zeros((self.animals.mass_vector.shape[0], self.animals.mass_vector.shape[0]))
        for i in range(self.animals.mass_vector.shape[0]):
            for k in range(self.animals.parameters.who_eats_who.shape[0]):
                interaction_term = self.animals.parameters.who_eats_who[k] * self.animals.populations * self.animals.parameters.handling_times[k] * \
                                   self.animals.parameters.clearance_rate[k]
                layer_action = strat_mat[i, :].reshape((self.animals.layers, 1)) * self.animals.parameters.layered_attack[:, i, :] * (
                            interaction_term.reshape((self.animals.populations.shape[0], 1)) * strat_mat).T

                foraging_term = self.animals.water.res_counts * self.animals.parameters.forager_or_not[k] * \
                                self.animals.parameters.handling_times[k] \
                                * self.animals.parameters.clearance_rate[k] * self.animals.parameters.layered_foraging[:, k]
                # print(eco.parameters.layered_attack[:,k,i], k, i, strat_mat[i])
                self.graph[1+i, k] = np.dot(strat_mat[k], np.dot(self.animals.spectral.M, strat_mat[i] * self.animals.populations[
                    k] * self.animals.parameters.layered_attack[:, k, i] * self.animals.parameters.clearance_rate[k] * self.animals.populations[
                                                                 i])) / \
                                 (1 + np.sum(np.dot(self.animals.ones, np.dot(self.animals.spectral.M,
                                                                     np.sum(layer_action, axis=1) + foraging_term))))
                self.graph[0,k] =  np.sum(np.dot(self.animals.spectral.M, foraging_term)/(1 + np.sum(np.dot(self.animals.ones, np.dot(self.animals.spectral.M,
                                                                     np.sum(layer_action, axis=1) + foraging_term)))))



    def sequential_nash(self, verbose=False):
        x_temp = np.copy(self.animals.strategy_matrix)  # np.zeros(size_classes*layers)
        x_temp2 = np.copy(self.animals.strategy_matrix)
        error = 1
        iterations = 0
        while error > 10 ** (-6) and iterations < 40:
            for k in range(self.animals.mass_vector.shape[0]):
                result = self.animals.one_actor_growth(k, x_temp)
                x_temp2[k] = np.array(result).flatten()
                if verbose is True:
                    print(np.dot(self.animals.ones, np.dot(self.animals.spectral.M, (result) ** 2)), k, "2 Norm of strategy")

            error = np.max(np.abs(x_temp - x_temp2))

            if verbose is True:
                print("Error: ", error)
            x_temp = np.copy(x_temp2)
            iterations += 1
        if iterations >= 40:
            x_temp = self.animals.casadi_total_growth()
        # print(x_temp,eco.water.res_counts)
        return x_temp

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


def sequential_nash(eco, verbose = False, circle_mode = False, l2 = False):
    x_temp = np.copy(eco.strategy_matrix)  # np.zeros(size_classes*layers)
    x_temp2 = np.copy(eco.strategy_matrix)
    error = 1
    iterations = 0
    while error > 10 ** (-6) and iterations < 20:
        if circle_mode is False:
            for k in range(eco.mass_vector.shape[0]):
                result = eco.one_actor_growth(k, x_temp)
                x_temp2[k] = np.array(result).flatten()
                if verbose is True:
                    print(np.dot(eco.ones, np.dot(eco.spectral.M, (result) ** 2)), k, "2 Norm of strategy")
        if circle_mode is True:
            for k in range(eco.mass_vector.shape[0]):
                result = eco.one_actor_growth(k, x_temp)
                x_temp[k] = np.array(result).flatten()
                if verbose is True:
                    print(np.dot(eco.ones, np.dot(eco.spectral.M, (result) ** 2)), k, "2 Norm of strategy")
        if l2 is False:
            error = np.max(np.dot(np.abs(x_temp2 - x_temp), eco.spectral.M))
        else:
            print(np.dot((x_temp2 - x_temp), eco.spectral.M).shape)
            error = np.max(np.dot((x_temp2 - x_temp), eco.spectral.M) @ (x_temp2 - x_temp).T)

        if verbose is True:
            print("Error: ", error)
        x_temp = np.copy(x_temp2)
        iterations += 1
    if iterations >= 20:
        #x_temp = np.array(eco.casadi_total_growth())
        #if verbose is True:
        #    for k in range(eco.mass_vector.shape[0]):
        #        x_temp2 = np.copy(x_temp)
        #        x_temp2[k] = np.array(eco.one_actor_growth(k, x_temp).full()).flatten()
        #        print(eco.one_actor_growth(k, x_temp2, solve_mode = False) - eco.one_actor_growth(k, x_temp, solve_mode=False), 'Wrong fitness', k)
        x_temp = eco.strategy_matrix
    return x_temp

def hillclimb_nash(eco, verbose = False, l2 = False):
    x_temp = np.copy(eco.strategy_matrix)  # np.zeros(size_classes*layers)
    error = 1
    iterations = 0
    best_matrices = np.zeros((eco.mass_vector.shape[0], eco.strategy_matrix.shape[0], eco.strategy_matrix.shape[1]))
    while error > 10 ** (-6) and iterations < 2:
        regrets = np.zeros(eco.mass_vector.shape[0])
        for k in range(eco.mass_vector.shape[0]):
            result = eco.one_actor_growth(k, x_temp)
            x_temp2 = np.copy(eco.strategy_matrix)
            x_temp2[k] = np.array(result).flatten()

            best_matrices[k] = x_temp2
        for j in range(eco.mass_vector.shape[0]):
            for i in range(eco.mass_vector.shape[0]):
                regrets[j] += eco.one_actor_growth(i, eco.strategy_matrix, solve_mode=False) - eco.one_actor_growth(i, best_matrices[j], solve_mode=False)

        print(regrets[0])
        if l2 is False:
            error = np.max(np.abs(regrets)) #np.sum(np.sum(np.dot(np.abs(best_matrices[np.argmax(regrets)] - x_temp), eco.spectral.M)))
        else:
            error = np.sum(np.dot((best_matrices[np.argmax(regrets)] - x_temp), eco.spectral.M) @ (best_matrices[np.argmax(regrets)] - x_temp).T)

        x_temp = best_matrices[np.argmax(regrets)]


        if verbose is True:
            print("Error: ", error, iterations)
        iterations += 1
    if iterations >= 100:
        x_temp = eco.strategy_matrix


    return x_temp

