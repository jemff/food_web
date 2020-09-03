import numpy as np
import scipy.optimize as optm
import matlab.engine
import scipy as scp
import itertools as itertools
import casadi as ca

import matplotlib.pyplot as plt

eng = matlab.engine.start_matlab()

#Add interpolation logic (Convert to Lagrange, evaluate Lagrange in bigger point set, optimize again)
#Add cheeger constant calculator
#Fix water resource renewal logic.
#Implement simulation class

def jacobian_calculator(f, x, h):
    if max(np.shape(np.array([f(x)]))) <= 1:
        jac = np.zeros(x.shape[0])
        x_m = np.copy(x)
        x_p = np.copy(x)
        for i in range(len((x))):
            x_m[i] -= h
            x_p[i] += h
            jac[i] = (f(x_p) - f(x_m))/(2*h)

    else:
        jac = np.zeros((x.shape[0],x.shape[0]))
        x_m = np.copy(x)
        x_p = np.copy(x)
        for i in range(len((x))):
            x_m[i] -= h
            x_p[i] += h
            jac[:, i] = (f(x_p) - f(x_m))/(2*h)
            x_m = np.copy(x)
            x_p = np.copy(x)

    return jac


class ecosystem_optimization:

    def __init__(self, mass_vector, layers, parameters, spectral, water, loss = 'l2'):
        self.layers = layers
        self.mass_vector = mass_vector
        self.spectral = spectral
        self.strategy_matrix = np.vstack([np.repeat(1/(spectral.x[-1]), layers)]*mass_vector.shape[0]) # np.zeros((mass_vector.shape[0], layers))
        self.populations = mass_vector**(-0.75) #Now this is a flat structure
        self.parameters = parameters
        self.ones = np.repeat(1, layers)
        self.water = water
        self.loss = loss

    def one_actor_growth(self, i, x_temp = None, solve_mode = True):
        temp_pops = np.copy(self.populations)
        temp_pops[i] = 0

        if x_temp is None:
            x_temp = self.strategy_matrix

        predation_ext_food = np.zeros(self.populations.shape[0])
        predator_hunger = np.zeros((self.populations.shape[0], self.layers))
        for k in range(self.parameters.who_eats_who.shape[0]):
            interaction_term = self.parameters.who_eats_who[k] * temp_pops * self.parameters.clearance_rate[k]

            layer_action = x_temp[k, :].reshape((self.layers, 1)) * self.parameters.layered_attack[:, k, :]\
                           * (interaction_term.reshape((self.populations.shape[0], 1)) * x_temp).T

            foraging_term = self.water.res_counts * self.parameters.forager_or_not[k] \
                                    * self.parameters.clearance_rate[k] * self.parameters.layered_foraging[:, k] * \
                                    x_temp[k]

            predation_ext_food[k] = \
                        np.sum(np.dot(self.ones, np.dot(self.spectral.M, np.sum(layer_action, axis=1) + foraging_term)))
            predator_hunger[k] = self.parameters.clearance_rate[k] * np.dot(self.spectral.M,
                                                                                    x_temp[k])


        x = ca.SX.sym('x', self.layers)

        interaction_term = self.parameters.who_eats_who[i] * self.populations * self.parameters.clearance_rate[i]

        lin_growth = ca.Function('lin_growth', [x], [self.ones.reshape((1, self.layers)) @ (self.parameters.layered_attack[:, i, :] @
                                       (interaction_term.reshape((self.populations.shape[0], 1)) *
                                        x_temp) @ x)])



        foraging_term = ca.Function('foraging_term', [x], [(self.water.res_counts * self.parameters.forager_or_not[i] \
                                * self.parameters.clearance_rate[i] * self.parameters.layered_foraging[:, i]).reshape((1,self.layers)) @ (self.spectral.M @ x)])


        actual_growth = self.parameters.efficiency * (lin_growth(x) + foraging_term(x)) \
                                / (1 + self.parameters.handling_times[i] * lin_growth(x)+foraging_term(x))

#        print("I get beyond this point", predation_ext_food.sum())
        predation_ext = (predation_ext_food*self.parameters.handling_times).sum()

#        print(predation_ext)

        const_thing = (temp_pops @ predator_hunger).reshape((1, self.layers))
        const_thing_2 = (self.parameters.handling_times @ predator_hunger).reshape((1, self.layers))*self.populations[i]


#        print(const_thing.shape, const_thing_2.shape, x, const_thing @ x)
        if np.sum(self.parameters.who_eats_who[:, i]) != 0:
            pred_loss = (const_thing @ x) / (1 + (const_thing_2 @ x + predation_ext))

        else:
            pred_loss = 0.1 * self.parameters.loss_term[i] * (ca.transpose(x) @ self.spectral.M @ self.parameters.layer_creator(np.array([1])))
           # print("Penalty!!", i)

        delta_p = ca.Function('delta_p', [x], [-(actual_growth - pred_loss)], ['x'], ['r'])

        if solve_mode is True:
            pop_change = -(actual_growth - pred_loss)

            g = x.T @ self.spectral.M @ x - 1
            x_min = [0]*self.layers
            x_max = [ca.inf]*self.layers
            arg = {"lbg": 0, "ubg": 0}
            arg["lbx"] = x_min
            arg["ubx"] = x_max
            s_opts = {'ipopt': {'print_level' : 0}}
            prob = {'x': x, 'f': pop_change, 'g': g}
            solver = ca.nlpsol('solver', 'ipopt', prob, s_opts)
    #        print(solver)
            sol = solver(x0=x_temp[i], **arg)
    #        print(sol)
            x_out = sol['x']
            #plt.plot(self.spectral.x, x_out)


#        if total is True and solve_mode is True:
#            total_x = ca.SX.sym('x', (self.layers, self.populations.shape[0])) #REMARK THE TRANSPOSE!!!!!!!!!
#            ca.jacobian(delta_p)
#            total_growth = delta_p.map(self.populations.shape[0])
            #total_growth.

        else:
            x_out = delta_p(self.strategy_matrix[i]) #ca.substitute(pop_change, x, self.strategy_matrix[i]) #Change to function type

        return x_out


    def casadi_total_growth(self):
        #x = ca.SX.sym('x', (self.layers, self.populations.shape[0]))  # REMARK THE TRANSPOSE!!!!!!!!!
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
            total_gradient = []

            predation_ext_food = np.zeros(self.populations.shape[0])
            predator_hunger = np.zeros((self.populations.shape[0], self.layers))
            for k in range(self.parameters.who_eats_who.shape[0]):
                interaction_term = self.parameters.who_eats_who[k] * temp_pops * self.parameters.clearance_rate[k]

                layer_action = x_temp[k, :].reshape((self.layers, 1)) * self.parameters.layered_attack[:, k, :] \
                               * (interaction_term.reshape((self.populations.shape[0], 1)) * x_temp).T

                foraging_term = self.water.res_counts * self.parameters.forager_or_not[k] \
                                * self.parameters.clearance_rate[k] * self.parameters.layered_foraging[:, k] * \
                                x_temp[k]

                predation_ext_food[k] = \
                    np.sum(np.dot(self.ones, np.dot(self.spectral.M, np.sum(layer_action, axis=1) + foraging_term)))
                predator_hunger[k] = self.parameters.clearance_rate[k] * np.dot(self.spectral.M,
                                                                                x_temp[k])

            interaction_term = self.parameters.who_eats_who[i] * self.populations * self.parameters.clearance_rate[
                i]

            lin_growth = ca.Function('lin_growth', [x_i],
                                     [self.ones.reshape((1, self.layers)) @ (
                                                 self.parameters.layered_attack[:, i, :] @
                                                 (interaction_term.reshape(
                                                     (self.populations.shape[0], 1)) *
                                                  x_temp) @ x_i)])

            foraging_term = ca.Function('foraging_term', [x_i],
                                        [(self.water.res_counts * self.parameters.forager_or_not[i] \
                                          * self.parameters.clearance_rate[i] * self.parameters.layered_foraging[:,
                                                                                i]).reshape((1, self.layers)) @ (
                                                 self.spectral.M @ x_i)])

            actual_growth = self.parameters.efficiency * (lin_growth(x_i) + foraging_term(x_i)) \
                            / (1 + self.parameters.handling_times[i] * lin_growth(x_i) + foraging_term(x_i))

            #        print("I get beyond this point", predation_ext_food.sum())
            predation_ext = (predation_ext_food * self.parameters.handling_times).sum()

            #        print(predation_ext)

            const_thing = (temp_pops @ predator_hunger).reshape((1, self.layers))
            const_thing_2 = (self.parameters.handling_times @ predator_hunger).reshape((1, self.layers)) * \
                            self.populations[i]

            #        print(const_thing.shape, const_thing_2.shape, x, const_thing @ x)
            if np.sum(self.parameters.who_eats_who[:, i]) != 0:
                pred_loss = (const_thing @ x_i) / (1 + (const_thing_2 @ x_i + predation_ext))

            else:
                pred_loss = 0.1 * self.parameters.loss_term[i] * (
                        ca.transpose(x_i) @ self.spectral.M @ np.repeat(1, self.layers))
            grad_i = ca.gradient(actual_growth - pred_loss, x_i)
            total_loss = total_loss + ca.norm_2(grad_i)
            g.append(x_i.T @ self.spectral.M @ x_i - 1)
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


        s_opts = {'ipopt': {'print_level' : 0}}
        prob = {'x': x_tot, 'f': total_loss, 'g': g}
        solver = ca.nlpsol('solver', 'ipopt', prob, s_opts)
        sol = solver(x0=x_temp.flatten(), lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
        x_out = sol['x']

        return(x_out.reshape(x_temp.shape))



    def total_growth(self):
        total_growth = np.zeros((self.populations.shape[0]))
        for i in range(self.populations.shape[0]):
            total_growth[i] = self.one_actor_growth(i, solve_mode=False)


        return (total_growth - self.parameters.loss_term)*self.populations



    def strategy_replacer(self, x, i, strategies): #Remove, deprecated
        strat = np.copy(strategies)
        strat[i*self.layers:(i+1)*self.layers] = x

        #print(i)
        return strat

    def one_actor_growth_num_derr(self, strategies, i, fineness = 0.000001): #Remove, deprecated
        return jacobian_calculator(lambda x: self.one_actor_growth(self.strategy_replacer(x, i, strategies), i), strategies[i:i+self.layers], fineness)

    def one_actor_hessian(self, strategies, i, fineness = 0.000001): #Remove, deprecated
        return jacobian_calculator(lambda x: self.one_actor_growth_num_derr(self.strategy_replacer(x, i, strategies), i),
                                   strategies[i:i + self.layers], fineness)

    def loss_function(self, strategies): #Remove, deprecated

        if self.loss == 'l2':
            total_loss = np.zeros([(self.layers + 1)*self.mass_vector.shape[0]])
           # print(total_loss.shape)
            v = 0
            for i in range(self.mass_vector.shape[0]):

                total_loss[v:v+self.layers] = self.one_actor_growth_num_derr(strategies, i) #[0]
                #print(strategies[i*self.layers:(i+1)*self.layers].shape, self.layers, i, i*layers, (i+1)*layers, i+1)
                total_loss[v+1] = np.dot(self.ones,np.dot(self.spectral.M, strategies[i*self.layers:(i+1)*self.layers]))-1
                v += self.layers+1

        else:
            total_loss = np.zeros([(self.layers)*self.mass_vector.shape[0]])
            v = 0
            for i in range(self.mass_vector.shape[0]):

                total_loss[v:v+self.layers] = self.one_actor_growth_num_derr(strategies, i)[0]
                #print(strategies[i*self.layers:(i+1)*self.layers].shape, self.layers, i, i*layers, (i+1)*layers, i+1)
                v += self.layers

        return total_loss

    def strategy_setter(self, strat_vec): #Keep in population state class
        s_m = np.reshape(strat_vec, self.strategy_matrix.shape)
        self.strategy_matrix = s_m

    def population_setter(self, new_pop): #Keep in population state class
        self.populations = new_pop #self.total_growth()*time_step

    def consume_resources(self, time_step): #Move to simulator class

        consumed_resources = np.zeros(self.layers)
        for i in range(len(self.mass_vector)):
            strat_mat = self.strategy_matrix
            interaction_term = self.parameters.who_eats_who[i] * self.populations
            interaction_term = interaction_term * self.parameters.clearance_rate[i]
            layer_action = np.zeros((self.layers, self.mass_vector.shape[0]))
            for j in range(self.layers):
                layer_action[j] = self.parameters.layered_attack[j, i] * strat_mat[i, j] * interaction_term * strat_mat[:,j] \
                                  * self.parameters.handling_times[i]

            foraging_term = self.water.res_counts*self.parameters.forager_or_not[i] * self.parameters.handling_times[i]\
                            *self.parameters.clearance_rate[i] * self.parameters.layered_foraging[:, i] * strat_mat[i]
            consumed_resources += foraging_term /(1 + np.sum(np.dot(self.ones, np.dot(self.spectral.M, (np.sum(layer_action, axis = 1) + foraging_term)))))

        self.water.resource_setter(self.water.res_counts - time_step * consumed_resources)

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
        fo_or_not[fo_or_not > 400] = 0
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

    def graph_builder(self):
        strat_mat = self.strategy_matrix
        outflows = np.zeros((self.mass_vector.shape[0], self.mass_vector.shape[0]))
        for i in range(self.mass_vector.shape[0]):
            for k in range(self.parameters.who_eats_who.shape[0]):
                interaction_term = self.parameters.who_eats_who[k] * self.populations * self.parameters.handling_times[
                    k] * self.parameters.clearance_rate[k]
                layer_action = strat_mat[i, :].reshape((self.layers, 1)) * self.parameters.layered_attack[:, i, :] * (
                            interaction_term.reshape((self.populations.shape[0], 1)) * strat_mat).T

                foraging_term = self.water.res_counts * self.parameters.forager_or_not[k] * \
                                self.parameters.handling_times[k] \
                                * self.parameters.clearance_rate[k] * self.parameters.layered_foraging[:, k]
                # print(self.parameters.layered_attack[:,k,i], k, i, strat_mat[i])
                outflows[i, k] = np.dot(strat_mat[k], np.dot(self.spectral.M, strat_mat[i] * self.populations[
                    k] * self.parameters.layered_attack[:, k, i] * self.parameters.clearance_rate[k] * self.populations[
                                                                 i])) / \
                                 (1 + np.sum(np.dot(self.ones, np.dot(self.spectral.M,
                                                                      np.sum(layer_action, axis=1) + foraging_term))))

        return outflows

class water_column:
    def __init__(self, spectral, res_vec, advection = 1, diffusion = 0.5, resource_max = 30, replacement = 1.2, time_step = 0.001, layers = 2):
        self.adv = advection
        self.diff = diffusion
        self.resource = resource_max
        self.lam = replacement
        self.time_step = time_step
        self.spectral = spectral
        self.res_counts = res_vec
        self.layers = layers
        self.res_top = np.zeros(layers)
        if diffusion !=0 or advection != 0:
            self.diff_op = (-self.adv*self.spectral.D+self.diff*np.linalg.matrix_power(self.spectral.D,2))*self.time_step+np.identity(self.layers)
            #diff_op[0] = self.spectral.D[0]
            self.diff_op = np.dot(spectral.M, self.diff_op)

            self.diff_op[-1] = self.spectral.D[-1]
            self.diff_op[0] = 0
            self.diff_op[0,0] = 1


    def resource_setter(self, new_res):
        self.res_counts = new_res

    def update_resources(self):
        ##Chemostat step
        #ones = np.repeat(1, self.layers)
        #self.res_top[0:int(self.layers/10)+1] = self.res_counts[0:int(self.layers/10)+1]
        total_top_mass = self.res_counts[0] #np.dot(ones, np.dot(self.spectral.M, self.res_top))
        #print(total_top_mass, self.res_counts)
        #print(total_top_mass, self.res_counts[0])
        self.res_counts[0] += self.lam*(self.resource - total_top_mass)*self.time_step
        ##Advection diffusion
        if self.diff != 0 or self.adv != 0:
            sol_vec = np.copy(self.res_counts)
            #sol_vec[0] = 0
            sol_vec = np.dot(self.spectral.M, sol_vec) #Add extra and interpolation steps here.
            sol_vec[-1] = 0
            #print(self.diff, self.adv, self.time_step)
            #diff_op[-1,-1] = 1
            #print(self.res_counts, np.linalg.solve(diff_op, sol_vec))
            self.res_counts = np.abs(np.linalg.solve(self.diff_op, sol_vec))




class simulator: #Expand this class
    def __init__(self, step_size, time, eco, simulate_population = False, simulate_water = False, gradient = False, verbose = False, stackelberg = False):
        self.step_size = step_size
        self.time = time
        self.eco = eco
        self.simulate

def constraint_builder(M, classes):
    lower_bound = np.zeros(M.shape[0]*classes)
    upper_bound = np.array([np.inf]*M.shape[0]*classes)
    identity_part = np.identity(M.shape[0]*classes)

    matrix_vec = M.sum(axis = 0)
#    print(matrix_vec)
    matrix = np.zeros((classes, M.shape[0]*classes))
    for i in range(classes):
        matrix[i, i*M.shape[0]: (i+1)*M.shape[0]] = matrix_vec

    one_bound = np.repeat(1, classes)

    bounds = optm.Bounds(lower_bound, upper_bound)

    return matrix, one_bound, bounds

def loss_func_coefficients(vec, size_classes = None, layers = None, spec = None, eco = None): #Remove, deprecated


    coeffs = vec.reshape((layers, size_classes))
    point_vec = np.dot(spec.vandermonde, coeffs)
    point_loss = eco.loss_function(point_vec.flatten())
    point_loss = point_loss.reshape((layers, size_classes)) #Attempt at finding minimum via. coefficients


    return np.sum(np.dot(point_loss.T, np.dot(spec.M, point_loss))) #np.sum((np.linalg.norm(loss_vec, axis = 0))**2) #np.sum(np.dot(loss_vec, np.dot(spec.M, loss_vec.T))) #

def loss_func(vec, size_classes = None, layers = None, spec = None): #Remove, deprecated

    loss_vec = np.reshape(vec, (size_classes, layers))

    return (np.sum((np.linalg.norm(loss_vec, axis=0))))**2  # np.sum(np.dot(loss_vec, np.dot(spec.M, loss_vec.T)))

def sequential_nash(eco, verbose = False):
    x_temp = np.copy(eco.strategy_matrix)  # np.zeros(size_classes*layers)
    x_temp2 = np.copy(eco.strategy_matrix)
    error = 1
#    A, one, bounds = constraint_builder(eco.spectral.M, eco.mass_vector.shape[0])
#    ibr_mode = True
#    newton_step = 0.1
#    cutoff = 1000000000
    iterations = 0
    while error > 10 ** (-2) and iterations < 40: #This is a bit large...
        for k in range(eco.mass_vector.shape[0]):
            result = eco.one_actor_growth(k, x_temp)
            x_temp2[k] = np.array(result).flatten()
            if verbose is True:
                print(np.dot(eco.ones, np.dot(eco.spectral.M, (result) ** 2)), k, "2 Norm of strategy")

            #print(x_temp.shape, x_temp2.shape)

        error = np.max(np.abs(x_temp - x_temp2))
        if verbose is True:
            print("Error: ", error)
           # print(np.dot(eco.ones, np.dot(eco.spectral.M,  (result)**2)), k, "2 Norm of strategy")

        x_temp = np.copy(x_temp2)
        iterations += 1

#        if error > cutoff:
#            ibr_mode = False
#        elif error < cutoff:
#            ibr_mode = True
#        if ibr_mode is True:
#
#            if verbose is True:
#                print("IBR Testing")#, np.dot(A[0:eco.layers], x_temp2**2), np.dot(A[0:eco.layers], x_temp**2))
#        elif ibr_mode is False:
#            if verbose is True:
#                print("Newtoning ", error)
#            x_temp = x_temp + newton_step*x_temp2 #Not sure if this is a good idea...
#            normalization_constants = np.dot(A, x_temp**2)
#            for i in range(eco.mass_vector[0]):
#                x_temp[i*eco.layers:(i+1)*eco.layers] = x_temp[i*eco.layers:(i+1)*eco.layers]/normalization_constants[i]
#            #newton_step = newton_step*1/2
            #print(iterations, newton_step)
    if iterations >= 40:
#        constr = ({'type': 'eq', 'fun': lambda x: np.dot(A[0:eco.layers], x**2) - 1})
#        opt_obj = optm.minimize(lambda x: loss_func(eco.loss_function(x), size_classes= eco.mass_vector.shape[0],
#                                                    layers = eco.layers, spec = eco.spectral), x0 = x_temp2, method = 'SLSQP',  constraints = constr, bounds = bounds)
#        print(opt_obj)
        print("Entering L2 mode")
        x_temp = eco.casadi_total_growth()
    return x_temp

def interpolater(old_vec, old_size, new_size, size_classes, old_spec, new_spec):
    new_vec = np.zeros(new_size*size_classes)
    for k in range(size_classes):

        new_strat = new_spec.expander(old_spec = old_spec, small_vec = old_vec[old_size*k:old_size*(k+1)])
        new_strat = np.abs(new_strat) #'#[new_strat < 0] = 0
        new_vec[new_size*k:new_size*(k+1)] = new_strat/np.dot(np.repeat(1, new_size),np.dot(new_spec.M, new_strat))

    return new_vec


def graph_builder(eco): #Move into ecosystem class
    strat_mat = eco.strategy_matrix
    outflows = np.zeros((eco.mass_vector.shape[0], eco.mass_vector.shape[0]))
    for i in range(eco.mass_vector.shape[0]):
        for k in range(eco.parameters.who_eats_who.shape[0]):
            interaction_term = eco.parameters.who_eats_who[k] * eco.populations * eco.parameters.handling_times[k] * eco.parameters.clearance_rate[k]
            layer_action = strat_mat[i,:].reshape((eco.layers, 1)) * eco.parameters.layered_attack[:,i,:]*(interaction_term.reshape((eco.populations.shape[0], 1))*strat_mat).T


            foraging_term = eco.water.res_counts * eco.parameters.forager_or_not[k] * \
                                eco.parameters.handling_times[k] \
                                * eco.parameters.clearance_rate[k] * eco.parameters.layered_foraging[:, k]
                #print(eco.parameters.layered_attack[:,k,i], k, i, strat_mat[i])
            outflows[i,k] = np.dot(strat_mat[k], np.dot(eco.spectral.M, strat_mat[i] * eco.populations[k] * eco.parameters.layered_attack[:, k, i] * eco.parameters.clearance_rate[k] * eco.populations[i])) / \
                        (1 + np.sum(np.dot(eco.ones, np.dot(eco.spectral.M, np.sum(layer_action, axis=1) + foraging_term))))

    return outflows


def replacer_function(x): #Deprecated
    return np.array([x, 1-x]).squeeze()

def nash_refuge(eco, verbose = False): #Deprecated
    x_temp = np.copy(eco.strategy_matrix.flatten())  # np.zeros(size_classes*layers)
    x_temp2 = np.copy(eco.strategy_matrix.flatten())
    error = 1
    bounds1 = [(0.00000001, 1)] #optm.Bounds(np.array([0] * eco.layers), np.array([np.inf] * eco.layers))
    iterations = 0
    while error > 10 ** (-8) and iterations < 2:
        for k in range(eco.mass_vector.shape[0]):
            x_temp2[k * eco.layers] = optm.minimize(
                lambda x: -eco.one_actor_growth(eco.strategy_replacer(replacer_function(x), k, x_temp), k),
                x0=x_temp[eco.layers * k], bounds=bounds1).x
            x_temp2[k * eco.layers + 1] = 1 - x_temp2[k * eco.layers]

        if verbose is True:
            print("Error: ", np.max(np.abs(x_temp - x_temp2)))
        error = np.max(np.abs(x_temp - x_temp2))
        iterations += 1
        if iterations >= 100:
            print(iterations, "aaaw man, it failed :(", error, x_temp - x_temp2, eco.populations)

        x_temp = np.copy(x_temp2+x_temp)*1/2
    return x_temp