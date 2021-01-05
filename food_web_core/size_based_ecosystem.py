import scipy as scp
import casadi as ca
import scipy.stats as stats
import numpy as np

#Implement simulation class

class ecosystem_optimization:

    def __init__(self, mass_vector, layers, parameters, spectral, water, l2 = True, output_level = 0, verbose = False, movement_cost = 0.1, time_step = 10**(-4)):
        self.layers = spectral.n*spectral.segments
        self.mass_vector = mass_vector
        self.spectral = spectral
        self.strategy_matrix = np.vstack([np.repeat(1/(spectral.x[-1]), spectral.segments*spectral.n)]*mass_vector.shape[0]) # np.zeros((mass_vector.shape[0], layers))
        self.populations = parameters.mass_vector**(-0.75)
        self.parameters = parameters
        self.ones = np.repeat(1, spectral.n*spectral.segments)
        self.water = water
        self.l2 = l2
        self.output_level = output_level
        self.verbose = verbose
        self.movement_cost = movement_cost
        self.heat_kernels = np.zeros((self.populations.shape[0], self.spectral.M.shape[0], self.spectral.M.shape[0]))
        self.heat_kernel_creator(time_step)
        for i in range(self.populations.shape[0]):
            norm_const = np.sum(spectral.M @ (self.strategy_matrix[i] @ self.heat_kernels[i]))
            self.strategy_matrix[i] = 1/norm_const*self.strategy_matrix[i]

    def one_actor_growth(self, i, x_temp_i = None, solve_mode = True, time_step = 10**(-4)):
        temp_pops = np.copy(self.populations)
        temp_pops[i] = 0
        x_temp = np.copy(x_temp_i)

        if x_temp_i is None:
            x_temp = np.copy(self.strategy_matrix)

        for k in range(self.populations.shape[0]):
            x_temp[k] = x_temp[k] @ self.heat_kernels[k] #Going smooth.


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
                                * self.parameters.clearance_rate[i] * self.parameters.layered_foraging[:, i]).reshape((1,self.layers)) @ (self.spectral.M @ (x.T @ self.heat_kernels[i]).T)])

        A = np.tri(self.layers)
        cum_diff = A @ (self.spectral.M @ ((self.strategy_matrix[i] @ self.heat_kernels[i]).reshape(x.size()) - (x.T @ self.heat_kernels[i]).T))


        actual_growth = self.parameters.efficiency * (lin_growth(x) + foraging_term_self(x)) \
                                / (1 + self.parameters.handling_times[i] * (lin_growth(x)+foraging_term_self(x))) - self.movement_cost*time_step * cum_diff.T @ self.spectral.M @ cum_diff

        pred_loss = ca.dot((predator_hunger @ (x.T @ self.heat_kernels[i]).T / (1 + self.parameters.handling_times.reshape((self.populations.shape[0], 1))*(self.populations[i]*predator_hunger @ (x.T @ self.heat_kernels[i]).T) + predation_ext_food)), self.populations) #ca.dot((x.T @ predator_hunger.T).T, self.populations)   This snippet is for the linear case, known good
        if self.verbose is True:
            print("Size Class: ", i, "Predation loss", ca.substitute(pred_loss, x, self.strategy_matrix[i]), "Actual growth", ca.substitute(actual_growth, x, self.strategy_matrix[i]))

        delta_p = ca.Function('delta_p', [x], [(actual_growth - pred_loss)], ['x'], ['r'])

        if solve_mode is True:
            pop_change = -(actual_growth - pred_loss)
            if self.l2 is True:
                g = x.T @ self.spectral.M @ x - 1
                arg = {"lbg": 0, "ubg": 0}

            else:
                g = ca.dot(self.ones, self.spectral.M @ (x.T @ self.heat_kernels[i]).T) - 1 #(x.T @ self.heat_kernels[i]).T)
                arg = {"lbg": 0, "ubg": 0}


            x_min = [0]*self.layers
            x_max = [1]*self.layers
            arg["lbx"] = x_min
            arg["ubx"] = x_max
            s_opts = {'ipopt': {'print_level' : self.output_level}}
            prob = {'x': x, 'f': pop_change, 'g': g}
            solver = ca.nlpsol('solver', 'ipopt', prob, s_opts)
            sol = solver(x0=x_temp[i], **arg)
            x_out = np.array(sol['x'])
            x_out = x_out #self.heat_kernels[i] @

        else:
            x_out = delta_p(x_temp[i])

        return x_out


    def heat_kernel_i(self, t, k):

        gridx, gridy = np.meshgrid(self.spectral.x, self.spectral.x)
        ker = lambda x, y: np.exp(-(x - y) ** 2 / (4 * k * t)) + np.exp(-(-y - x) ** 2 / (4 * k * t)) \
                           + np.exp(-(2*self.spectral.x[-1] - x - y) ** 2 / (4 * k * t)) #np.exp(-(x - y) ** 2 / (4 * k * t))
        out = (4 * t * k * np.pi) ** (-1 / 2) * ker(gridx, gridy)
        normalizations = self.spectral.M @ (self.ones @ (self.spectral.M @ out))
        normalizations = np.diag(1/normalizations)
        return normalizations @ self.spectral.M @ out


    def heat_kernel_creator(self, t, k = None):
        if k is None:
            for i in range(self.populations.shape[0]):
                self.heat_kernels[i] = self.heat_kernel_i(t, self.parameters.clearance_rate[i])
        else:
            for i in range(self.populations.shape[0]):
                self.heat_kernels[i] = self.heat_kernel_i(t, k)



    def dirac_delta_creator_i(self, i, normalize = True):
        I_n = np.identity(self.layers)
        normalizations = np.sum(self.spectral.M @ I_n, axis = 0)
        #normalizations = self.spectral.M @ (self.ones @ (self.spectral.M @ out))
        if normalize is True:
            normalizations = np.diag(1/normalizations)
        else:
            normalizations = np.diag(normalizations)
        return normalizations @ I_n


    def dirac_delta_creator(self):
        for i in range(self.populations.shape[0]):
            self.heat_kernels[i] = self.dirac_delta_creator_i(i)


    def total_growth(self, x_temp = None):
        if x_temp is None:
            x_temp = self.strategy_matrix

        total_growth = np.zeros((self.populations.shape[0]))
        for i in range(self.populations.shape[0]):
            total_growth[i] = self.one_actor_growth(i, x_temp_i=x_temp, solve_mode=False)
        #print(self.parameters.loss_term, total_growth)
        return (total_growth - self.parameters.loss_term)*self.populations

    def strategy_setter(self, strat_vec): #Keep in population state class
        s_m = np.reshape(strat_vec, self.strategy_matrix.shape)
        self.strategy_matrix = s_m

    def population_setter(self, new_pop): #Keep in population state class
        self.populations = new_pop #self.total_growth()*time_step

    def consumed_resources(self): #Move to simulator class
        strat_mat = np.copy(self.strategy_matrix)
        for i in range(strat_mat.shape[0]):
            strat_mat[i] = strat_mat[i] @ self.heat_kernels[i]
        consumed_resources = np.zeros(self.layers)

        for i in range(len(self.mass_vector)):
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

    def lin_growth(self, i, j, strategy, current_layered_attack = None):
        if current_layered_attack is None:
            current_layered_attack = self.parameters.layered_attack

        x_temp = np.copy(strategy)
        x_temp[0] = x_temp[0] @ self.heat_kernels[i]  # Going smooth.

        x_temp[1] = x_temp[1] @ self.heat_kernels[j]  # Going smooth.

        predator_hunger = self.parameters.clearance_rate[j] * self.populations[j] * np.dot(self.spectral.M, current_layered_attack[:, j, i] * x_temp[1]) * self.parameters.who_eats_who[j, i]

        x = x_temp[0].reshape(-1, 1)
        interaction_term = self.parameters.who_eats_who[i, j] * self.populations[j] * self.parameters.clearance_rate[i]
        lin_growth = interaction_term * (x_temp[1] * current_layered_attack[:, i, j].T) @ self.spectral.M @ x

        foraging_term_self = (self.water.res_counts * self.parameters.forager_or_not[i] *
                              self.parameters.clearance_rate[i] * self.parameters.layered_foraging[:, i]).reshape(
            (1, self.layers)) @ (self.spectral.M @ x)
        foraging_term_self = foraging_term_self / (self.populations.size - 1)

        actual_growth = self.parameters.efficiency * (lin_growth + foraging_term_self)

        pred_loss = x.T @ predator_hunger

        return actual_growth - pred_loss

class spectral_method:
    def __init__(self, depth, layers, segments = 1):

        self.n = layers

#        JacobiGL = lambda x, y, z: eng.JacobiGL(float(x), float(y), float(z), nargout=1)

        self.x = self.JacobiGL(0, 0, layers-1)

        D_calc = lambda n: np.matmul(np.transpose(self.vandermonde_dx()),
                                                   np.linalg.inv(np.transpose(self.vandermonde_calculator())))*(depth/2)

        self.D = D_calc(layers)
        M_calc = lambda n: np.dot(np.linalg.inv(self.vandermonde_calculator()),
                                       np.linalg.inv(np.transpose(self.vandermonde_calculator())))*(depth/2)

        self.M = M_calc(layers)
        self.x = ((self.x+1)/2) * depth
        self.segments = segments

        if segments>1:
            M_T = np.zeros((layers*segments, layers*segments))
            x_T = np.zeros(layers*segments)
            s_x = depth/segments
            x_n = np.copy(self.x)/segments

            for k in range(segments):
                M_T[k*layers:(k+1)*layers, k*layers:(k+1)*layers] = self.M/segments
                x_T[k*layers:(k+1)*layers] = x_n + k*s_x
            self.M = M_T
            self.x = x_T

    def JacobiGL(self, a, b, n):

        alpha = a + 1
        beta = b + 1
        N = n - 2
        if N == 0:
            x = np.array([(alpha - beta) / (alpha + beta + 2)])
            w = 2
            return x, w
        else:
            h1 = 2 * np.arange(0, N + 1) + alpha + beta
            J1 = np.diag(-1 / 2 * (alpha ** 2 - beta ** 2) / (h1 + 2) / h1)
            J2 = np.diag(2 / (h1[0:N] + 2) * np.sqrt(np.arange(1, N + 1) * (np.arange(1, N + 1) + alpha + beta) *
                                                     (np.arange(1, N + 1) + alpha) * (np.arange(1, N + 1) + beta) * (
                                                                 1 / (h1[0:N] + 1)) * (1 / (h1[0:N] + 3))), 1)
            J = J1 + J2
            J = J + J.T
            x, w = np.linalg.eig(J)

        return np.array([-1, *np.sort(x), 1])

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
        return (self.JacobiP_n(x, 0, 0, n))


    def vandermonde_dx(self):
        n = self.n
        x = self.x
        return (self.GradJacobi_n(x, 0, 0, n))

    def expander(self, old_spec = None, small_vec = None):

        new_vm = self.JacobiP((self.x/(self.x[-1])-1), 0, 0, small_vec.shape[0])
        old_vm = np.linalg.inv(old_spec.vandermonde_calculator())

        #print(small_vec)
        return(np.dot(new_vm.T, np.dot(old_vm.T, small_vec)))

    def projector(self, old_spec, big_vec):
        pass

    def interpolater(self, old_vec, old_size, new_size, size_classes, old_spec, new_spec):
        new_vec = np.zeros(new_size * size_classes)
        for k in range(size_classes):
            new_strat = new_spec.expander(old_spec=old_spec, small_vec=old_vec[old_size * k:old_size * (k + 1)])
            new_strat = np.abs(new_strat)  # '#[new_strat < 0] = 0
            new_vec[new_size * k:new_size * (k + 1)] = new_strat / np.dot(np.repeat(1, new_size),
                                                                          np.dot(new_spec.M, new_strat))

        return new_vec


class ecosystem_parameters:
    def __init__(self, mass_vector, spectral, lam = 0.8, min_attack_rate = 10**(-4), forage_mass = 0.05):
        self.forage_mass = forage_mass
        self.min_attack_rate = min_attack_rate

        self.mass_vector = mass_vector
        self.spectral = spectral
        self.attack_matrix = self.attack_matrix_setter()
        self.handling_times = self.handling_times_setter()
        self.who_eats_who = self.who_eats_who_setter()
        self.clearance_rate = self.clearance_rate_setter() #330/12 * mass_vector**(3/4)
        self.layered_attack = self.layer_creator(self.attack_matrix, lam = lam)
        self.efficiency = 0.7 #Very good number.
        self.loss_term = self.loss_rate_setter() #mass_vector**(0.75)*0.01
        self.forager_or_not = self.forager_or_not_setter()
        self.foraging_attack_prob = self.foraging_attack_setter()
        self.layered_foraging = self.layer_creator(self.foraging_attack_prob, lam = lam)
        self.layered_foraging = self.layered_foraging/self.layered_foraging
    def forager_or_not_setter(self):
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
        who_eats_who[who_eats_who > 2000] = 0

        who_eats_who[who_eats_who != 0] = 1

        who_eats_who = who_eats_who.T

        return who_eats_who

    def handling_times_setter(self):
        return 1/(22.3*self.mass_vector**(0.75))

    def loss_rate_setter(self):
        return 0.1*22.3*self.mass_vector**(0.75) #Used to be 1.2, but this was annoying

    def layer_creator(self, obj, lam = 0.8):
        weights = 2/(1+np.exp(lam*self.spectral.x)) #Replace with actual function
        layers = np.zeros((self.spectral.x.shape[0], *obj.shape))
       # print(layers.shape, obj.shape, self.spectral.x.shape[0])

        for i in range(self.spectral.x.shape[0]):
            layers[i] = (weights[i] + self.min_attack_rate)* obj
        return layers

    def clearance_rate_setter(self):
        return 330*self.mass_vector**(0.75) #This should probably be more sophisticated... But a good first approximation

    def layered_attack_setter(self, layered_attack):
        self.layered_attack = layered_attack


class water_column:
    def __init__(self, spectral, res_vec, advection = 1, diffusion = 0.5, resource_max = None, replacement = 1.2, layers = 2, logistic = False):
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

        self.logistic = logistic
    def resource_setter(self, new_res):
        self.res_counts = new_res

    def update_resources(self, consumed_resources = 0, time_step = 0.001):
        ##Chemostat step
        if self.logistic is False:
            self.res_counts += (self.lam*(self.resource_max - self.res_counts) - consumed_resources)*time_step
            self.res_counts[self.res_counts < 0] = 0

        else:
            self.res_counts += (self.lam*self.res_counts*(1-self.res_counts/self.resource_max) - consumed_resources)*time_step
            self.res_counts[self.res_counts < 0] = 10 ** (-8)

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


