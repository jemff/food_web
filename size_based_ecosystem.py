import scipy as scp
import casadi as ca
import scipy.stats as stats
import lemkelcp as lcp
from matrix_infrastructure import *
from numba import njit
#eng = matlab.engine.start_matlab()

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


def jit_wrapper(eco, i, j, current_layered_attack, dirac_mode = False):
    resources = eco.water.res_counts
    populations = eco.populations
    current_layered_attack = current_layered_attack
    heat_kernel = eco.heat_kernels[0]
    clearance_rate = eco.parameters.clearance_rate
    M = eco.spectral.M
    who_eats_who = eco.parameters.who_eats_who
    forager_or_not = eco.parameters.forager_or_not
    layered_foraging = eco.parameters.layered_foraging
    efficiency = eco.parameters.efficiency
    payoff_i = jit_payoff_matrix_builder(resources, populations, i, j, current_layered_attack = current_layered_attack,
                   heat_kernel = heat_kernel, clearance_rate = clearance_rate,
                   M = M, who_eats_who = who_eats_who,
                   forager_or_not = forager_or_not, layered_foraging = layered_foraging, efficiency = efficiency)

    return payoff_i


#@njit
def lin_growth_jit(resources, populations, i, j, strategy, current_layered_attack = None,
                   heat_kernel = None, clearance_rate = None,
                   M = None, who_eats_who = None,
                   forager_or_not = None, layered_foraging = None, efficiency = None):

    x_temp = np.copy(strategy)
    x_temp[0] = x_temp[0] @ heat_kernel  # Going smooth.

    x_temp[1] = x_temp[1] @ heat_kernel  # Going smooth.

    predator_hunger = clearance_rate[j] * populations[j] * np.dot(M, current_layered_attack[:, j, i] * x_temp[1]) * who_eats_who[j, i]

    x = x_temp[0].reshape(-1, 1)
    interaction_term = who_eats_who[i, j] * populations[j] * clearance_rate[i]
    lin_growth = interaction_term * (x_temp[1] * current_layered_attack[:, i, j].T) @ M @ x

    foraging_term_self = (resources * forager_or_not[i] *
                          clearance_rate[i] * layered_foraging[:, i]).reshape(
        (1, -1)) @ (M @ x)
    foraging_term_self = foraging_term_self / (populations.size - 1)

    actual_growth = efficiency * (lin_growth + foraging_term_self)

    pred_loss = x.T @ predator_hunger

    return actual_growth - pred_loss

@njit(fastmath=True)
def jit_payoff_matrix_builder(resources, populations, i, j, current_layered_attack,
                   heat_kernel = None, clearance_rate = None,
                   M = None, who_eats_who = None,
                   forager_or_not = None, layered_foraging = None, efficiency = None):
    layers = current_layered_attack.shape[0]
    payoff_i = np.zeros((current_layered_attack.shape[0], current_layered_attack.shape[0]))
    for k in range(current_layered_attack.shape[0]):
        one_k_vec = np.zeros(layers)
        one_k_vec[k] = 1
        for n in range(layers):
            one_n_vec = np.zeros(layers)
            one_n_vec[n] = 1
            strat_mat = np.vstack((one_k_vec, one_n_vec))
            x_temp = np.copy(strat_mat)
            x_temp[0] = x_temp[0] @ heat_kernel  # Going smooth.

            x_temp[1] = x_temp[1] @ heat_kernel  # Going smooth.

            predator_hunger = clearance_rate[j] * populations[j] * np.dot(M, current_layered_attack[:, j, i] * x_temp[1]) * \
                              who_eats_who[j, i]

            x = x_temp[0].reshape(-1, 1)
            interaction_term = who_eats_who[i, j] * populations[j] * clearance_rate[i]
            lin_growth = interaction_term * (x_temp[1] * current_layered_attack[:, i, j].T) @ M @ x

            foraging_term_self = (resources * forager_or_not[i] *
                                  clearance_rate[i] * layered_foraging[:, i]).reshape(
                (1, -1)) @ (M @ x)
            foraging_term_self = foraging_term_self / (populations.size - 1)

            actual_growth = efficiency * (lin_growth[0] + foraging_term_self[0])

            pred_loss = x.T @ predator_hunger
            #print(pred_loss, actual_growth)

            payoff_i[k, n] = actual_growth[0] - pred_loss[0]

            #lin_growth_jit(resources, populations, i, j, strat_mat, current_layered_attack = current_layered_attack,
                   #heat_kernel = heat_kernel, clearance_rate = clearance_rate,
                   #M = M, who_eats_who = who_eats_who,
                   #forager_or_not = forager_or_not, layered_foraging = layered_foraging, efficiency = efficiency).astype(float)

    return payoff_i

def lin_growth_no_pops_no_res(eco, i, j, layered_attack, strategy):
    x_temp = np.copy(strategy)
    x_temp[0] = x_temp[0] @ eco.heat_kernels[i]  # Going smooth.

    x_temp[1] = x_temp[1] @ eco.heat_kernels[j]  # Going smooth.

    x = x_temp[0].reshape(-1, 1)
    interaction_term = eco.parameters.who_eats_who[i, j] * eco.parameters.clearance_rate[i]
    lin_growth = interaction_term * (x_temp[1] * layered_attack[:, i, j].T) @ eco.spectral.M @ x

    actual_growth = lin_growth

    return actual_growth


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


def lemke_optimizer(eco, payoff_matrix = None):
    A = np.zeros((eco.populations.size, eco.populations.size * eco.layers))
    for k in range(eco.populations.size):
        A[k, k * eco.layers:(k + 1) * eco.layers] = -1

    q = np.zeros(eco.populations.size + eco.populations.size * eco.layers)
    q[eco.populations.size * eco.layers:] = -1
    q = q.reshape(-1, 1)
    if payoff_matrix is None:
        payoff_matrix = total_payoff_matrix_builder(eco)

    H = np.block([[-payoff_matrix, A.T], [-A, np.zeros((A.shape[0], eco.populations.size))]])

    sol = lcp.lemkelcp(H, q, maxIter = 10000)
    if sol[1]==0:
        return sol[0][0:eco.layers*eco.populations.size].reshape((eco.populations.size, -1))
    else:
        return quadratic_optimizer(eco, payoff_matrix=payoff_matrix)[0:eco.layers*eco.populations.size].reshape((eco.populations.size, -1))

def quadratic_optimizer(eco, payoff_matrix = None, prior_sol=None):

    A=np.zeros((eco.populations.size, eco.populations.size*eco.layers))
    if eco.spectral.segments == 1:
        for k in range(eco.populations.size):
            A[k,k*eco.layers:(k+1)*eco.layers] = -1
    if eco.spectral.segments != 1:
        Temp = -np.copy(eco.ones).astype(float)
        Temp[::eco.spectral.n] += 1/2
        Temp[eco.spectral.n-1::eco.spectral.n] += 1/2

        Temp[0] = -1
        Temp[-1] = -1
        print(Temp)
        for k in range(eco.populations.size):
            A[k, k * eco.layers:(k + 1) * eco.layers] = -Temp

    for k in range(eco.populations.size):
        A[k, k * eco.layers:(k + 1) * eco.layers] = -1

    q = np.zeros(eco.populations.size+eco.populations.size*eco.layers)
    q[eco.populations.size*eco.layers:] = -1
    q = q.reshape(-1, 1)
    if payoff_matrix is None:
        payoff_matrix = total_payoff_matrix_builder(eco)

    p = ca.SX.sym('p', eco.populations.size*eco.layers)
    y = ca.SX.sym('y', eco.populations.size)
    u = ca.SX.sym('u', eco.populations.size*eco.layers)
    v = ca.SX.sym('v', eco.populations.size)

    cont_conds = []
    if eco.spectral.segments>1:
        for j in range(eco.spectral.segments-1):
            cont_conds.append(p[j*eco.spectral.n] - p[(j+1)*eco.spectral.n])
            cont_conds.append(u[j*eco.spectral.n] - u[(j+1)*eco.spectral.n])

    print("Here")
    z = ca.vertcat(*[p, y])
    w = ca.vertcat(*[u, v])
    #
#    print(np.block([-A]).shape)
    H = np.block([[-payoff_matrix, A.T], [-A, np.zeros((A.shape[0], eco.populations.size))]])
  #  w = H @ z + q

#    print(H.shape)
    print("Here")
    f = ca.norm_2(w.T @ z) #ca.norm_2()
    if eco.spectral.segments > 1:
        g = ca.vertcat(*[*cont_conds, w - H @ z - q])

    else:
        g = w - H @ z - q #ca.norm_2() H @ z + q

    print("Here")

    x = z #ca.vertcat(z, w) #
    lbx = np.zeros(x.size())
    ubg = np.zeros(g.size()) #[ca.inf]*int(g.size()[0]) #np.zeros(g.size())
    lbg = np.zeros(g.size())

    print("Just before optimizing")
    s_opts = {'ipopt': {'print_level': 5, 'tol':1E-5}}
    prob = {'x': x, 'f': f, 'g': g}
    solver = ca.nlpsol('solver', 'ipopt', prob, s_opts)
    print("Solver decleared")
    #prior_sol = False
    if prior_sol is None:
        sol = solver(lbx=lbx, lbg=lbg, ubg = ubg) #ubg=ubg
        print("Solved")

    else:
        sol = solver(x0=prior_sol, lbx=lbx, lbg=lbg, ubg = ubg) #ubg=ubg,


    print(sol['f'])
    x_out = np.array(sol['x']).flatten()
    #print(np.min(x_out), np.dot(x_out[0:eco.populations.size*(eco.layers+1)], x_out[eco.populations.size*(eco.layers+1):]))
    #print(x_out[0:eco.layers*eco.populations.size])
    return x_out





def total_payoff_matrix_builder(eco, current_layered_attack = None, dirac_mode = False):
    total_payoff_matrix = np.zeros((eco.populations.size*eco.spectral.n*eco.spectral.segments, eco.populations.size*eco.spectral.n*eco.spectral.segments))

    if current_layered_attack is None:
        current_layered_attack = eco.parameters.layered_attack

    for i in range(eco.populations.size):
        for j in range(eco.populations.size):
            if i != j:
                i_vs_j = jit_wrapper(eco, i, j, current_layered_attack = current_layered_attack, dirac_mode = dirac_mode) #payoff_matrix_builder
            elif i == j:
                i_vs_j = np.zeros((eco.layers, eco.layers))
            #if i == 1:
            #    total_payoff_matrix[i*eco.layers:(i+1)*eco.layers, j*eco.layers: (j+1)*eco.layers] = i_vs_j.T
            #else:

            total_payoff_matrix[i * eco.layers:(i + 1) * eco.layers, j * eco.layers: (j + 1) * eco.layers] = i_vs_j
#    print("MAXIMM PAYDAY ORIGINAL",  np.max(total_payoff_matrix))
    total_payoff_matrix = total_payoff_matrix - np.max(total_payoff_matrix) #- 1 #Making sure everything is negative  #- 0.00001
    #total_payoff_matrix = total_payoff_matrix/np.max(-total_payoff_matrix)
    return total_payoff_matrix


#@njit(parallel=True)
def payoff_matrix_builder(eco, i, j, current_layered_attack, dirac_mode = False):
    payoff_i = np.zeros((current_layered_attack.shape[0], current_layered_attack.shape[0]))
    diracs = eco.dirac_delta_creator_i(0, normalize = True)
    for k in range(current_layered_attack.shape[0]):
        one_k_vec = np.zeros(eco.layers)
        if dirac_mode is True:
            one_k_vec = diracs[i] #np.zeros(eco.layers)
        else:
            one_k_vec[k] = 1
        for n in range(eco.layers):
            one_n_vec = np.zeros(eco.layers)
            if dirac_mode is True:
                one_n_vec = diracs[n]
            else:
                one_n_vec[n] = 1
            strat_mat = np.vstack([one_k_vec, one_n_vec])
            payoff_i[k, n] = eco.lin_growth(i, j, strat_mat, current_layered_attack)

    return payoff_i




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
        x_temp[0] = strategies[i] @ eco.heat_kernels[0]  # Going smooth.
        inflows[i + 1, 0] = (eco.parameters.forager_or_not[i] * eco.parameters.clearance_rate[
            i] * eco.parameters.layered_foraging[:, i] * resources) @ (eco.spectral.M @ x_temp[0])

        for j in range(classes):
            x_temp[1] = strategies[j] @ eco.heat_kernels[0]  # Going smooth.
            x = x_temp[0].reshape(-1, 1)
            interaction_term = eco.parameters.who_eats_who[i, j] * eco.parameters.clearance_rate[i]
            lin_growth = interaction_term * (x_temp[1] * layered_attack[:, i, j].T) @ (eco.spectral.M @ x)

            actual_growth = eco.parameters.efficiency * lin_growth

            inflows[i + 1, j + 1] = actual_growth * populations[j]

    return inflows


def periodic_attack(layered_attack, day_interval = 96, darkness_length = 0, minimum_attack = 0):
    OG_layered_attack = np.copy(layered_attack)
    periodic_layers = np.zeros((day_interval,*layered_attack.shape))
    for i in range(day_interval):
        periodic_layers[i] = 1/2*(1+minimum_attack+min(max((darkness_length+1)*np.cos(i*2*np.pi/day_interval), -1), 1))*OG_layered_attack

    return periodic_layers

def reward_loss_time_dependent(eco, periodic_layers, dirac_mode = False):
    rewards_t = np.zeros((periodic_layers.shape[0], eco.layers*eco.populations.size, eco.layers*eco.populations.size))
    losses_t = np.zeros((periodic_layers.shape[0], eco.layers*eco.populations.size, eco.layers*eco.populations.size))
    for i in range(periodic_layers.shape[0]):
        print(i)
        reward_i, loss_i = loss_and_reward_builder(eco, periodic_layers[i], dirac_mode = dirac_mode)
        rewards_t[i] = reward_i
        losses_t[i] = loss_i

    return rewards_t, losses_t


def total_payoff_matrix_builder_memory_improved(eco, populations, total_reward_matrix, total_loss_matrix, foraging_gain):
    total_rew_mat = np.copy(total_reward_matrix)
    total_loss_mat = np.copy(total_loss_matrix)

    for i in range(eco.populations.size):
        for j in range(eco.populations.size):
            total_rew_mat[i * eco.layers:(i + 1) * eco.layers, j * eco.layers: (j + 1) * eco.layers] = populations[j]*total_reward_matrix[i * eco.layers:(i + 1) * eco.layers, j * eco.layers: (j + 1) * eco.layers]
            total_loss_mat[i * eco.layers:(i + 1) * eco.layers, j * eco.layers: (j + 1) * eco.layers] = populations[j]*total_loss_matrix[i * eco.layers:(i + 1) * eco.layers, j * eco.layers: (j + 1) * eco.layers]

    total_payoff_matrix = eco.parameters.efficiency*(total_rew_mat + foraging_gain) - total_loss_mat
    #print(np.max(total_payoff_matrix), "MAXIMUM PAYDAY")
    return total_payoff_matrix - np.max(total_payoff_matrix) #- 0.00001

def foraging_gain_builder(eco, resources = None, dirac_mode = False):
    if resources is None:
        resources = eco.water.res_counts

    foragers = np.where(eco.parameters.forager_or_not == 1)
    foraging_gain = np.zeros((eco.populations.size*eco.layers, eco.populations.size*eco.layers))
    foraging_gain_i = np.zeros((eco.layers, eco.layers))
    diracs = eco.dirac_delta_creator_i(0, normalize=False)

    for forager in foragers:
        for i in range(eco.layers):
            one = np.zeros(eco.layers)
            if dirac_mode is True:
                one = diracs[i]  # np.zeros(eco.layers)
            else:
                one[i] = 1
            foraging_gain_i[i] = eco.parameters.clearance_rate[forager[0]]*(one @ eco.heat_kernels[forager[0]] * eco.parameters.layered_foraging[:,forager[0]]) @ (eco.spectral.M @ (resources))/np.sum(eco.parameters.who_eats_who[:, forager[0]])

        eaters = np.where(eco.parameters.who_eats_who[:,forager[0]] == 1)
        for eater in eaters:
            foraging_gain[forager[0] * eco.layers:(forager[0] + 1) * eco.layers, eater[0] * eco.layers: (eater[0] + 1) * eco.layers] = foraging_gain_i

    return foraging_gain



def depth_based_loss(who_eats_who, depth_loss_matrix, heat_kernels):
    layers = depth_loss_matrix.shape[1]
    actors = depth_loss_matrix.shape[0]
    giant_loss_matrix = np.zeros((layers*actors, layers*actors))
    for i in range(depth_loss_matrix.shape[0]):
        eaters = np.where(who_eats_who[:,i] == 1)
        for eater in eaters:
            giant_loss_matrix[i * layers:(i + 1) * layers, eater[0] * layers: (eater[0] + 1) * layers] = heat_kernels[i]@depth_loss_matrix[i]/np.sum(who_eats_who[:, i])

    return giant_loss_matrix


def loss_and_reward_builder(eco, layered_attack = None, dirac_mode = False):
    if layered_attack is None:
        layered_attack = eco.parameters.layered_attack

    total_reward_matrix = np.zeros((eco.populations.size*eco.layers, eco.populations.size*eco.layers))
    total_loss_matrix = np.zeros((eco.populations.size*eco.layers, eco.populations.size*eco.layers))
    for i in range(eco.populations.size):
        for j in range(eco.populations.size):
            if i != j:
                i_vs_j = reward_matrix_builder(eco, layered_attack, i, j, dirac_mode = dirac_mode)
                j_vs_i = loss_matrix_builder(eco, layered_attack, i, j, dirac_mode = dirac_mode)
            elif i == j:
                i_vs_j = np.zeros((eco.layers, eco.layers))
                j_vs_i = np.zeros((eco.layers, eco.layers))

            total_reward_matrix[i * eco.layers:(i + 1) * eco.layers, j * eco.layers: (j + 1) * eco.layers] = i_vs_j
            total_loss_matrix[i * eco.layers:(i + 1) * eco.layers, j * eco.layers: (j + 1) * eco.layers] = j_vs_i

    return total_reward_matrix, total_loss_matrix


def reward_matrix_builder(eco, layered_attack, i, j, dirac_mode = False):
    reward_i = np.zeros((eco.layers, eco.layers))
    diracs = eco.dirac_delta_creator_i(0, normalize = True)

    for k in range(eco.layers):
        one_k_vec = np.zeros(eco.layers)
        if dirac_mode is True:
            one_k_vec = diracs[i] #np.zeros(eco.layers)
        else:
            one_k_vec[k] = 1
        for n in range(eco.layers):
            one_n_vec = np.zeros(eco.layers)
            if dirac_mode is True:
                one_n_vec = diracs[n]
            else:
                one_n_vec[n] = 1
            strat_mat = np.vstack([one_k_vec, one_n_vec])
            reward_i[k, n] = lin_growth_no_pops_no_res(eco, i, j, layered_attack, strat_mat)

    return reward_i


def loss_matrix_builder(eco, layered_attack, i, j, dirac_mode = False):
    loss_i = np.zeros((eco.layers, eco.layers))
    diracs = eco.dirac_delta_creator_i(0, normalize = True)
    for k in range(eco.layers):
        one_k_vec = np.zeros(eco.layers)
        if dirac_mode is True:
            one_k_vec = diracs[i] #np.zeros(eco.layers)
        else:
            one_k_vec[k] = 1
        for n in range(eco.layers):
            one_n_vec = np.zeros(eco.layers)
            if dirac_mode is True:
                one_n_vec = diracs[n]
            else:
                one_n_vec[n] = 1
            strat_mat = np.vstack([one_k_vec, one_n_vec])
            loss_i[k, n] = lin_growth_no_pops_no_res(eco, j, i, layered_attack, strat_mat)

    return loss_i