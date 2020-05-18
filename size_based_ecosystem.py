import numpy as np
import scipy.optimize as optm
import matlab.engine
import scipy as scp
import itertools as itertools
import matplotlib.pyplot as plt

eng = matlab.engine.start_matlab()

#Add eating functionality of resources so they actually get removed
#Add interpolation logic (Convert to Lagrange, evaluate Lagrange in bigger point set, optimize again)
#Add cheeger constant calculator
#Add graph viewer
#Create extra file


def jacobian_calculator(f, x, h):
    #if len(f(x))>0:
    #    jac = np.zeros((x.shape[0], f(x).shape[0]))
    #else:
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

def grad_calculator(f, x, h):
    grad = np.zeros(x.shape[0])
    #Finish me

class ecosystem_optimization:

    def __init__(self, mass_vector, layers, parameters, spectral, water):
        self.layers = layers
        self.mass_vector = mass_vector
        self.spectral = spectral
        self.strategy_matrix = np.vstack([np.repeat(1/(spectral.x[-1])*1/layers, layers)]*mass_vector.shape[0]) # np.zeros((mass_vector.shape[0], layers))
        self.populations = np.array([10, 0.5, 0.25, 0.125]) #mass_vector**() #Now this is a flat structure
        self.resource_distribution = np.array([range(0,layers)])/layers
        self.parameters = parameters
        self.ones = np.repeat(1, layers)
        self.water = water


    def one_actor_growth(self, strategies, i):
        strat_mat = strategies.reshape(self.strategy_matrix.shape)
        interaction_term = self.parameters.who_eats_who[i]*self.populations
        interaction_term = interaction_term*self.parameters.clearance_rate[i]
        layer_action = np.zeros((self.layers, self.mass_vector.shape[0]))
#        print(self.parameters.layered_attack.shape, strat_mat.shape, self.parameters.handling_times.shape, self.strategy_matrix.shape)
        for j in range(self.layers):
            layer_action[j] = self.parameters.layered_attack[j,i]*strat_mat[i,j]*interaction_term*strat_mat[:,j]\
                              *self.parameters.handling_times[i]
            #print("This is where I die ", self.layers, i, j, self.parameters.layered_attack.shape, self.parameters.handling_times.shape, strat_mat.shape)


        foraging_term = self.water.res_counts * self.parameters.forager_or_not[i] * self.parameters.handling_times[i] \
                        * self.parameters.clearance_rate[i] * self.parameters.layered_foraging[:,i]

        #foraging_term =  np.dot(self.water.res_counts, self.parameters.forager_or_not[i]*self.parameters.handling_times[i]*self.parameters.clearance_rate[i]*self.parameters.layered_foraging)


        growth_term = np.sum(np.dot(self.ones, np.dot(self.spectral.M,np.sum(layer_action, axis = 1)+foraging_term))/(1+np.sum(np.dot(self.ones, np.dot(self.spectral.M,np.sum(layer_action, axis = 1)++foraging_term)))))

        loss = 0
        for k in range(self.parameters.who_eats_who.shape[0]):
            if(self.parameters.who_eats_who[k,i] == 1):
                interaction_term = self.parameters.who_eats_who[k] * self.populations
                interaction_term = interaction_term * self.parameters.handling_times[k] * self.parameters.clearance_rate[k]
                layer_action = np.zeros((self.layers, self.mass_vector.shape[0]))
                for j in range(self.layers):
                    layer_action[j] = self.parameters.layered_attack[j, i] * strat_mat[i, j] * interaction_term * strat_mat[:, j]
                #foraging_term = np.dot(self.water.res_counts, self.parameters.forager_or_not[k] * self.parameters.handling_times[k] * \
                #                self.parameters.clearance_rate[k] * self.parameters.layered_foraging)
                foraging_term = self.water.res_counts * self.parameters.forager_or_not[k] * \
                                self.parameters.handling_times[k] \
                                * self.parameters.clearance_rate[k] * self.parameters.layered_foraging[:,k]

                loss += np.dot(strat_mat[k], np.dot(self.spectral.M, strat_mat[i]*self.populations[k]*self.parameters.layered_attack[:,k,i]*self.parameters.clearance_rate[k]))/(1+np.sum(np.dot(self.ones, np.dot(self.spectral.M,np.sum(layer_action, axis = 1)++foraging_term))))

            if loss == 0:
                loss = self.parameters.loss_term[i]*np.dot(strat_mat[i],np.dot(self.spectral.M, strat_mat[i]))

        return growth_term - loss

    def total_growth(self, strategies = None):
        if strategies is None:
            strategies = np.reshape(self.strategy_matrix, self.layers*self.mass_vector.shape[0])
        #total_growth_tensor = np.array((self.mass_vector.shape[0], layers, self.mass_vector.shape[0]))
        total_growth = np.zeros(self.mass_vector.shape[0])

        for j in range(self.mass_vector.shape[0]):
            total_growth[j] = self.one_actor_growth(strategies, j)

        return total_growth*self.populations



    def strategy_replacer(self, x, i, strategies):
        strat = np.copy(strategies)
        strat[i:i+self.layers] = x

        #print(i)
        return strat

    def one_actor_growth_num_derr(self, strategies, i, fineness = 0.000005):
        return jacobian_calculator(lambda x: self.one_actor_growth(self.strategy_replacer(x, i, strategies), i), strategies[i:i+self.layers], fineness)


    def loss_function(self, strategies):
        total_loss = np.zeros([(self.layers + 1)*self.mass_vector.shape[0]])
       # print(total_loss.shape)
        v = 0
        for i in range(self.mass_vector.shape[0]):

            total_loss[v:v+self.layers] = self.one_actor_growth_num_derr(strategies, i)[0]
            #print(strategies[i*self.layers:(i+1)*self.layers].shape, self.layers, i, i*layers, (i+1)*layers, i+1)
            total_loss[v+1] = np.dot(self.ones,np.dot(self.spectral.M, strategies[i*self.layers:(i+1)*self.layers]))-1
            v += self.layers+1

        #total_loss[-1] = np.sum(np.abs(strategies) - strategies)
        return total_loss

    def strategy_setter(self, strat_vec):
        s_m = np.reshape(strat_vec, self.strategy_matrix.shape)
        self.strategy_matrix = s_m

    def population_setter(self, new_pop):
        self.populations = new_pop #self.total_growth()*time_step

    def consume_resources(self, time_step):

        consumed_resources = np.zeros(self.layers)
        for i in range(len(self.mass_vector)):
            strat_mat = self.strategy_matrix
            interaction_term = self.parameters.who_eats_who[i] * self.populations
            interaction_term = interaction_term * self.parameters.clearance_rate[i]
            layer_action = np.zeros((self.layers, self.mass_vector.shape[0]))
            #        print(self.parameters.layered_attack.shape, strat_mat.shape, self.parameters.handling_times.shape, self.strategy_matrix.shape)
            for j in range(self.layers):
                layer_action[j] = self.parameters.layered_attack[j, i] * strat_mat[i, j] * interaction_term * strat_mat[:,j] \
                                  * self.parameters.handling_times[i]
            # print("This is where I die ", self.layers, i, j, self.parameters.layered_attack.shape, self.parameters.handling_times.shape, strat_mat.shape)

            foraging_term = self.water.res_counts*self.parameters.forager_or_not[i] * self.parameters.handling_times[i]\
                            *self.parameters.clearance_rate[i] * self.parameters.layered_foraging[:, i]

            #print(foraging_term.shape, self.layers, self.spectral.x, self.water.res_counts.shape, self.parameters.layered_foraging.shape)
            #print(np.sum(np.dot(self.ones, np.dot(self.spectral.M, (layer_action + foraging_term)))))
            consumed_resources += foraging_term /(1 + np.sum(np.dot(self.ones, np.dot(self.spectral.M, (np.sum(layer_action, axis = 1) + foraging_term)))))

        self.water.resource_setter(self.water.res_counts - time_step *consumed_resources)

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
        old_vm = np.linalg.inv((old_spec.vandermonde_calculator()))

        #print(small_vec)
        return(np.dot(new_vm.T, np.dot(old_vm.T, small_vec)))

class ecosystem_parameters:
    def __init__(self, mass_vector, spectral, attack_matrix = None, handling_times = None, who_eats_who = None):
        self.mass_vector = mass_vector
        self.spectral = spectral
        self.attack_matrix = self.attack_matrix_setter()
        self.handling_times = self.handling_times_setter()
        self.who_eats_who = self.who_eats_who_setter()
        self.clearance_rate = self.clearance_rate_setter() #330/12 * mass_vector**(3/4)
        self.layered_attack = self.layer_creator(self.attack_matrix)
        self.efficiency = 0.7 #Very good number.
        self.loss_term = self.loss_rate_setter() #mass_vector**(0.75)*0.01
        self.forager_or_not = self.forager_or_not_setter()
        self.foraging_attack_prob = self.foraging_attack_setter()
        self.layered_foraging = self.layer_creator(self.foraging_attack_prob)



    def forager_or_not_setter(self):
        forage_mass = 0.05 #Should probably be in the ecosystem parameters explicitly
        fo_or_not = (np.copy(self.mass_vector))/forage_mass
        fo_or_not[fo_or_not > 1000] = 0
        fo_or_not[fo_or_not != 0] = 1

        return fo_or_not

    def foraging_attack_setter(self):
        forage_mass = 0.05
        sigma = 1.3
        beta = 408

        foraging_attack = np.exp(-(np.log(self.mass_vector/(beta*forage_mass))) ** 2 / (2 * sigma ** 2))

        return foraging_attack

    def attack_matrix_setter(self):
        sigma = 1.3
        beta = 408

        att_m = self.mass_vector*1/(beta*self.mass_vector.reshape((self.mass_vector.shape[0],1)))

        att_m = np.exp(-(np.log(att_m))**2/(2*sigma**2))

        return att_m


    def who_eats_who_setter(self):
        who_eats_who = self.mass_vector * 1/self.mass_vector.reshape((self.mass_vector.shape[0], 1))
        who_eats_who[who_eats_who < 10] = 0
        who_eats_who[who_eats_who != 0] = 1

        return who_eats_who

    def handling_times_setter(self):
        return 1/(22.3*self.mass_vector**(0.75))

    def loss_rate_setter(self):
        return 1.2*self.mass_vector**(0.75)

    def layer_creator(self, obj):
        weights = 2/(1+np.exp(self.spectral.x)) #Replace with actual function
        layers = np.zeros((self.spectral.x.shape[0], *obj.shape))
       # print(layers.shape, obj.shape, self.spectral.x.shape[0])
        for i in range(self.spectral.x.shape[0]):
            layers[i] = weights[i] * obj
        return layers

    def clearance_rate_setter(self):
        return 330*self.mass_vector**(0.75) #This should probably be more sophisticated... But a good first approximation




class water_column:
    def __init__(self, spectral, res_vec, advection = 1, diffusion = 2, resource_max = 15, replacement = 1.2, time_step = 0.001, layers = 2):
        self.adv = advection
        self.diff = diffusion
        self.resource = resource_max
        self.lam = replacement
        self.time_step = time_step
        self.spectral = spectral
        self.res_counts = res_vec
        self.layers = layers
        self.res_top = np.zeros(layers)

    def resource_setter(self, new_res):
        self.res_counts = new_res

    def update_resources(self):
        ##Chemostat step
        ones = np.repeat(1, self.layers)
        total_top_mass = np.dot(ones, np.dot(self.spectral.M, self.res_top))
        self.res_counts[0] += self.lam*(self.resource - total_top_mass)*self.time_step

        ##Advection diffusion step
        if self.diff != 0 or self.adv != 0:
            diff_op = -self.spectral.D*self.adv+self.diff*np.linalg.matrix_power(self.spectral.D,2)+np.identity(self.layers)/self.time_step
            diff_op[0] = self.spectral.D[0]
            diff_op[-1] = 0
            diff_op[-1,-1] = 1
            self.res_counts = np.linalg.solve(diff_op, self.res_counts)



def interpolater(old_vec, old_size, new_size, size_classes, old_spec, new_spec):
    new_vec = np.zeros(new_size*size_classes)
    for k in range(size_classes):

        new_strat = new_spec.expander(old_spec = old_spec, small_vec =old_vec[old_size*k:old_size*(k+1)])
        new_strat[new_strat < 0] = 0
        new_vec[new_size*k:new_size*(k+1)] = new_strat/np.dot(np.repeat(1, new_size),np.dot(new_spec.M, new_strat))

    return new_vec


depth = 20 #Previously 5 has worked well.
layers = 10 #5 works well.
log_size_range = 5 #8 NOT worked well.
size_classes = 4

obj = spectral_method(depth, layers-1) #This is the old off-by-one error... Now we have added another fucked up error!
mass_vector = np.array([1, 300, 300**2, 10*300**2]) #np.exp(np.linspace(0,log_size_range,size_classes)) #
print(mass_vector)
res_start = 4*1/(1+np.exp(np.linspace(0, depth, layers))) #Previously used the constant 1. Now\re fully loaded!
water_start = water_column(obj, res_start, layers = layers)

params = ecosystem_parameters(mass_vector, obj)

eco = ecosystem_optimization(mass_vector, layers, params, obj, water_start)

#print(params.layered_foraging.shape)
x_res = optm.least_squares(eco.loss_function, x0=eco.strategy_matrix.flatten(), bounds = (0, np.inf), loss = 'cauchy').x
#print(x_res)
for k in range(size_classes):
    print(np.dot(np.repeat(1, layers),np.dot(obj.M,x_res[layers*k:layers*(k+1)])))


new_size = 40 #Off-by-one
new_spectral = spectral_method(depth, new_size - 1)
res_start2 = 4*1/(1+np.exp(np.linspace(0, depth, new_size)))
water_start2 = water_column(new_spectral, res_start2, layers = new_size)
params2 = ecosystem_parameters(mass_vector, new_spectral)

seed_vec2 = interpolater(x_res, layers, new_size, size_classes, obj, new_spectral)
print(seed_vec2)
eco2 = ecosystem_optimization(mass_vector, new_size, params2, new_spectral, water_start2)
print(eco2.layers, "Eco2shape")

eco2.strategy_setter(seed_vec2)
print(eco2.strategy_matrix.flatten().shape)
print(np.dot(np.repeat(1, new_size),np.dot(new_spectral.M,np.repeat(1, new_size))), "What, why no 5")

x_res2 = optm.least_squares(eco2.loss_function, x0=eco2.strategy_matrix.flatten(), tr_solver='lsmr', bounds = (0, np.inf)).x

for k in range(size_classes):
    print(np.dot(np.repeat(1, new_size),np.dot(new_spectral.M, x_res2[new_size*k:new_size*(k+1)])))

plt.figure()
plt.plot(obj.x, x_res[0:layers])
plt.show()

plt.figure()

for i in range(size_classes):
    plt.plot(new_spectral.x, x_res2[i*new_size:(i+1)*new_size])

plt.show()

#new_strat = new_strat/np.dot(np.repeat(1, 10),np.dot(new_spectral.M, new_strat))




eco.strategy_setter(x_res)
t_end = 2
time_step = 0.001

simulate = False
if simulate is True:
    time_span = np.linspace(0, t_end, int(t_end/time_step))

    for t in time_span:
        eco.population_setter(eco.total_growth()*time_step + eco.populations)
        eco.consume_resources(time_step)
        eco.water.update_resources()
        x_res = optm.newton_krylov(eco.loss_function, x0=np.reshape(eco.strategy_matrix, layers*size_classes)).x #bounds=([0, np.inf]))
        eco.strategy_setter(x_res)


    plt.figure()
    plt.plot(obj.x, x_res[0:layers])
    plt.show()