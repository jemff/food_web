import matplotlib.pyplot as plt
depth = 100 #Previously 5 has worked well.
layers = 100 #5 works well.
segments = 1
lam = 2
simulate = False
verbose = True
l2 = False



from  utility_functions import *
import scipy.special as special
import pickle as pkl



mass_vector = np.array([0.05, 20, 400, 8000]) #np.array([1, 30, 300, 400, 800, 16000])


from scipy import stats
obj = spectral_method(depth, layers) #This is the old off-by-one error... Now we have added another fucked up error!
logn = stats.lognorm.pdf(obj.x, 1, 0)

obj = spectral_method(depth, layers, segments = segments)
logn = stats.lognorm.pdf(obj.x, 1, 0)


norm_dist = stats.norm.pdf(obj.x, loc = 6, scale = 6) #+ 0.1*stats.norm.pdf(obj.x, loc = depth-6, scale = 6)
res_start = norm_dist #0.1*(1-obj.x/depth)
res_max = 10*norm_dist

def depth_dependent_clearance(I, swim_speed, min_vis=10**(-2), k=0.1, c=10**(6), K = 10**(3)):
    """ k specifies the attenueation rate, c is the light detection threshold, I specifies the light level"""
    D = np.real(2.0 * special.lambertw(np.sqrt((K + I) * I / c) * k / 2.0) / k)

    beta = swim_speed*D**2+swim_speed*min_vis**2

    return beta

def layer_creator_better(attack_matrix, I, clearance_rates):
    layers = I.shape

    total_layers = np.zeros(layers, *attack_matrix.shape)
    for i, clearance in enumerate(clearance_rates):
        total_layers[:,i,:] = np.kron(clearance,attack_matrix[i,:])
    return total_layers




water_start = water_column(obj, res_start, layers = layers*segments, resource_max = res_max, replacement = lam, advection = 0, diffusion = 0)

params = ecosystem_parameters(mass_vector, obj, lam=0.2, forage_mass=0.05/408)

print(depth_dependent_clearance(np.max(solar_input_calculator(normalized=False))*np.exp(-0.1*obj.x), 8))
params.clearance_rate = params.clearance_rate/(24*365)
print(params.clearance_rate)

params.handling_times = np.zeros(len(mass_vector))
params.layered_attack = new_layer_attack(params, np.max(solar_input_calculator(normalized=False)), k = 0.1, beta_0 = 10**(-2))
eco = ecosystem_optimization(mass_vector, layers, params, obj, water_start, l2 = l2, movement_cost=0)
eco.population_setter(np.array([10, 1, 1, 0.01]) )
eco.heat_kernel_creator(1, k = 1)
print(params.who_eats_who)
print(params.forager_or_not)
#eco.dirac_delta_creator()
print(eco.heat_kernels[0].shape)


SOL = lemke_optimizer(eco, payoff_matrix=total_payoff_matrix_builder_sparse(eco))
for i in range(len(mass_vector)):
    plt.plot(obj.x, SOL[i*layers:(i+1)*layers]@eco.heat_kernels[0])
plt.show()

#simulator_new(eco, "4_species", end_date='2014-04-02', population_dynamics=False, k=0.3, sparse=True, lemke=True)