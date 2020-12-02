import matplotlib.pyplot as plt
depth = 45   #Previously 5 has worked well.
layers = 60 #5 works well.
segments = 1
lam = 2
simulate = False
verbose = True
l2 = False



from  utility_functions import *
import scipy.special as special



mass_vector = np.array([0.05, 20, 400, 8000]) #np.array([1, 30, 300, 400, 800, 16000])


from scipy import stats
obj = spectral_method(depth, layers) #This is the old off-by-one error... Now we have added another fucked up error!
logn = stats.lognorm.pdf(obj.x, 1, 0)

obj = spectral_method(depth, layers, segments = segments)
logn = stats.lognorm.pdf(obj.x, 1, 0)

from time import perf_counter
norm_dist = stats.norm.pdf(obj.x, loc = 6, scale = 6) #+ 0.1*stats.norm.pdf(obj.x, loc = depth-6, scale = 6)
res_start = norm_dist #0.1*(1-obj.x/depth)
res_max = 10*norm_dist

def depth_dependent_clearance(I, k, c, swim_speed, min_vis):
    """ k specifies the attenueation rate, c is the light detection threshold, I specifies the light level"""
    D = 2*special.lambertw(1/2*k**2*np.sqrt((c*I+c*k)/(I*k**2))*(1/c - k/(c*(I+k))))
    beta = swim_speed*D**2+swim_speed*min_vis**2

    return beta

def layer_creator_better(attack_matrix, I, K, c, clearance_rates, min_attack):
    pass


water_start = water_column(obj, res_start, layers = layers*segments, resource_max = res_max, replacement = lam, advection = 0, diffusion = 0)

params = ecosystem_parameters(mass_vector, obj, lam=0.2, forage_mass=0.05/408)
params.handling_times = np.zeros(len(mass_vector))
params.clearance_rate = params.clearance_rate/(24*365)
params.layered_attack = new_layer_attack(params, 1, k = 0.2, beta_0 = 10**(-3))
eco = ecosystem_optimization(mass_vector, layers, params, obj, water_start, l2 = l2, movement_cost=0)
eco.population_setter(np.array([10, 1, 0.1, 0.01]) )
#eco.heat_kernel_creator(1, k = 1)
print(params.who_eats_who)
print(params.forager_or_not)
print(eco.heat_kernels[0].shape)
eco.dirac_delta_creator()
# Start the stopwatch / counter


#t1_start = perf_counter()
#SOL1 = lemke_optimizer(eco, payoff_matrix=total_payoff_matrix_builder(eco))
#t1_stop = perf_counter()

t2_start = perf_counter()
SOL2 = lemke_optimizer(eco, payoff_matrix=total_payoff_matrix_builder_sparse(eco))
t2_stop = perf_counter()

#t3_start = perf_counter()
#SOL3 = quadratic_optimizer(eco, payoff_matrix=total_payoff_matrix_builder_sparse(eco))
#t3_stop = perf_counter()

#t4_start = perf_counter()
#SOL4 = quadratic_optimizer(eco, payoff_matrix=total_payoff_matrix_builder(eco))
#t4_stop = perf_counter()

#print(t1_stop-t1_start, t2_stop-t2_start, t3_stop-t3_start, t4_stop-t4_start)

#for i in range(mass_vector.shape[0]):
#    plt.plot(obj.x, SOL[i*layers:(i+1)*layers]@eco.heat_kernels[0])
#plt.show()

for i in range(mass_vector.shape[0]):
    plt.plot(obj.x, SOL2[i*layers:(i+1)*layers]@eco.heat_kernels[0])
plt.show()
#simulator_new(eco, "4_species", end_date='2014-04-02', population_dynamics=False, k=0.3, sparse=True, lemke=True)