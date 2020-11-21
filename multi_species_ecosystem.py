import matplotlib.pyplot as plt
depth = 45 #Previously 5 has worked well.
layers = 100 #5 works well.
segments = 1
size_classes = 4
lam = 2
simulate = False
verbose = True
l2 = False



from  utility_functions import *

import pickle as pkl


mass_vector = np.array([0.05, 20, 400, 8000]) #np.array([1, 30, 300, 400, 800, 16000])
from scipy import stats
obj = spectral_method(depth, layers) #This is the old off-by-one error... Now we have added another fucked up error!
logn = stats.lognorm.pdf(obj.x, 1, 0)

obj = spectral_method(depth, layers, segments = segments)
logn = stats.lognorm.pdf(obj.x, 1, 0)

norm_dist = stats.norm.pdf(obj.x, loc = 6, scale = 6)
res_start = 8*norm_dist #0.1*(1-obj.x/depth)
res_max = 10*norm_dist

water_start = water_column(obj, res_start, layers = layers*segments, resource_max = res_max, replacement = lam, advection = 0, diffusion = 0)

params = ecosystem_parameters(mass_vector, obj, lam=0.2, forage_mass=0.05/408)
params.handling_times = np.zeros(len(mass_vector))
params.clearance_rate = params.clearance_rate/(24*365)
eco = ecosystem_optimization(mass_vector, layers, params, obj, water_start, l2 = l2, movement_cost=0)
eco.population_setter(np.array([10, 0.1, 0.01, 0.001]) )
print(params.who_eats_who)
eco.dirac_delta_creator()
SOL = lemke_optimizer(eco, payoff_matrix=total_payoff_matrix_builder_sparse(eco))
for i in range(size_classes):
    plt.plot(obj.x, SOL[i*layers:(i+1)*layers]@eco.heat_kernels[0])
plt.show()

simulator_new(eco, "4_species", end_date='2014-04-02', population_dynamics=False, k=0.3, sparse=True, lemke=True)