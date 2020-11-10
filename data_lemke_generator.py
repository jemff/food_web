from utility_functions import *

depth = 45
layers = 60
segments = 1
size_classes = 2
lam = 100
simulate = False
verbose = True
l2 = False
min_attack_rate = 10**(-3)
mass_vector = np.array([0.05, 0.05*408])  # np.array([1, 30, 300, 400, 800, 16000])

obj = spectral_method(depth, layers, segments=segments)
logn = stats.lognorm.pdf(obj.x, 1, 0)

norm_dist = stats.norm.pdf(obj.x, loc=0, scale=3)
res_start = 8*norm_dist  # 0.1*(1-obj.x/depth)
res_max = 10*norm_dist

water_start = water_column(obj, res_start, layers=layers * segments, resource_max=res_max, replacement=lam, advection=0,
                           diffusion=0, logistic = True)

params = ecosystem_parameters(mass_vector, obj, lam=0.3, min_attack_rate = min_attack_rate, forage_mass = 0.05/408)
params.handling_times = np.zeros(2)

eco = ecosystem_optimization(mass_vector, layers * segments, params, obj, water_start, l2=l2, movement_cost=0)
eco.population_setter(np.array([10, 0.1]))

eco.dirac_delta_creator()
simulator_new(eco, "non_random_oresund", k = 0.2, min_attack_rate=min_attack_rate)

res_start = 8*norm_dist  # 0.1*(1-obj.x/depth)
res_max = 10*norm_dist

water_start = water_column(obj, res_start, layers=layers * segments, resource_max=res_max, replacement=lam, advection=0,
                           diffusion=0, logistic = True)

params = ecosystem_parameters(mass_vector, obj, lam=0.3, min_attack_rate = min_attack_rate, forage_mass = 0.05/408)
params.handling_times = np.zeros(2)

eco = ecosystem_optimization(mass_vector, layers * segments, params, obj, water_start, l2=l2, movement_cost=0)

eco.heat_kernel_creator(10**(-1))
eco.heat_kernels[1] = eco.heat_kernels[0]

eco.population_setter(np.array([10, 0.1]))

simulator_new(eco, "more_random_oresund", k = 0.2)

res_start = 8*norm_dist  # 0.1*(1-obj.x/depth)
res_max = 10*norm_dist

water_start = water_column(obj, res_start, layers=layers * segments, resource_max=res_max, replacement=lam, advection=0,
                           diffusion=0, logistic = True)

params = ecosystem_parameters(mass_vector, obj, lam=0.3, min_attack_rate = min_attack_rate, forage_mass = 0.05/408)
params.handling_times = np.zeros(2)

eco = ecosystem_optimization(mass_vector, layers * segments, params, obj, water_start, l2=l2, movement_cost=0)

#eco.heat_kernel_creator(1/8*10**(-1))
eco.heat_kernel_creator(10**(-1))
eco.heat_kernels[1] = eco.heat_kernels[0]

#eco.heat_kernels[1] = eco.heat_kernels[0]

eco.population_setter(np.array([10, 0.1]))

simulator_new(eco, "completely_random_oresund", optimal = False, k = 0.2, min_attack_rate=min_attack_rate)