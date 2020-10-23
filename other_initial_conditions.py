import matplotlib.pyplot as plt
from utility_functions import *


depth = 45
layers = 80
segments = 1
size_classes = 2
lam = 300
simulate = False
verbose = True
l2 = False
min_attack_rate = 10**(-3)
mass_vector = np.array([0.05, 20, 2000])  # np.array([1, 30, 300, 400, 800, 16000])

obj = spectral_method(depth, layers, segments=segments)
logn = stats.lognorm.pdf(obj.x, 1, 0)

norm_dist = stats.norm.pdf(obj.x, loc=0, scale=3)
res_start = 8*norm_dist  # 0.1*(1-obj.x/depth)
res_max = 10 * norm_dist

water_start = water_column(obj, res_start, layers=layers * segments, resource_max=res_max, replacement=lam, advection=0,
                           diffusion=0, logistic = True)

params = ecosystem_parameters(mass_vector, obj, lam=0.3, min_attack_rate = min_attack_rate, forage_mass = 0.05/408)
params.handling_times = np.zeros(3)

eco = ecosystem_optimization(mass_vector, layers * segments, params, obj, water_start, l2=l2, movement_cost=0)
eco.population_setter(np.array([1, 0.1, 0.01]))
eco.heat_kernel_creator(10**(-1))
eco.heat_kernels[1] = eco.heat_kernels[0]
eco.heat_kernels[2] = eco.heat_kernels[0]
eco.parameters.who_eats_who[1,0] = 1

opt_sol = lemke_optimizer(eco)
opt_sol_quad_opt = quadratic_optimizer(eco)


plt.plot(obj.x, opt_sol[0:layers]@eco.heat_kernels[0])
plt.plot(obj.x, opt_sol[layers:2*layers]@eco.heat_kernels[0])
plt.plot(obj.x, opt_sol[2*layers:3*layers]@eco.heat_kernels[0])
plt.show()

plt.plot(obj.x, opt_sol_quad_opt[0:layers]@eco.heat_kernels[0])
plt.plot(obj.x, opt_sol_quad_opt[layers:2*layers]@eco.heat_kernels[0])
plt.plot(obj.x, opt_sol_quad_opt[2*layers:3*layers]@eco.heat_kernels[0])
plt.show()

simulator(eco, params, "proper_tritrophic", total_days=180, lemke = False)