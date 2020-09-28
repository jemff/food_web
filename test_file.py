depth = 100 #Previously 5 has worked well.
layers = 5 #5 works well.
segments = 50
size_classes = 2
lam = 2
simulate = False
verbose = True
l2 = False



from size_based_ecosystem import *
import pickle as pkl


mass_vector = np.array([20, 8000]) #np.array([1, 30, 300, 400, 800, 16000])
from scipy import stats
obj = spectral_method(depth, layers, segments = segments)
logn = stats.lognorm.pdf(obj.x, 1, 0)

norm_dist = stats.norm.pdf(obj.x, loc = 6, scale = 6)
res_start = norm_dist #0.1*(1-obj.x/depth)
res_max = 10*norm_dist

water_start = water_column(obj, res_start, layers = layers*segments, resource_max = res_max, replacement = lam, advection = 0, diffusion = 0)

params = ecosystem_parameters(mass_vector, obj, lam=0.4)
params.handling_times = np.zeros(2)

eco = ecosystem_optimization(mass_vector, layers*segments, params, obj, water_start, l2 = l2, movement_cost=0)
eco.population_setter(np.array([2, 0.1]) )
OG_layered_attack = np.copy(eco.parameters.layered_attack)
time_step = 1/48*1/365 #Time-step is half an hour.
eco.heat_kernels[1] = eco.heat_kernels[0]
error = 1
strategies = []
population_list = []
resource_list = []
time = 0
prior_sol = quadratic_optimizer(eco)

print(prior_sol)

plt.plot(eco.spectral.x, prior_sol[0:layers*segments]@eco.heat_kernels[0])
plt.show()


#plt.plot(eco.spectral.x, prior_sol[layers*segments:2*(layers*segments)]@eco.heat_kernels[1])
#plt.show()