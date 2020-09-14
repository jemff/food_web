depth = 20 #Previously 5 has worked well.
layers = 50 #5 works well.
size_classes = 2
lam = 2
time_step = 0.0001
simulate = False
verbose = True
l2 = False



from size_based_ecosystem import *
import pickle as pkl


mass_vector = np.array([20, 8000]) #np.array([1, 30, 300, 400, 800, 16000])
from scipy import stats
obj = spectral_method(depth, layers-1) #This is the old off-by-one error... Now we have added another fucked up error!
logn = stats.lognorm.pdf(obj.x, 1, 0)

norm_dist = stats.norm.pdf(obj.x, loc = 2)
res_start = 3*norm_dist #0.1*(1-obj.x/depth)
res_max = 10*norm_dist

water_start = water_column(obj, res_start, layers = layers, resource_max = res_max, replacement = lam, advection = 0, diffusion = 0)

params = ecosystem_parameters(mass_vector, obj)
#params.handling_times = np.zeros(2)

eco = ecosystem_optimization(mass_vector, layers, params, obj, water_start, l2 = l2, movement_cost=0)
eco.population_setter(np.array([1, 0.0000001]) )
OG_layered_attack = np.copy(eco.parameters.layered_attack)
time_step = 10**(-4)

error = 1
strategies = []
population_list = []
time = 0

while time<0.2:
    x_res = sequential_nash(eco, verbose=verbose, l2=l2)
    strategies.append(x_res)
    pop_old = np.copy(eco.populations)
    delta_pop = eco.total_growth(x_res)
    new_pop = delta_pop * time_step + eco.populations
    population_list.append(new_pop)
    error = np.linalg.norm(new_pop - pop_old)

    eco.population_setter(eco.total_growth(x_res) * time_step + eco.populations)
    eco.strategy_setter(x_res)
    eco.water.update_resources(consumed_resources=eco.consumed_resources(), time_step=time_step)
    print("I'm here")
    print(error, eco.populations, np.sum(eco.water.res_counts), time_step, new_pop - pop_old, time)
    time += time_step
    eco.parameters.layered_attack = 1/2*(1+np.cos(time*365*2*np.pi))*OG_layered_attack

with open('eco_test.pkl', 'wb') as f:
    pkl.dump(eco, f, pkl.HIGHEST_PROTOCOL)

with open('strategies_eco_test.pkl', 'wb') as f:
    pkl.dump(strategies, f, pkl.HIGHEST_PROTOCOL)

with open('population_eco_test.pkl', 'wb') as f:
    pkl.dump(population_list, f, pkl.HIGHEST_PROTOCOL)