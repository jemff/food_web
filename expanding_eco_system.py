from size_based_ecosystem import *
import copy as copy
import scipy.stats as stats
import pickle as pkl

depth = 20 #Previously 5 has worked well.
layers = 30 #5 works well.
log_size_range = 12 # 9.5 #8 did NOT work well.
size_classes = 1
t_end = 2/730 #30/365
lam = 1
time_step = 10**(-5) #1/500000
res_max = 15
simulate = True
verbose = False
daily_cycle = 365*2*np.pi


obj = spectral_method(depth, layers-1) #This is the old off-by-one error... Now we have added another fucked up error!
norm_dist = stats.norm.pdf(obj.x, loc = 2)
res_start = 5*norm_dist #0.1*(1-obj.x/depth)


water_start = water_column(obj, res_start, layers = layers, resource_max = res_max, time_step = time_step, replacement = lam, advection = 0.01, diffusion = 0)
list_of_sizes = np.array([1, 20, 400, 1600, 8000, 40000])


size_classes = 1
m_v_t = np.array([list_of_sizes[size_classes - 1]])
params = ecosystem_parameters(m_v_t, obj)
eco = ecosystem_optimization(m_v_t, layers, params, obj, water_start, loss = 'constr')
#OG_layered_attack = np.copy(eco.parameters.layered_attack)
eco.population_setter(np.array([1]) )#, 1, 1, 1, 0.1]))


frozen_ecos = []

stability = False
time_step = 10**(-4)

if simulate is True:
    while size_classes < len(list_of_sizes):

        x_res = sequential_nash(eco, verbose=verbose)
        eco.strategy_setter(x_res)
        pop_old = np.copy(eco.populations)
        eco.population_setter(eco.total_growth() * time_step + eco.populations)
        error = np.linalg.norm(eco.populations - pop_old)
        print(error)

        if error < 10**(-10):
            stability = True

        if stability is True:
            old_eco = copy.deepcopy(eco)
            strat_old = np.copy(eco.strategy_matrix)

            print("New regime")
            frozen_ecos.append(old_eco)
            pops = np.copy(old_eco.populations)
            size_classes += 1
            m_v_t = copy.deepcopy(size_classes[0:size_classes])

            params = ecosystem_parameters(m_v_t, obj)
            eco = ecosystem_optimization(m_v_t, layers, params, obj, water_start, loss='constr')
            eco.strategy_matrix[0:size_classes-1] = strat_old
            eco.populations[0:size_classes-1] = pops
            eco.populations[-1] = 10**(-10)

            stability = False

    with open('eco_systems.pkl', 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pkl.dump(frozen_ecos, f, pkl.HIGHEST_PROTOCOL)
