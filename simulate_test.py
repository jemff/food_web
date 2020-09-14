from size_based_ecosystem import *
import matplotlib.animation as animation
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
for i in range(len(tableau20)):
    r, g, b = tableau20[i]
    tableau20[i] = (r / 255., g / 255., b / 255.)

from size_based_ecosystem import *
import copy as copy
import scipy.stats as stats
import pickle as pkl

depth = 10
layers = 100
size_classes = 1
lam = 1
simulate = True
verbose = False
daily_cycle = 365*2*np.pi


obj = spectral_method(depth, layers-1) #This is the old off-by-one error... Now we have added another fucked up error!
#norm_dist = stats.norm.pdf(obj.x, loc = 3, scale = 3)
#print(norm_dist)
norm_dist = stats.norm.pdf(obj.x, loc = 3)
res_start = 3*norm_dist #0.1*(1-obj.x/depth)
res_max = 10*norm_dist

water_start = water_column(obj, res_start, layers = layers, resource_max = res_max, replacement = lam, advection = 0, diffusion = 0)
list_of_sizes = np.array([1, 20, 8000]) #, 1, 400, 1600, 40000])

l2 = False
size_classes = 2
m_v_t = list_of_sizes #list_of_sizes[0:size_classes]
params = ecosystem_parameters(m_v_t, obj)
eco = ecosystem_optimization(m_v_t, layers, params, obj, water_start, l2 = l2, output_level = 5, movement_cost = 0)
#OG_layered_attack = np.copy(eco.parameters.layered_attack)
eco.population_setter(np.array([0.1, 20, 0.1]) )#, 1, 1, 1, 0.1]))
eco.parameters.handling_times = 0 * eco.parameters.handling_times
OG_layered_attack = np.copy(params.layered_attack)
frozen_ecos = []

stability = False
time_step = 10**(-4)
#max_err = time_step*1/10
x_res = sequential_nash(eco, verbose=True, l2=l2, max_its_seq = 20)
for i in range(3):
    plt.plot(obj.x, x_res[i]@eco.heat_kernels[i])
plt.show()

for i in range(100):
    eco.population_setter(np.array([0.1*(i+2), 20, 0.1]) )#, 1, 1, 1, 0.1]))
    x_res = sequential_nash(eco, verbose=True, l2=l2, max_its_seq=40)
    eco.strategy_setter(x_res)
    print("Current iteration", i)


for i in range(3):
    plt.plot(obj.x, x_res[i]@eco.heat_kernels[i])

plt.show()

with open('simulate_test.pkl', 'wb') as f:
    pkl.dump(eco, f, pkl.HIGHEST_PROTOCOL)