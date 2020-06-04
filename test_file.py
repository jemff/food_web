depth = 20 #Previously 5 has worked well.
layers = 120 #5 works well.
log_size_range = 12 # 9.5 #8 did NOT work well.
size_classes = 2
t_end = 1
lam = 8
time_step = 0.0001
res_max = 15
simulate = False
verbose = False

from size_based_ecosystem import *

mass_vector = np.array([1, 20, 400, 8000, 16000]) #np.array([1, 30, 300, 400, 800, 16000])
from scipy import stats
obj = spectral_method(depth, layers-1) #This is the old off-by-one error... Now we have added another fucked up error!
logn = stats.lognorm.pdf(obj.x, 1, 0)
print(logn, "Logn")

res_start = logn


water_start = water_column(obj, res_start, layers = layers, resource_max = res_max, time_step = time_step, replacement = lam, advection = 1, diffusion = 0)

params = ecosystem_parameters(mass_vector, obj)
eco = ecosystem_optimization(mass_vector, layers, params, obj, water_start, loss = 'constr')
eco.population_setter(np.array([1, 1, 1, 1, 0.1]) )#, 1, 1, 1, 0.1]))
#eco.strategy_setter(np.sqrt(eco.strategy_matrix.flatten())) THis is for the L2 version... Quantum fish ahoy
print(graph_builder(eco), "Original graph")
seq_nash = sequential_nash(eco, verbose = True)
eco.strategy_setter(seq_nash)
print(graph_builder(eco), "New graph")

plt.figure()
for i in range(size_classes):
    plt.plot(obj.x, seq_nash[i * layers:(i + 1) * layers], label='Creature ' + str(i))
    plt.legend(loc='upper right')
    print(eco.one_actor_growth(eco.strategy_matrix.flatten(), i))
plt.show()
#print(eco.one_actor_growth(eco.strategy_matrix.flatten(), 1))
#print(eco.strategy_matrix[0,:], eco.strategy_matrix[1,:])
#print(eco.one_actor_growth(eco.strategy_replacer(eco.strategy_matrix[0,:], 1, eco.strategy_matrix.flatten()), 1), "What if we moved to the prey??")

#strat_mat_unif = np.repeat(depth/layers, layers)
#seq_nash[0:layers] = strat_mat_unif
#eco.strategy_setter(seq_nash)

#for i in range(59):
#    eco.population_setter(np.array([1+i*0.1, 1, 1, 1, 1, 0.1]))
#    print(i)
#    seq_nash = sequential_nash(eco, verbose=True)
#    eco.strategy_setter(seq_nash)

#for i in range(100):
#    eco.population_setter(np.array([6.8+i*0.01, 1, 1, 1+0.01*i, 1, 0.1]))
#    print(i, "Second refinement")
#    seq_nash = sequential_nash(eco, verbose=True)
#    eco.strategy_setter(seq_nash)

plt.figure()
for i in range(size_classes):
    plt.plot(obj.x, seq_nash[i * layers:(i + 1) * layers], label='Creature ' + str(i))
    plt.legend(loc='upper right')
    print(eco.one_actor_growth(eco.strategy_matrix.flatten(), i))
plt.show()

time_range = 0

for i in range(time_range):
    eco.population_setter(eco.total_growth() * 1/time_range + eco.populations)
    seq_nash = sequential_nash(eco, verbose=True)
    eco.strategy_setter(seq_nash)
    print(eco.populations)


plt.figure()
for i in range(size_classes):
    plt.plot(obj.x, seq_nash[i * layers:(i + 1) * layers], label='Creature ' + str(i))
    plt.legend(loc='upper right')
    print(eco.one_actor_growth(eco.strategy_matrix.flatten(), i))
plt.show()


