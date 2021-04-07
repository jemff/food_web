from food_web_core.utility_functions import *
from food_web_core.size_based_ecosystem import *
import matplotlib.pyplot as plt
lam = 1
l2 = False
layers = 60
segments = 1
depth = 45
pop_max = 100
pop_min = 0
carrying_capacity = 50
fidelity = 50
ppop_max = 5
pop_varc = np.linspace(pop_min, pop_max, fidelity)
pop_varp = np.linspace(pop_min, ppop_max, fidelity)

mass_vector = np.array([1, 408])  # np.array([1, 30, 300, 400, 800, 16000])
min_attack_rate = 5*10**(-3)

obj = spectral_method(depth, layers, segments=segments)
logn = stats.lognorm.pdf(obj.x, 1, 0)

norm_dist = stats.norm.pdf(obj.x, loc=0, scale=3)
res_start = 8*norm_dist  #not used
res_max = 10*norm_dist

water_start = water_column(obj, res_start, layers=layers * segments, resource_max=res_max, replacement=lam, advection=0,
                           diffusion=0, logistic = True)

params = ecosystem_parameters(mass_vector, obj, lam=0.3, min_attack_rate = min_attack_rate, forage_mass = 1/408)
params.handling_times = np.zeros(2)

eco = ecosystem_optimization(mass_vector, layers * segments, params, obj, water_start, l2=l2, movement_cost=0)
eco.population_setter(np.array([1, 1]))
eco.parameters.layered_foraging[:,0] = 100*np.tanh(obj.x[::-1])

eco.parameters.layered_attack = new_layer_attack(eco.parameters, 1, beta_0=0.001, k = 10**(-3))
eco.dirac_delta_creator()

def lotka_volterra_forager(populations, eco, carrying_capacity=1):
    foraging_gain = np.zeros((eco.populations.size * eco.layers, eco.populations.size * eco.layers))

    foraging_gain_t = eco.parameters.clearance_rate[0] * (np.ones(eco.layers) @ eco.heat_kernels[0] * eco.parameters.layered_foraging[:, 0])*(1-populations[0]/carrying_capacity)

    foraging_gain[0:eco.layers,1 * eco.layers: (1 + 1) * eco.layers] = foraging_gain_t

    return foraging_gain

test1 = np.zeros(60)
test2 = np.zeros(60)

test1[0] = 1
test2[3] = 1

#t1 = test1 @ eco.heat_kernels[0]
#t2 = test2 @ eco.heat_kernels[1]

#t3 = np.sin(obj.x)
#t4 = np.cos(obj.x)
#plt.plot(obj.x, t3)
#plt.plot(obj.x, t4)
#plt.show()
#print(eco.heat_kernels)

total_reward_matrix, total_loss_matrix = loss_and_reward_builder(eco)
gridx, gridy= np.meshgrid(pop_varc, pop_varp)
z1 = np.zeros(120)
z2 = np.zeros(120)
vectors = np.zeros((fidelity, fidelity, 2))
print(eco.parameters.loss_term)
#print(total_loss_matrix)
#total_payoff_matrix_builder_memory_improved(eco, populations, total_reward_matrix, total_loss_matrix, foraging_gain)
print(total_reward_matrix[60:,0:60])
for i in range(fidelity):
    foraging_gain = lotka_volterra_forager(np.array([pop_varc[i], pop_varp[0]]), eco, carrying_capacity=carrying_capacity)
    for j in range(fidelity):
        payoff_matrix = total_payoff_matrix_builder_memory_improved(eco, np.array([pop_varc[i], pop_varp[j]]), total_reward_matrix, total_loss_matrix, foraging_gain)

        z = lemke_optimizer(eco, payoff_matrix)[0:-2]
        g1 = pop_varc[i] * z  @ foraging_gain @ z - pop_varc[i]*pop_varp[j]*z @ total_loss_matrix @ z - pop_varc[i]*eco.parameters.loss_term[0]
        print(z @ total_loss_matrix @ z, g1, pop_varc[i] * z  @ foraging_gain @ z, pop_varc[i]*pop_varp[j])

        g2 = eco.parameters.efficiency * pop_varc[i]*pop_varp[j]*z @ total_reward_matrix @ z - pop_varp[j]*eco.parameters.loss_term[1]
        vectors[i,j] = g1, g2


plt.quiver(gridx, gridy, vectors[:,:,0], vectors[:,:,1], angles = 'xy')

plt.show()