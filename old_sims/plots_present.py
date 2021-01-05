from size_based_ecosystem import *
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size = 10)

depth = 10
layers = 15
segments = 1
obj = spectral_method(depth, layers, segments=segments)
plt.figure(figsize=(2.5, 5))
plt.scatter(y=obj.x, x=0*obj.x, marker='x', linewidths=0.001, alpha = 0.6, c = 'b')
plt.scatter(y=np.linspace(0,10,layers), x =0*obj.x, marker = 'v', linewidths=0.001, alpha = 0.6, c='r')
plt.savefig("Discretization.pdf")

hat_10 = np.zeros(layers)
hat_10[9] = 1
plt.figure(figsize=(2.5,2.5))
plt.plot(obj.x, hat_10, marker='x',  alpha = 0.6, c = 'b', label = '$h_{10}$')
plt.savefig("Hat_function.pdf")

hat_mix = np.zeros(layers)
hat_mix[9] = 1/3
hat_mix[5] = 1/3
hat_mix[6] = 1/3

plt.figure(figsize=(2.5,2.5))
plt.plot(obj.x, hat_mix, marker='x',  alpha = 0.6, c = 'b')
plt.savefig("Hat_function_mix.pdf")



obj = spectral_method(depth, layers*10, segments=segments)


plt.figure(figsize=(2.5,2.5))
plt.plot(obj.x, np.exp(-(1-obj.x)**2) + np.exp(-(2*10-1-obj.x)**2) + np.exp(-(-1-obj.x)**2), alpha = 0.6, c = 'b')
plt.savefig("Method_of_images_x1.pdf")


plt.figure(figsize=(2.5,2.5))
plt.plot(obj.x, np.exp(-(9-obj.x)**2) + np.exp(-(2*10-9-obj.x)**2) + np.exp(-(-9-obj.x)**2),  alpha = 0.6, c = 'b')
plt.savefig("Method_of_images_x9.pdf")
