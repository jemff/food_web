import scipy.stats as stats
import numpy as np
import scipy.optimize as optm



class ecosystem_palow_proew:
    def __init__(self, X0 = 1, D = 2, no_specs = 4):
        self.X0 = X0
        self.dimensions = D
        self.no_specs = no_specs
        self.pd = 0.2
        self.pv = 0.25
        self.Rp = 0.1 #
        self.h0 = 10**(2) #kg**(beta-1) s
        self.L = 1 #CHECK PETCHEY!
        self.beta = 0.75
        self.Cbar0 = 100 #kg**(beta) m**(-3)
        self.lam = 1.71 * 10**(-6) #kg**(1-b)s**(-1)
        self.efficiency = 0.7 #Arbitrary
        self.loss_rate = 4.15*10**(-8) #Remark this is in seconds... Multiply by 3600*24*30 to get to a month
        self.attack_probability = np.zeros((no_specs, no_specs))
        self.search_rate = np.zeros((no_specs, no_specs))
        self.handling_time = np.zeros((no_specs, no_specs))
        self.who_eats_who = np.zeros((no_specs, no_specs))
        self.current_stategy = np.zeros((no_specs, no_specs))

    def set_mass(self, distribution, n):
        self.mass_vector = distribution(size = n)

    def set_search_rate(self):
        self.starting_numbers = self.X0*self.mass_vector
        if self.dimensions == 2:
            self.search_rate = \
                2*self.V0*self.d0*(self.mass_vector)**(self.pv+self.pd-1).reshape((self.no_specs,1))\
                *self.mass_vector**self.pd
        elif self.dimensions == 3:
            self.search_rate = \
                np.pi * self.V0 * self.d0**2 \
                * (self.mass_vector) ** (self.pv + 2*self.pd - 1).reshape((self.no_specs, 1))\
                *self.mass_vector ** (2*self.pd)
    def set_attack_probability(self):
        column_mass = np.reshape(self.mass_vector, (self.no_specs, 1))

        inner_term = self.Rp*(column_mass*1/self.mass_vector)
        second_term = (1+(np.log10(inner_term))**2)**(-0.2)
        outer_term = 1/(1+0.25*np.exp(-column_mass**(0.33)))

        self.attack_probability = outer_term*second_term

    def set_handling_time(self):


        column_mass = np.reshape(self.mass_vector, (self.no_specs, 1))
        if self.handling_type == "petchey":

            self_eat = np.hstack([column_mass**2]*self.no_specs)

            handling_time = self.h0*column_mass**2*(self.L*column_mass*self.mass_vector - self_eat)

            handling_time[handling_time < 0] = 0 #np.nan, infty in model. This also gives who eats who.
            self.handling_time = handling_time
            self.who_eats_who = handling_time
            self.who_eats_who[handling_time > 0] = 1
        else:
            self.handling_time = self.h0*self.mass_vector**(1-self.beta) #This is too simplistic..? Use gauss-curve to reflect preferred food-size

    def set_basal_metabolic_rate(self):
        self.ci = self.loss_rate*self.mass_vector**(self.beta-1)


    def flow_matrix(self):
        interaction_term = self.who_eats_who*self.search_rate * self.attack_probability


    def growth(self,j):
        interaction_term = self.who_eats_who*self.search_rate * self.attack_probability
        denom = 1+np.dot(self.handling_time*interaction_term[:,j],self.X0)
        enumerator = self.efficiency*np.dot(interaction_term[:,j], self.X0)

        return denom/enumerator

    def loss(self, j):
        #Define loss functions, potentially based on habitat risk?

    def habitat_specific_probability(self):
        #Define as simply constants times attack probability (So it becomes a tensor!)
        #Start by getting the code to work without this feature.
        #Define 5 (or n) habitats ranging from risky to not so risky
    def optimal_strategy(self):

        #Use least squares ond erivatives of all pr-capita fitness functions to find optima.

#Implement pymoo to find the optima.



