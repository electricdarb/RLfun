import numpy as np
from math import floor
import time
import matplotlib.pyplot as plt

POSITION_SPACE = (-1.2, .6)
VELO_SPACE = (-.07, .07)
ACTION_SPACE = [-1, 0, 1]
START_RANGE = (-.6, -.4)

"""I am frequently weary of oop because sometimes dev time is needlessly high from my experience, however, it think
it provides a really clean and nice abstraction in this senerio and actually keeps dev time lower. I believe it 
is import to use tools when appropriate, i think oop is here"""

class Q_table():
    def __init__(self, discrete_ps = 50, discrete_vs = 50, discrete_as = 3):
        self.p_len, self.v_len, self.a_len = discrete_ps, discrete_vs, discrete_as
        self.map_p_to_i = lambda p: floor((p - POSITION_SPACE[0])  / (POSITION_SPACE[1] - POSITION_SPACE[0]) * (discrete_ps - 1))
        self.map_v_to_i = lambda v: floor((v - VELO_SPACE[0])  / (VELO_SPACE[1] - VELO_SPACE[0]) * (discrete_vs - 1))
        self.map_a_to_i = lambda a: floor(a + 1)
        self.table = np.zeros((discrete_ps, discrete_vs, discrete_as))
        self.mappings = [self.map_p_to_i, self.map_v_to_i, self.map_a_to_i]

    def get_value(self, p, v, a):
        i, j, k = [f(x) for x, f in zip([p, v, a], self.mappings)] # self.mappings countian the mapping of p, v, a to an index, the zip returns a
        # tuple of a mapping function and a corresponding variable, 
        return self.table[i, j, k]
    
    def get_random_action(self, *args): # args is included to allow p, v variables to be passed to function 
        return np.random.choice(ACTION_SPACE) 

    def get_best_action(self, p, v):
        return np.argmax(self.table[self.map_p_to_i(p), self.map_v_to_i(v), :]) - 1 # will return the action from [-1, 0, 1]

    def update_state(self, p, v, a):
        v += a  * 0.001 - np.cos(3 * p) * 0.0025
        v = np.clip(v, *VELO_SPACE)
        p += v
        if p < POSITION_SPACE[0]:
            p = POSITION_SPACE[0]
            v = 0
        return p, v

    def update_Q(self, reward, lr, discount, p, v, a): 
        """
        this updates the Q value based on the bellman equation: https://en.wikipedia.org/wiki/Bellman_equation
        """
        i, j, k = [f(x) for x, f in zip([p, v, a], self.mappings)]  # self.mappings countian the mapping of p, v, a to an index, the zip returns a
        # tuple of a mapping function and a corresponding variable, 
        Q = self.table[i, j, k] 
        p_prime, v_prime = self.update_state(p, v, a) # tuple of new state
        i_prime, j_prime = [f(x) for x, f in zip([p_prime, v_prime] , self.mappings[:2])]
        i_prime, j_prime = min(self.p_len-1, i_prime), min(self.v_len -1, j_prime) # prevent the next state from being out of bounds
        # bellman eq: Qnew = Q + lr(reward + discount * best_action_est, - Q) 
        self.table[i, j, k] = Q + lr * (reward + discount * np.max(self.table[i_prime, j_prime, :]) - Q) 

def simulate(discount, lr, e_decay_rate, max_iters = 6000):
    reward = np.zeros(max_iters) 
    Qt = Q_table()
    epsilon = 1 # prob of exploring, 1- epsilon = prob of exploiting (best)
    e_decay = lambda e: max(e * e_decay_rate, .01) # how fast epsilon decays, the max comes from deep minds paper
    for i in range(max_iters):
        p, v = np.random.uniform(*START_RANGE), 0
        steps = 0
        while p < POSITION_SPACE[1] and steps < 250: # stop infintite loop
            a_fn = np.random.choice([Qt.get_best_action, Qt.get_random_action], p = [1 - epsilon, epsilon])
            a = a_fn(p, v)
            Qt.update_Q(-1, lr, discount, p, v, a)
            p, v = Qt.update_state(p, v, a)
            steps += 1
        reward[i] = steps
        epsilon = e_decay(epsilon)
    return reward

def hyper_search():
    """
    Lets do some hyperparam search! I am using an interative random search method. I will do a few trails, see what hyper params work and dont work then from
    there iterate. 
    """
    trials = 20
    scores = np.zeros(trials)
    discount_range = (.1, 1.)
    lr_range = (.1, .4)
    e_decay_range = (.5, 1.)
    hp = {}
    hp['discount'] = np.random.uniform(*discount_range, trials)
    hp['lr'] = np.random.uniform(*lr_range, trials)
    hp['epsilon_decay'] = np.random.uniform(*e_decay_range, trials)

    for i in range(trials):
        score = simulate(hp['discount'][i], hp['lr'][i], hp['epsilon_decay'][i])
        scores[i] = min(score)

    fix, axs = plt.subplots(3, 1)
    for ax, key in zip(axs, hp.keys()):
        ax.scatter(hp[key], scores)
        ax.set_title(key)
        ax.set_ylim([0, 1000])
    plt.show()

if __name__ == "__main__":
    hyper_search()
    

