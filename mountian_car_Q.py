import numpy as np
from math import floor
import matplotlib.pyplot as plt
from celluloid import Camera


POSITION_SPACE = (-1.2, .6)
VELO_SPACE = (-.07, .07)
ACTION_SPACE = [-1, 0, 1]
START_RANGE = (-.6, -.4)

"""I am frequently weary of oop because sometimes dev time is needlessly high from my experience, however, it think
it provides a really clean and nice abstraction in this senerio and actually keeps dev time lower. I believe it 
is import to use tools when appropriate, i think oop is here"""

class Q_table():
    def __init__(self, discrete_ps = 30, discrete_vs = 30, discrete_as = 3):
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

def simulate(discount, lr, e_decay_rate, max_iters = 5000, return_table = False):
    rewards = np.zeros(max_iters) 
    Qt = Q_table()
    epsilon = 1 # prob of exploring, 1- epsilon = prob of exploiting (best)
    e_decay = lambda e: max(e * e_decay_rate, .01) # how fast epsilon decays, the max comes from deep minds paper
    for i in range(max_iters):
        p, v = np.random.uniform(*START_RANGE), 0
        reward = 0
        while p < POSITION_SPACE[1] and reward < 250: # stop infintite loop
            a_fn = np.random.choice([Qt.get_best_action, Qt.get_random_action], p = [1 - epsilon, epsilon])
            a = a_fn(p, v)
            Qt.update_Q(-1, lr, discount, p, v, a)
            p, v = Qt.update_state(p, v, a)
            reward += 1
        rewards[i] = reward
        epsilon = e_decay(epsilon)
    if return_table:
        return rewards, Qt
    else:
        return rewards

def hyper_search():
    """
    Lets do some hyperparam search! I am using an interative random search method. I will do a few trails, see what hyper params work and dont work then from
    there iterate. 
    """
    trials = 20
    scores = []
    discount_range = (.1, 1.)
    lr_range = (.1, .4)
    e_decay_range = (.5, 1.)
    hp = {}
    hp['discount'] = np.random.uniform(*discount_range, trials)
    hp['lr'] = np.random.uniform(*lr_range, trials)
    hp['epsilon_decay'] = np.random.uniform(*e_decay_range, trials)

    for i in range(trials):
        hold = (hp['discount'][i], hp['lr'][i], hp['epsilon_decay'][i])
        score = simulate(*hold)
        scores.append(score)
        print("score: {}, discount: {}, lr: {}, epsilon_decay: {}".format(np.average(score[-500:]), *hold))

    fig, axs = plt.subplots(3, 1)
    for ax, key in zip(axs, hp.keys()):
        ax.scatter(hp[key], np.average(scores[i]))
        ax.set_title(key)
    fig.show()
    """
    score: 211.72, discount: 0.2824933520659566, lr: 0.17244939592088193, epsilon_decay: 0.9422534370060514
    score: 221.968, discount: 0.449881569041625, lr: 0.2797153971134617, epsilon_decay: 0.5612766255280709
    score: 212.474, discount: 0.47365344896391615, lr: 0.34464527505672415, epsilon_decay: 0.7545763229312964
    score: 147.08, discount: 0.8941097749008459, lr: 0.281720165576596, epsilon_decay: 0.6402003793646793
    score: 191.864, discount: 0.694823346681323, lr: 0.1975957805539046, epsilon_decay: 0.6081500458326561
    score: 223.624, discount: 0.5669016355520428, lr: 0.3532305508549993, epsilon_decay: 0.6683549632717782
    score: 193.684, discount: 0.9403139125426704, lr: 0.38720828795289886, epsilon_decay: 0.9810833438406337
    score: 175.858, discount: 0.4590723333145095, lr: 0.22499849644943165, epsilon_decay: 0.5712505297039662
    score: 183.182, discount: 0.4603140408915277, lr: 0.1298694973714855, epsilon_decay: 0.7293091177345252
    score: 250.0, discount: 0.15668291093964357, lr: 0.37399716877833344, epsilon_decay: 0.6667601971835406
    score: 189.818, discount: 0.9302288566529271, lr: 0.3461452836964455, epsilon_decay: 0.5225729267470409
    score: 206.89, discount: 0.7393388279997576, lr: 0.38404711840235606, epsilon_decay: 0.6978357947772437
    score: 250.0, discount: 0.13798033479312335, lr: 0.3941747862632219, epsilon_decay: 0.9399145331098893
    score: 198.196, discount: 0.7249406276042496, lr: 0.233962415206557, epsilon_decay: 0.65366939065216
    score: 222.288, discount: 0.18603651792662607, lr: 0.10025206638754855, epsilon_decay: 0.746878295600906
    score: 214.428, discount: 0.2933310640544758, lr: 0.11295598407531794, epsilon_decay: 0.9064368432918122
    score: 179.178, discount: 0.6077255751840971, lr: 0.1592095997311603, epsilon_decay: 0.7834561753174173
    score: 172.55, discount: 0.8588934763634241, lr: 0.11672460536562612, epsilon_decay: 0.700945100569841
    score: 231.534, discount: 0.6922898626050366, lr: 0.25278751556240864, epsilon_decay: 0.8105051350601509
    score: 248.862, discount: 0.1599106876760406, lr: 0.13941395688375366, epsilon_decay: 0.7350357615629758
    """

def train_and_plot(lr, discount, epsilon_decay):
    score, Qt = simulate(lr, discount, epsilon_decay, return_table = True)
    N = 400 # thanks bill leonard
    x = np.linspace(*POSITION_SPACE, N)
    fn = lambda x: np.sin(3*x)
    y = fn(x)

    fig, ax = plt.subplots(1, 1)
    camera = Camera(fig)
    
    p, v = -.5, 0.
    while p < POSITION_SPACE[1]: # stop infintite loop
        a = Qt.get_best_action(p, v)
        p, v = Qt.update_state(p, v, a)
        ax.scatter(p, fn(p), s = 15, c = 'r')
        ax.plot(x, y, c = 'b') # i know repeative but im not memory constrained so im not going to rewrite
        camera.snap()
    animation = camera.animate(fps = 30))
    animation.save('mountiancar.gif', writer='Pillow')

if __name__ == "__main__":
    train_and_plot(.894, .282, .640)
    
