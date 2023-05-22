from math import gamma
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import numpy as np
import envs
import gym
import matplotlib.pyplot as plt
import seaborn as sns

env = gym.make('ToyEnv-v0', onehot_state=False)
num_epis_train = int(1e4)
init_lr = 0.01
gamma = 0.95
eps = 0.3

length = 100
num_iter = 4 * length
nS = env.nS
nA = env.nA
Qtable = np.zeros((nS, nA))
render = False
visualize_learning_result = False
limited = True
# tau = 0.5

def qlearning(tau=0.5):
    for epis in range(num_epis_train):
        s = env.reset()
        # print(f"--------episode:{epis+1}---------")
        for iter in range(1, num_iter+1):
            if np.random.uniform(0, 1) < eps:
                a = np.random.choice(nA)
            else:
                a = np.argmax(Qtable[s,:])
            n_s, r, done,_ = env.step(a)
            # Qtable[s,a] = Qtable[s, a] + lr * (r + gamma*np.max(Qtable[n_s,:]) - Qtable[s, a])
            delta = r + gamma*np.max(Qtable[n_s,:]) - Qtable[s, a]
            alpha = 1 / (2 * max(tau,(1-tau)))
            if epis >= 1e3:
                frac = 1 - (epis - 1e3) / (num_epis_train - 1e3)
                lr = init_lr * frac
            else:
                lr = init_lr
            Qtable[s,a] = Qtable[s, a] + lr * 2 * alpha * (tau * np.maximum(delta, 0) + (1 - tau) * np.minimum(delta, 0))
            s = n_s
            if done: 
                break



for tau in range(11):
    tau = tau / 10
    env.seed(2000)
    qlearning(tau)
    ### state frequency
    env.use_wind = False
    num_rollouts = 10000
    state_freq = np.zeros(env.shape)
    episode_rewards = []
    for _ in range(num_rollouts):
        s = env.reset()
        episode_reward = 0
        while True:
            a  = np.argmax(Qtable[s,:])
            s, reward, done, info = env.step(a)
            episode_reward += reward
            if done:
                state_freq += env.visited
                episode_rewards.append(episode_reward)
                break

    state_freq = state_freq / np.sum(state_freq)
    print("state frequency: ", state_freq)
    plt.figure(figsize=(6,4))
    sns.heatmap(state_freq, cmap="Reds")
    plt.savefig(f"tau{tau}_state_frequency.png")