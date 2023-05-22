import argparse
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import gym
import numpy as np
from envs import toy
import random
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')
from rppo import Agent

def eval(args):
    ## config
    seed = 2021
    num_rollouts = 1000
    device = torch.device('cuda:0')
    tpdv = dict(dtype=torch.float32, device=device)
    env = toy.ToyEnv(wind=True, onehot_state=args.onehot_state)
    env.seed(seed)
    policy = Agent(env.observation_space, env.action_space).to(device)
    policy.restore(args.run_dir)

    ### state frequency
    state_freq = np.zeros(env.shape)
    episode_rewards = []
    violations = []
    for _ in range(num_rollouts):
        state = env.reset()
        episode_reward = 0
        water_times = 0
        step = 0
        while True:
            act = policy.get_action(torch.from_numpy(np.expand_dims(state, axis=0)).to(**tpdv))
            state, reward, done, info = env.step(act.item())
            episode_reward += reward
            water_times += 1 if reward == -1 else 0
            step += 1
            if done:
                state_freq += env.visited
                episode_rewards.append(episode_reward)
                violations.append(water_times / step)
                break
    np.save(args.run_dir + "/episode_rewards.npy", np.array(episode_rewards))
    np.save(args.run_dir + "/state_freq.npy", state_freq)
    state_freq = state_freq / np.sum(state_freq)
    print("state frequency: ", state_freq)
    plt.figure(figsize=(6,4))
    sns.heatmap(state_freq, cmap="Reds")
    plt.savefig(args.run_dir + "/state_frequency.pdf")

    # determinstic optimal path for no wind
    env.use_wind = False
    state = env.reset()
    while True:
        act = policy.get_action(torch.from_numpy(np.expand_dims(state, axis=0)).to(**tpdv))
        state, reward, done, info = env.step(act.item())
        if done:
            break
    print("policy path for no wind: ", env.visited)
    plt.figure(figsize=(6,4))
    sns.heatmap(env.visited, cmap="Reds")
    plt.savefig(args.run_dir + "/determinstic_path.pdf")
    return np.mean(episode_rewards), np.mean(violations)


if __name__ == "__main__":
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("--run-dir", type=str, default="runs/ToyEnv-v0/0.5/Apr07-121744")
        args = parser.parse_args()
        return args
    args = parse_args()
    args.onehot_state = True
    print(eval(args))
    # returns = []
    # violations = []
    # for tau in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    #     args.run_dir = f"toyexample/runs/risks/tau{tau}.pt"
    #     r, v = eval(args)
    #     returns.append(r)
    #     violations.append(v)
    # returns = np.array(returns)
    # violations = np.array(violations)
    # np.save("toy_returns.npy", returns)
    # np.save("toy_violations.npy", violations)