import argparse
import os
import random
import time
from distutils.util import strtobool
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import envs

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="ToyEnv-v0",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=1e6,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=200,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--gamma", type=float, default=0.95,
        help="the discount factor gamma")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=False,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")

    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Use GAE for advantage computation")
    parser.add_argument("--risk", type=lambda x: bool(strtobool(x)), default=True,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--onehot-state", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)

    args = parser.parse_args()
    args.batch_size = int(args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    return args


def make_env(args):
    env = gym.make(args.env_id)
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)
    return env

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, obs_space, act_space):
        super().__init__()
        obs_dim = int(np.prod(obs_space.shape))
        act_dim = act_space.n
        hidden_dim = 128
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, 1), std=0.01),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, act_dim), std=0.01),
        )

    def get_value(self, x):
        if x.ndim == 1:
            x = x.view((-1, 1))
        return self.critic(x)

    def get_action(self, x):
        if x.ndim == 1:
            x = x.view((-1, 1))
        logits = self.actor(x)
        dist = Categorical(logits=logits)
        action = dist.probs.argmax(dim=-1)
        return action

    def get_action_and_value(self, x, action=None):
        if x.ndim == 1:
            x = x.view((-1, 1))
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)
    
    def save(self, run_dir):
        torch.save(self.actor.state_dict(), run_dir + "/actor.pt")
        torch.save(self.critic.state_dict(), run_dir + "/critic.pt")

    def restore(self, run_dir):
        actor_state_dict = torch.load(run_dir + '/actor.pt')
        # critic_state_dict = torch.load(run_dir + '/critic.pt')
        # actor_state_dict = torch.load(run_dir)
        self.actor.load_state_dict(actor_state_dict)
        # self.critic.load_state_dict(critic_state_dict)

if __name__ == "__main__":
    args = parse_args()
    run_dir = f"runs/{args.env_id}/{args.exp_name}/{args.tau}/" + time.strftime("%b%d-%H%M%S", time.localtime())
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    args.run_dir = run_dir
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = make_env(args)
    assert isinstance(envs.action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs.observation_space, envs.action_space).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, 1) + envs.observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, 1) + envs.action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, 1)).to(device)
    rewards = torch.zeros((args.num_steps, 1)).to(device)
    dones = torch.zeros((args.num_steps+1, 1)).to(device)
    values = torch.zeros((args.num_steps+1, 1)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()).to(device).unsqueeze(0)
    next_done = False
    num_updates = int(args.total_timesteps // args.batch_size)
    value_loss = []
    for update in range(1, num_updates + 1):

        for step in range(0, args.num_steps):
            global_step += 1
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob
            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, info = envs.step(action.cpu().numpy().item())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            if done:
                next_obs = envs.reset()
            next_obs = torch.Tensor(next_obs).to(device).unsqueeze(0)
            next_done = torch.tensor(done).to(device).view(-1)


        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).view(-1)
            values[args.num_steps] = next_value
            dones[args.num_steps] = next_done

            max_depth = 25 # in toy example, max episode length is 25.
            advantages = torch.zeros_like(rewards).to(device)
            returns = torch.zeros_like(rewards).to(device)
            depths = torch.zeros(args.num_steps+1,)
            masks = 1- dones
            for i in reversed(range(args.num_steps)):
                depths[i] = min(depths[i+1]+1, max_depth)
                if masks[i+1]==0:
                    depths[i] = 1
            depths = depths.cpu().numpy().astype(np.int32)
            values_table = torch.zeros(max_depth, args.num_steps+1).to(device)
            values_table[0, args.num_steps] = next_value

            def operator(reward, value, nextvalue, mask):
                delta = reward + args.gamma * nextvalue * mask - value
                delta = args.tau * torch.maximum(delta, torch.zeros_like(delta)) + \
                        (1-args.tau) * torch.minimum(delta, torch.zeros_like(delta))
                alpha = 1 / (2 * max(args.tau, (1-args.tau)))
                delta = 2 * alpha * delta
                return value + delta

            if args.risk:
                if args.gae:
                    for step in reversed(range(args.num_steps)):
                        values_table[0, step] = operator(rewards[step], values[step], values[step+1], masks[step+1])
                        for d in range(depths[step]):
                            values_table[d][step] = operator(rewards[step], values_table[d-1][step], values_table[d-1][step+1], masks[step+1])
                        for d in reversed(range(depths[step])):
                            returns[step] = values_table[d][step] + args.gae_lambda * returns[step]
                        returns[step] *= (1-args.gae_lambda) / (1-args.gae_lambda**(depths[step]))
                else:
                    for t in reversed(range(args.num_steps)):
                        returns[t] = operator(rewards[t], values[t], values[t+1], masks[t+1])
                advantages = returns - values[:-1,...]
            else:
                if args.gae:
                    gae = 0
                    for t in reversed(range(args.num_steps)):
                        delta = rewards[t] + args.gamma * values[t+1] * masks[t+1] - values[t]
                        advantages[t] = gae = delta + args.gamma * args.gae_lambda * masks[t+1] * gae
                else:
                    raise NotImplementedError
                returns = advantages + values[:-1, ...]

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values[:-1, ...].reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                # pg_loss = pg_loss1.mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        value_loss.append(v_loss)
        if update % 1 == 0:
            print(f"Updates: {update}/{num_updates}, Returns:{(rewards.sum()/dones.sum()).item()}")
        agent.save(run_dir)
    
    envs.close()

    from toyexample.eval import eval
    print(eval(args))