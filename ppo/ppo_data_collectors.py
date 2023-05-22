import imp
import numpy as np
import torch
import time
import gym
import ray
import envs
from ppo.ppo_buffer import PPOBuffer
from ppo.ppo_policy import PPOPolicy
from gym.core import Wrapper
import wandb
def _t2n(x):
    return x.detach().cpu().numpy()

class GymEnv(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_shape = self.env.action_space.shape
        self.num_agents = 1

    def reset(self):
        observation = self.env.reset()
        return np.array(observation).reshape((1, -1))

    def step(self, action):
        action = np.array(action).reshape(self.action_shape)
        observation, reward, done, info = self.env.step(action)
        observation = np.array(observation).reshape((1, -1))
        done = np.array(done).reshape((1,-1))
        reward = np.array(reward).reshape((1, -1))
        info['score'] = 0.5
        return observation, reward, done, info

def make_env(env_name):
    if env_name in ['Ant-v2', 'Hopper-v2']: # single-agent
        env = GymEnv(gym.make(env_name))
    else: # selfplay
        env = gym.make(env_name)
    return env

class BaseDataCollector(object):
    def __init__(self, args) -> None:
        self.args = args
        self.env = make_env(args.env_name)
        if args.capture_video and args.use_wandb and args.env_id == 0:
            wandb.init(
                project=args.env_name, 
                entity=args.wandb_name, 
                name=args.experiment_name, 
                group=args.experiment_name,
                job_type="video",
                monitor_gym=True)
            self.env = gym.wrappers.RecordVideo(self.env, args.save_dir+"_videos")
        self.recurrent_hidden_layers = args.recurrent_hidden_layers
        self.recurrent_hidden_size = args.recurrent_hidden_size
        self.buffer_size = args.buffer_size
        self.num_agents = getattr(self.env, 'num_agents', 1)
        self.random_side = args.random_side
        self.buffer = PPOBuffer(args, self.num_agents, self.env.observation_space, self.env.action_space)
        self.ego = PPOPolicy(args, self.env.observation_space, self.env.action_space)
    
    def collect_data(self, ego_params, hyper_params={}):
        self.buffer.clear()
        if 'tau' in hyper_params:
            self.buffer.tau = hyper_params['tau'] 
        if 'reward' in hyper_params:
            self.buffer.reward_hyper = hyper_params['reward']
        self.ego.restore_from_params(ego_params)
        obs = self.reset()
        self.buffer.obs[0] = obs.copy()
        for step in range(self.buffer_size):
            # 1. get actions
            values, actions, action_log_probs, rnn_states_actor, rnn_states_critic = \
                self.ego.get_actions(self.buffer.obs[step],
                                     self.buffer.rnn_states_actor[step],
                                     self.buffer.rnn_states_critic[step],
                                     self.buffer.masks[step])
            values = _t2n(values)
            actions = _t2n(actions)
            action_log_probs = _t2n(action_log_probs)
            rnn_states_actor = _t2n(rnn_states_actor)
            rnn_states_critic = _t2n(rnn_states_critic)
            # 2. env step
            obs, rewards, dones, info = self.step(actions)
            if np.all(dones):
                obs = self.reset()
                rnn_states_actor = np.zeros((self.num_agents, self.recurrent_hidden_layers, self.recurrent_hidden_size))
                rnn_states_critic = np.zeros((self.num_agents, self.recurrent_hidden_layers, self.recurrent_hidden_size))
                masks = np.zeros((self.num_agents, 1), dtype=np.float32)
            else:
                masks = np.ones((self.num_agents, 1), dtype=np.float32)
            # 3. insert experience in buffer
            self.buffer.insert(obs, actions, rewards, masks, action_log_probs, values, rnn_states_actor, rnn_states_critic)
        status_code = 0 if step > 0 else 1
        last_value = self.ego.get_values(self.buffer.obs[-1], self.buffer.rnn_states_critic[-1], self.buffer.masks[-1])
        self.buffer.compute_returns(_t2n(last_value))
        return self.buffer
    
    @torch.no_grad()
    def evaluate_data(self, ego_params, hyper_params={}, ego_elo=0, enm_elo=0):
        self.ego.restore_from_params(ego_params)
        total_episode_rewards = []
        eval_scores = []
        max_episode_length = self.args.buffer_size
        episode_reward = 0
        eval_episodes = self.args.eval_episodes
        for _ in range(eval_episodes):
            obs = self.reset()
            rnn_states_actor = np.zeros((self.num_agents, self.recurrent_hidden_layers, self.recurrent_hidden_size))
            masks = np.ones((self.num_agents, 1))
            step = 0
            while True:
                action, rnn_states_actor = self.ego.act(obs, rnn_states_actor, masks)
                action = _t2n(action)
                rnn_states_actor = _t2n(rnn_states_actor)
                obs, reward, done, info = self.step(action)
                step += 1
                episode_reward += reward
                if np.all(done):
                    total_episode_rewards.append(episode_reward)
                    episode_reward = 0
                    if isinstance(info, tuple) or isinstance(info, list):
                        score = info[0]['score']
                    elif isinstance(info, dict):
                        score = info['score']
                    else:
                        raise NotImplementedError
                    eval_scores.append(score)
                    break
                if step >= max_episode_length:
                    break
        expected_score = 1 / (1 + 10 ** ((enm_elo - ego_elo) / 400))
        elo_gain = 32 * (np.mean(eval_scores) - expected_score)
        eval_info = {}
        eval_info["episode_reward"] = np.mean(total_episode_rewards)
        return  elo_gain, eval_info

    def reset(self):
        return self.env.reset()
    
    def step(self, action):
        return self.env.step(action)

class SelfPlayDataCollector(BaseDataCollector):

    def __init__(self, args):
        super().__init__(args)
        self.enm = PPOPolicy(args, self.env.observation_space, self.env.action_space)
        self.num_agents = self.num_agents // 2
        self.buffer = PPOBuffer(args, self.num_agents, self.env.observation_space, self.env.action_space)
    
    def collect_data(self, ego_params, enm_params, hyper_params={}):
        self.enm.restore_from_params(enm_params)
        return super().collect_data(ego_params, hyper_params)
    
    @torch.no_grad()
    def evaluate_data(self, ego_params, enm_params, hyper_params={}, ego_elo=0, enm_elo=0):
        self.enm.restore_from_params(enm_params)
        return super().evaluate_data(ego_params, hyper_params, ego_elo, enm_elo)
    
    def reset(self):
        if self.random_side:
            self.ego_side = np.random.randint(2) # shuffle rl control side
        else:
            self.ego_side = 0
        self.enm_rnn_states_actor = np.zeros((self.num_agents, self.recurrent_hidden_layers, self.recurrent_hidden_size))
        self.enm_masks = np.ones((self.num_agents, 1))
        obs = super().reset()
        return self._parse_obs(obs)
    
    def step(self, action):
        enm_action, enm_rnn_states_actor = self.enm.act(self.enm_obs, self.enm_rnn_states_actor, self.enm_masks)
        enm_action = _t2n(enm_action)
        self.enm_rnn_states_actor = _t2n(enm_rnn_states_actor)
        actions = np.concatenate((action, enm_action), axis=0)
        n_obs, rewards, done, info = self.env.step(actions)
        if self.ego_side == 0:
            ego_rewards = rewards[:self.num_agents, ...]
            ego_done = done[:self.num_agents, ...]
        else:
            ego_rewards = rewards[self.num_agents:, ...]
            ego_done = done[self.num_agents:, ...]
        return self._parse_obs(n_obs), ego_rewards, ego_done, info
    
    def _parse_obs(self, obs):
        if self.ego_side == 0:
            self.enm_obs = obs[self.num_agents:, ...]
            ego_obs = obs[:self.num_agents, ...]
        else:
            self.enm_obs = obs[:self.num_agents, ...]
            ego_obs = obs[self.num_agents:, ...]
        return ego_obs

@ray.remote(num_cpus=0.5)
class DataCollectorMix(object):
    def __init__(self, args) -> None:
        self.collector = None
        self.mode = None
        self.args = args
    
    def set_collector(self, mode):
        if self.mode == mode:
            return
        self.mode = mode
        if self.collector is not None:
            self.collector.close()
        if self.mode == 'base':
            self.collector = BaseDataCollector(self.args)
        elif self.mode == 'selfplay':
            self.collector = SelfPlayDataCollector(self.args)
        else:
            raise NotImplementedError

    def collect_data(self, ego_params, enm_params=None, hyper_params={}):
        if enm_params == None:
            mode = 'base'
        elif isinstance(enm_params, dict):
            mode = 'selfplay'
        else:
            raise NotImplementedError
        self.set_collector(mode)
        if self.mode == 'base':
            return self.collector.collect_data(ego_params, hyper_params)
        elif self.mode == 'selfplay':
            return self.collector.collect_data(ego_params, enm_params, hyper_params)
        else: 
            raise NotImplementedError
    
    def evaluate_data(self, ego_params, enm_params=None, hyper_params={}, ego_elo=0, enm_elo=0):
        if enm_params == None:
            mode = 'base'
        elif isinstance(enm_params, dict):
            mode = 'selfplay'
        else:
            raise NotImplementedError
        self.set_collector(mode)
        if self.mode == 'base':
            return self.collector.evaluate_data(ego_params, hyper_params, ego_elo, enm_elo)
        elif self.mode == 'selfplay':
            return self.collector.evaluate_data(ego_params, enm_params, hyper_params, ego_elo, enm_elo)
        else: 
            raise NotImplementedError