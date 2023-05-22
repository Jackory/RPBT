import numpy as np
import torch
from typing import Union, List

from util.util_util import get_shape_from_space

class PPOBuffer():

    @staticmethod
    def _flatten(T: int, N: int, x: np.ndarray):
        return x.reshape(T * N, *x.shape[2:])

    @staticmethod
    def _cast(x: np.ndarray):
        return x.transpose(1, 0, *range(2, x.ndim)).reshape(-1, *x.shape[2:])

    def __init__(self, args, num_agents, obs_space, act_space):
        # buffer config
        self.buffer_size = args.buffer_size
        self.num_agents = num_agents
        self.gamma = args.gamma
        self.use_gae = args.use_gae
        self.gae_lambda = args.gae_lambda
        self.use_risk_sensitive = args.use_risk_sensitive
        self.reward_hyper = 1
        # rnn config
        self.recurrent_hidden_size = args.recurrent_hidden_size
        self.recurrent_hidden_layers = args.recurrent_hidden_layers

        obs_shape = get_shape_from_space(obs_space)
        act_shape = get_shape_from_space(act_space)

        # (o_0, a_0, r_0, d_1, o_1, ... , d_T, o_T)
        self.obs = np.zeros((self.buffer_size + 1, self.num_agents, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.num_agents, *act_shape), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.num_agents, 1), dtype=np.float32)
        # NOTE: masks[t] = 1 - dones[t-1], which represents whether obs[t] is a terminal state
        self.masks = np.ones((self.buffer_size + 1, self.num_agents, 1), dtype=np.float32)
        # NOTE: bad_masks[t] = 'bad_transition' in info[t-1], which indicates whether obs[t] a true terminal state or time limit end state
        self.bad_masks = np.ones((self.buffer_size + 1, self.num_agents, 1), dtype=np.float32)

        # pi(a)
        self.action_log_probs = np.zeros((self.buffer_size, self.num_agents, 1), dtype=np.float32)
        # V(o), R(o) while advantage = returns - value_preds
        self.value_preds = np.zeros((self.buffer_size + 1, self.num_agents, 1), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size + 1, self.num_agents, 1), dtype=np.float32)
        # rnn
        self.rnn_states_actor = np.zeros((self.buffer_size + 1, self.num_agents,
                                          self.recurrent_hidden_layers, self.recurrent_hidden_size), dtype=np.float32)
        self.rnn_states_critic = np.zeros_like(self.rnn_states_actor)
        self.step = 0

    @property
    def advantages(self) -> np.ndarray:
        advantages = self.returns[:-1] - self.value_preds[:-1]  # type: np.ndarray
        return (advantages - advantages.mean()) / (advantages.std() + 1e-5)

    def insert(self,
               obs: np.ndarray,
               actions: np.ndarray,
               rewards: np.ndarray,
               masks: np.ndarray,
               action_log_probs: np.ndarray,
               value_preds: np.ndarray,
               rnn_states_actor: np.ndarray,
               rnn_states_critic: np.ndarray,
               bad_masks: Union[np.ndarray, None] = None,
               **kwargs):
        """Insert numpy data.
        Args:
            obs:                o_{t+1}
            actions:            a_{t}
            rewards:            r_{t}
            masks:              mask[t+1] = 1 - done_{t}
            action_log_probs:   log_prob(a_{t})
            value_preds:        value(o_{t})
            rnn_states_actor:   ha_{t+1}
            rnn_states_critic:  hc_{t+1}
        """
        self.obs[self.step + 1] = obs.copy()
        self.actions[self.step] = actions.copy()
        self.rewards[self.step] = rewards.copy()
        self.masks[self.step + 1] = masks.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.value_preds[self.step] = value_preds.copy()
        self.rnn_states_actor[self.step + 1] = rnn_states_actor.copy()
        self.rnn_states_critic[self.step + 1] = rnn_states_critic.copy()
        if bad_masks is not None:
            self.bad_masks[self.step + 1] = bad_masks.copy()

        self.step = (self.step + 1) % self.buffer_size

    def after_update(self):
        """Copy last timestep data to first index. Called after update to model."""
        self.obs[0] = self.obs[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.bad_masks[0] = self.bad_masks[-1].copy()
        self.rnn_states_actor[0] = self.rnn_states_actor[-1].copy()
        self.rnn_states_critic[0] = self.rnn_states_critic[-1].copy()

    def clear(self):
        self.step = 0
        self.obs = np.zeros_like(self.obs, dtype=np.float32)
        self.actions = np.zeros_like(self.actions, dtype=np.float32)
        self.rewards = np.zeros_like(self.rewards, dtype=np.float32)
        self.masks = np.ones_like(self.masks, dtype=np.float32)
        self.bad_masks = np.ones_like(self.bad_masks, dtype=np.float32)
        self.action_log_probs = np.zeros_like(self.action_log_probs, dtype=np.float32)
        self.value_preds = np.zeros_like(self.value_preds, dtype=np.float32)
        self.returns = np.zeros_like(self.returns, dtype=np.float32)
        self.rnn_states_actor = np.zeros_like(self.rnn_states_critic)
        self.rnn_states_critic = np.zeros_like(self.rnn_states_actor)
    
    def dict(self):
        return {'buffer': self.step}

    def compute_returns(self, next_value: np.ndarray):
        """
        Compute returns either as discounted sum of rewards, or using GAE.

        Args:
            next_value(np.ndarray): value predictions for the step after the last episode step.
        """
        self.rewards[self.rewards==1] = self.rewards[self.rewards==1] * self.reward_hyper
        self.rewards[self.rewards==-1] = self.rewards[self.rewards==-1] / self.reward_hyper
        if self.use_risk_sensitive:
            max_depth = min(100, self.buffer_size)
            max_depth_vec = np.zeros((self.num_agents, 1)) + max_depth
            depths = np.zeros((self.buffer_size+1, self.num_agents, 1))
            for i in reversed(range(self.buffer_size)):
                depths[i] = np.minimum(depths[i+1] + 1, max_depth_vec)
                depths[i][self.masks[i+1]==0] = 1
            values_table = np.zeros((max_depth, self.buffer_size, self.num_agents, 1))

            def operator(reward, value, nextvalue, mask):
                delta = reward + self.gamma * nextvalue * mask - value
                delta = self.tau * np.maximum(delta, np.zeros_like(delta)) + \
                        (1-self.tau) * np.minimum(delta, np.zeros_like(delta))
                alpha = 1 / (2 * max(self.tau, (1-self.tau)))
                delta = 2 * alpha * delta
                return value + delta

            if self.use_gae:
                self.value_preds[-1] = next_value
                for t in reversed(range(self.rewards.shape[0])):
                    values_table[0, t] = operator(self.rewards[t], self.value_preds[t], self.value_preds[t+1], self.masks[t+1])
                    for h in range(1, int(depths[t].item())):
                        values_table[h, t] = operator(self.rewards[t], values_table[h-1][t], values_table[h-1][t+1], self.masks[t+1])
                    for h in reversed(range(int(depths[t].item()))):
                        self.returns[t] = values_table[h, t] + self.gae_lambda * self.returns[t]
                    self.returns[t] *= (1-self.gae_lambda) / (1-self.gae_lambda**(int(depths[t].item()))) 
            else:
                self.value_preds[-1] = next_value
                for step in reversed(range(self.rewards.shape[0])):
                    self.returns[step] = operator(self.rewards[step], self.value_preds[step], self.value_preds[step+1], self.masks[step])
        else:
            if self.use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.shape[0])):
                    td_delta = self.rewards[step] + self.gamma * self.value_preds[step + 1] * self.masks[step + 1] - self.value_preds[step]
                    gae = td_delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                    self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.shape[0])):
                    self.returns[step] = self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[step]


    @staticmethod
    def recurrent_generator(buffer, num_mini_batch: int, data_chunk_length: int):
        """
        A recurrent generator that yields training data for chunked RNN training arranged in mini batches.
        This generator shuffles the data by sequences.

        Args:
            buffers (Buffer or List[Buffer]) 
            num_mini_batch (int): number of minibatches to split the batch into.
            data_chunk_length (int): length of sequence chunks with which to train RNN.

        Returns:
            (obs_batch, actions_batch, masks_batch, old_action_log_probs_batch, advantages_batch, \
                returns_batch, value_preds_batch, rnn_states_actor_batch, rnn_states_critic_batch)
        """
        buffer = [buffer] if isinstance(buffer, PPOBuffer) else buffer  
        buffer_size = buffer[0].buffer_size
        num_agents = buffer[0].num_agents
        assert all([b.buffer_size == buffer_size for b in buffer]) \
            and all([b.num_agents == num_agents for b in buffer]) \
            and all([isinstance(b, PPOBuffer) for b in buffer]), \
            "Input buffers must has the same type and shape"
        buffer_size = buffer_size * len(buffer)

        assert buffer_size >= data_chunk_length, (
            "PPO requires the number of buffer size ({}) * num_agents ({})"
            "to be greater than or equal to the number of "
            "data chunk length ({}).".format(buffer_size, num_agents, data_chunk_length))

        # Transpose and reshape parallel data into sequential data
        obs = np.vstack([PPOBuffer._cast(buf.obs[:-1]) for buf in buffer])
        actions = np.vstack([PPOBuffer._cast(buf.actions) for buf in buffer])
        masks = np.vstack([PPOBuffer._cast(buf.masks[:-1]) for buf in buffer])
        old_action_log_probs = np.vstack([PPOBuffer._cast(buf.action_log_probs) for buf in buffer])
        advantages = np.vstack([PPOBuffer._cast(buf.advantages) for buf in buffer])
        returns = np.vstack([PPOBuffer._cast(buf.returns[:-1]) for buf in buffer])
        value_preds = np.vstack([PPOBuffer._cast(buf.value_preds[:-1]) for buf in buffer])
        rnn_states_actor = np.vstack([PPOBuffer._cast(buf.rnn_states_actor[:-1]) for buf in buffer])
        rnn_states_critic = np.vstack([PPOBuffer._cast(buf.rnn_states_critic[:-1]) for buf in buffer])

        # Get mini-batch size and shuffle chunk data
        data_chunks = buffer_size // data_chunk_length
        mini_batch_size = data_chunks // num_mini_batch
        rand = torch.randperm(data_chunks).numpy()
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]
        for indices in sampler:
            obs_batch = []
            actions_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            advantages_batch = []
            returns_batch = []
            value_preds_batch = []
            rnn_states_actor_batch = []
            rnn_states_critic_batch = []

            for index in indices:

                ind = index * data_chunk_length
                # size [T+1, N, Dim] => [T, N, Dim] => [N, T, Dim] => [N * T, Dim] => [L, Dim]
                obs_batch.append(obs[ind:ind + data_chunk_length])
                actions_batch.append(actions[ind:ind + data_chunk_length])
                masks_batch.append(masks[ind:ind + data_chunk_length])
                old_action_log_probs_batch.append(old_action_log_probs[ind:ind + data_chunk_length])
                advantages_batch.append(advantages[ind:ind + data_chunk_length])
                returns_batch.append(returns[ind:ind + data_chunk_length])
                value_preds_batch.append(value_preds[ind:ind + data_chunk_length])
                # size [T+1, N, Dim] => [T, N, Dim] => [N, T, Dim] => [N * T, Dim] => [1, Dim]
                rnn_states_actor_batch.append(rnn_states_actor[ind])
                rnn_states_critic_batch.append(rnn_states_critic[ind])

            L, N = data_chunk_length, mini_batch_size
            # These are all from_numpys of size (L, N, Dim)
            obs_batch = np.stack(obs_batch, axis=1)
            actions_batch = np.stack(actions_batch, axis=1)
            masks_batch = np.stack(masks_batch, axis=1)
            old_action_log_probs_batch = np.stack(old_action_log_probs_batch, axis=1)
            advantages_batch = np.stack(advantages_batch, axis=1)
            returns_batch = np.stack(returns_batch, axis=1)
            value_preds_batch = np.stack(value_preds_batch, axis=1)

            # States is just a (N, -1) from_numpy
            rnn_states_actor_batch = np.stack(rnn_states_actor_batch).reshape(N, *buffer[0].rnn_states_actor.shape[2:])
            rnn_states_critic_batch = np.stack(rnn_states_critic_batch).reshape(N, *buffer[0].rnn_states_critic.shape[2:])

            # Flatten the (L, N, ...) from_numpys to (L * N, ...)
            obs_batch = PPOBuffer._flatten(L, N, obs_batch)
            actions_batch = PPOBuffer._flatten(L, N, actions_batch)
            masks_batch = PPOBuffer._flatten(L, N, masks_batch)
            old_action_log_probs_batch = PPOBuffer._flatten(L, N, old_action_log_probs_batch)
            advantages_batch = PPOBuffer._flatten(L, N, advantages_batch)
            returns_batch = PPOBuffer._flatten(L, N, returns_batch)
            value_preds_batch = PPOBuffer._flatten(L, N, value_preds_batch)

            yield obs_batch, actions_batch, masks_batch, old_action_log_probs_batch, advantages_batch, \
                returns_batch, value_preds_batch, rnn_states_actor_batch, rnn_states_critic_batch