import torch
import torch.nn as nn
import numpy as np
import ray

from typing import Union, List
from util.util_util import check, get_gard_norm
from ppo.ppo_policy import PPOPolicy
from ppo.ppo_buffer import PPOBuffer

class PPOTrainer():
    def __init__(self, args, obs_space, act_space):
        self.device = torch.device("cuda:0") if args.cuda and torch.cuda.is_available() \
                                            else torch.device("cpu")
        self.tpdv = dict(dtype=torch.float32, device=self.device)
        # ppo config
        self.ppo_epoch = args.ppo_epoch
        self.clip_param = args.clip_param
        self.use_clipped_value_loss = args.use_clipped_value_loss
        self.num_mini_batch = args.num_mini_batch
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.use_max_grad_norm = args.use_max_grad_norm
        self.max_grad_norm = args.max_grad_norm
        # rnn configs
        self.use_recurrent_policy = args.use_recurrent_policy
        self.data_chunk_length = args.data_chunk_length
        self.buffer_size = args.buffer_size
        self.policy = PPOPolicy(args, obs_space, act_space)

    def ppo_update(self, sample):

        obs_batch, actions_batch, masks_batch, old_action_log_probs_batch, advantages_batch, \
            returns_batch, value_preds_batch, rnn_states_actor_batch, rnn_states_critic_batch = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        advantages_batch = check(advantages_batch).to(**self.tpdv)
        returns_batch = check(returns_batch).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)

        # Reshape to do in a single forward pass for all steps
        values, action_log_probs, dist_entropy = self.policy.evaluate_actions(obs_batch,
                                                                         rnn_states_actor_batch,
                                                                         rnn_states_critic_batch,
                                                                         actions_batch,
                                                                         masks_batch)

        # Obtain the loss function
        ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
        surr1 = ratio * advantages_batch
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages_batch
        policy_loss = torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True)
        policy_loss = -policy_loss.mean()

        if self.use_clipped_value_loss:
            value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
            value_losses = (values - returns_batch).pow(2)
            value_losses_clipped = (value_pred_clipped - returns_batch).pow(2)
            value_loss = 0.5 * torch.max(value_losses, value_losses_clipped)
        else:
            value_loss = 0.5 * (returns_batch - values).pow(2)
        value_loss = value_loss.mean()

        policy_entropy_loss = -dist_entropy.mean()

        loss = policy_loss + value_loss * self.value_loss_coef + policy_entropy_loss * self.entropy_coef

        # Optimize the loss function
        self.policy.optimizer.zero_grad()
        loss.backward()
        if self.use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm).item()
            critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm).item()
        else:
            actor_grad_norm = get_gard_norm(self.policy.actor.parameters())
            critic_grad_norm = get_gard_norm(self.policy.critic.parameters())
        self.policy.optimizer.step()

        return policy_loss, value_loss, policy_entropy_loss, ratio, actor_grad_norm, critic_grad_norm

    def train(self, buffer: Union[List[PPOBuffer], PPOBuffer], params, hyper_params={}):
        if 'entropy' in hyper_params:
            self.entropy_coef = hyper_params['entropy']
        self.policy.restore_from_params(params)
        buffer = [buffer] if isinstance(buffer, PPOBuffer) else buffer
        train_info = {}
        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['policy_entropy_loss'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0
        train_info['episode_reward'] = 0
        train_info['episode_length'] = 0

        for _ in range(self.ppo_epoch):
            if self.use_recurrent_policy:
                data_generator = PPOBuffer.recurrent_generator(buffer, self.num_mini_batch, self.data_chunk_length)
            else:
                raise NotImplementedError

            for sample in data_generator:

                policy_loss, value_loss, policy_entropy_loss, ratio, \
                    actor_grad_norm, critic_grad_norm = self.ppo_update(sample)

                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['policy_entropy_loss'] += policy_entropy_loss.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += ratio.mean().item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates
        
        episode_dones = max(np.sum(np.concatenate([buf.masks==0 for buf in buffer])),1)
        episode_reward = np.sum(np.concatenate([buf.rewards for buf in buffer])) / episode_dones
        episode_length = self.buffer_size * len(buffer) / episode_dones
        train_info['episode_reward'] = episode_reward
        train_info['episode_length'] = episode_length
        return self.policy.params(), train_info

@ray.remote(num_gpus=0.2)
class PBTPPOTrainer(PPOTrainer):

    def __init__(self, args, obs_space, act_space):
        PPOTrainer.__init__(self, args, obs_space, act_space)
