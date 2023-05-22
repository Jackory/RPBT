import torch
from ppo.ppo_actor import PPOActor
from ppo.ppo_critic import PPOCritic


class PPOPolicy:
    def __init__(self, args, obs_space, act_space):

        self.args = args
        self.device = torch.device("cuda:0") if args.cuda and torch.cuda.is_available() \
                                            else torch.device("cpu")
        # optimizer config
        self.lr = args.lr

        self.obs_space = obs_space
        self.act_space = act_space

        self.actor = PPOActor(args, self.obs_space, self.act_space, self.device)
        self.critic = PPOCritic(args, self.obs_space, self.device)

        self.optimizer = torch.optim.Adam([
            {'params': self.actor.parameters()},
            {'params': self.critic.parameters()}
        ], lr=self.lr)

    def get_actions(self, obs, rnn_states_actor, rnn_states_critic, masks):
        """
        Returns:
            values, actions, action_log_probs, rnn_states_actor, rnn_states_critic
        """
        actions, action_log_probs, rnn_states_actor = self.actor(obs, rnn_states_actor, masks)
        values, rnn_states_critic = self.critic(obs, rnn_states_critic, masks)
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def get_values(self, obs, rnn_states_critic, masks):
        """
        Returns:
            values
        """
        values, _ = self.critic(obs, rnn_states_critic, masks)
        return values

    def evaluate_actions(self, obs, rnn_states_actor, rnn_states_critic, action, masks, active_masks=None):
        """
        Returns:
            values, action_log_probs, dist_entropy
        """
        action_log_probs, dist_entropy = self.actor.evaluate_actions(obs, rnn_states_actor, action, masks, active_masks)
        values, _ = self.critic(obs, rnn_states_critic, masks)
        return values, action_log_probs, dist_entropy

    def act(self, obs, rnn_states_actor, masks, deterministic=False):
        """
        Returns:
            actions, rnn_states_actor
        """
        actions, _, rnn_states_actor = self.actor(obs, rnn_states_actor, masks, deterministic)
        return actions, rnn_states_actor

    def prep_training(self):
        self.actor.train()
        self.critic.train()

    def prep_rollout(self):
        self.actor.eval()
        self.critic.eval()

    def copy(self):
        return PPOPolicy(self.args, self.obs_space, self.act_space, self.device)

    def params(self, device='cuda'):
        checkpoint = {}
        if device == 'cuda':
            checkpoint['actor_state_dict'] = self.actor.state_dict()
            checkpoint['critic_state_dict'] = self.critic.state_dict()
        elif device == 'cpu':
            checkpoint['actor_state_dict'] = {k: v.cpu() for k, v in self.actor.state_dict().items()}
            checkpoint['critic_state_dict'] = {k: v.cpu() for k, v in self.critic.state_dict().items()}

        return checkpoint
        
    def save(self, rootpath, epoch=0):
        checkpoint = {}
        checkpoint['actor_state_dict'] = self.actor.state_dict()
        checkpoint['critic_state_dict'] = self.critic.state_dict()
        torch.save(checkpoint, str(rootpath) + f'/agent_{epoch}.pt')
        torch.save(checkpoint, str(rootpath) + f'/agent_latest.pt')
    
    def restore_from_path(self, rootpath, epoch=''):
        checkpoint = torch.load(str(rootpath) + f'/agent0_{epoch}.pt')
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        if 'critic_state_dict' in checkpoint.keys():
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
    
    def restore_from_params(self, params):
        self.actor.load_state_dict(params['actor_state_dict'])
        if 'critic_state_dict' in params.keys():
            self.critic.load_state_dict(params['critic_state_dict'])