import numpy as np
import torch
import gym
import os
import sys
from pathlib import Path
base_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(base_dir)
import envs
from PIL import Image
from config import get_config
from ppo.ppo_buffer import PPOBuffer
from ppo.ppo_policy import PPOPolicy
from gym.wrappers import RecordVideo
from ppo.ppo_data_collectors import make_env


def _t2n(x):
    return x.detach().cpu().numpy()

episode_rewards = 0
render_video = True
render_image = False
num_agents = 2
args = sys.argv[1:]
parser = get_config()
all_args = parser.parse_known_args(args)[0]
all_args.env_name = 'SumoAnts-v0'
env = make_env(all_args.env_name)
ego_policy = PPOPolicy(all_args, env.observation_space, env.action_space)
enm_policy = PPOPolicy(all_args, env.observation_space, env.action_space)
ego_dir = ""
enm_dir = ""
ego_path = ""
enm_path = ""
ego_policy.restore_from_params(torch.load(ego_dir+ego_path))
enm_policy.restore_from_params(torch.load(enm_dir+enm_path))
if render_video == True:
    env = RecordVideo(env, video_folder="render/render_videos", name_prefix=f"0.1vs0.9")
env.seed(0)

print("Start render")
obs = env.reset()
step = 0
if render_image and step % 1 == 0:
    arr = env.render(mode="rgb_array")
    img = Image.fromarray(arr)
    img.save(f"render/images/step{step}.png")
ego_rnn_states = np.zeros((1, 1, 128), dtype=np.float32)
enm_obs =  obs[num_agents // 2:, ...]
ego_obs =  obs[:num_agents // 2, ...]
enm_rnn_states = np.zeros_like(ego_rnn_states, dtype=np.float32)
masks = np.ones((num_agents//2, 1))
while True:
    step += 1
    # env.render()
    ego_actions, ego_rnn_states = ego_policy.act(ego_obs, ego_rnn_states, masks, deterministic=False)
    ego_actions = _t2n(ego_actions)
    ego_rnn_states = _t2n(ego_rnn_states)
    # print(ego_actions)
    enm_actions, enm_rnn_states = enm_policy.act(enm_obs, enm_rnn_states, masks, deterministic=False)
    enm_actions = _t2n(enm_actions)
    enm_rnn_states = _t2n(enm_rnn_states)
    actions = np.concatenate((ego_actions, enm_actions), axis=0)
    obs, rewards, dones, infos = env.step(actions)
    rewards = rewards[:num_agents // 2, ...]
    episode_rewards += rewards
    if render_image and step % 1 == 0:
        arr = env.render(mode="rgb_array")
        img = Image.fromarray(arr)
        img.save(f"render/images/step{step}.png")
    if dones.all():
        print(infos)
        break
    enm_obs =  obs[num_agents // 2:, ...]
    ego_obs =  obs[:num_agents // 2, ...]

