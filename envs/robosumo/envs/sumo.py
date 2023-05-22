"""
Multi-agent sumo environment.
"""
import os
import tempfile

import numpy as np

import gym
from gym.spaces import Tuple
from gym.utils import EzPickle

# MuJoCo 1.5+
from mujoco_py import MjViewer
from envs.robosumo.envs import MujocoEnv

from . import agents
from .utils import construct_scene


_AGENTS = {
    'ant': os.path.join(os.path.dirname(__file__), "assets", "ant.xml"),
    'bug': os.path.join(os.path.dirname(__file__), "assets", "bug.xml"),
    'spider': os.path.join(os.path.dirname(__file__), "assets", "spider.xml"),
}


class SumoEnv(MujocoEnv, EzPickle):
    """
    Multi-agent sumo environment.

    The goal of each agent is to push the other agent outside the tatami area.
    The reward is shaped such that agents learn to prefer staying in the center
    and pushing each other further away from the center. If any of the agents
    gets outside of the tatami (even accidentially), it gets -WIN_REWARD_COEF
    and the opponent gets +WIN_REWARD_COEF.
    """
    STAY_IN_CENTER_COEF = 0.1
    DRAW_PENALTY = -1000.
    # MOVE_TO_CENTER_COEF = 0.1
    MOVE_TO_OPP_COEF = 0.1
    PUSH_OUT_COEF = 10.0

    def __init__(self, agent_names,
                 xml_path=None,
                 init_pos_noise=.1,
                 init_vel_noise=.1,
                 agent_kwargs=None,
                 frame_skip=5,
                 tatami_size=2.0,
                 timestep_limit=500,
                 reward_shape=1.,
                 **kwargs):
        EzPickle.__init__(self)
        self._tatami_size = tatami_size + 0.1
        self._timestep_limit = timestep_limit
        self._init_pos_noise = init_pos_noise
        self._init_vel_noise = init_vel_noise
        self._n_agents = len(agent_names)
        self._mujoco_init = False
        self._num_steps = 0
        self._spec = None
        self.WIN_REWARD = 2000 * reward_shape
        self.LOSE_REWARD = - 2000 / reward_shape

        # Resolve agent scopes
        agent_scopes = [
            "%s%d" % (name, i)
            for i, name in enumerate(agent_names)
        ]

        # Consturct scene XML
        scene_xml_path = os.path.join(os.path.dirname(__file__),
                                      "assets", "tatami.xml")
        agent_xml_paths = [_AGENTS[name] for name in agent_names]
        scene = construct_scene(scene_xml_path, agent_xml_paths,
                                agent_scopes=agent_scopes,
                                tatami_size=tatami_size,
                                **kwargs)
        self.tatami_height = 0.5

        # Init MuJoCo
        if xml_path is None:
            with tempfile.TemporaryDirectory() as tmpdir_name:
                scene_filepath = os.path.join(tmpdir_name, "scene.xml")
                scene.write(scene_filepath)
                MujocoEnv.__init__(self, scene_filepath, frame_skip)
        else:
            with open(xml_path, 'w') as fp:
                scene.write(fp.name)
            MujocoEnv.__init__(self, fp.name, frame_skip)
        self._mujoco_init = True

        # Construct agents
        agent_kwargs = agent_kwargs or {}
        self.agents = [
            agents.get(name, env=self, scope=agent_scopes[i], **agent_kwargs)
            for i, name in enumerate(agent_names)
        ]

        # Set opponents
        for i, agent in enumerate(self.agents):
            agent.set_opponents([
                agent for j, agent in enumerate(self.agents) if j != i
            ])

        # Setup agents
        for i, agent in enumerate(self.agents):
            agent.setup_spaces()

        # Set observation and action spaces
        # self.observation_space = Tuple([
        #     agent.observation_space for agent in self.agents
        # ])
        # self.action_space = Tuple([
        #     agent.action_space for agent in self.agents
        # ])
        self.observation_space = self.agents[0].observation_space
        self.action_space = self.agents[0].action_space
        self.num_agents = 2
        self.is_vector_env = True

    def simulate(self, actions):
        a = np.concatenate(actions, axis=0)
        self.do_simulation(a, self.frame_skip)

    def step(self, actions):
        if not self._mujoco_init:
            return self._get_obs(), 0, False, None

        dones = [False for _ in range(self._n_agents)]
        rewards = [0. for _ in range(self._n_agents)]
        infos = [{} for _ in range(self._n_agents)]

        # Call `before_step` on the agents
        for i in range(self._n_agents):
            self.agents[i].before_step()

        # Do simulation
        self.simulate(actions)

        # Call `after_step` on the agents
        for i in range(self._n_agents):
            infos[i]['ctrl_reward'] = self.agents[i].after_step(actions[i])

        # Get obs
        obs = self._get_obs()

        self._num_steps += 1

        # Compute rewards and dones
        for i, agent in enumerate(self.agents):
            self_xyz = agent.get_qpos()[:3]
            # Loose penalty
            infos[i]['lose_penalty'] = 0.
        
            if (self_xyz[2] < 0.29 + self.tatami_height or
                    np.max(np.abs(self_xyz[:2])) >= self._tatami_size):
                infos[i]['lose_penalty'] += self.LOSE_REWARD
                dones[i] = True
            # Win reward
            infos[i]['win_reward'] = 0.
            for opp in agent._opponents:
                opp_xyz = opp.get_qpos()[:3]
                if (opp_xyz[2] < 0.29 + self.tatami_height or
                        np.max(np.abs(opp_xyz[:2])) >= self._tatami_size):
                    infos[i]['win_reward'] += self.WIN_REWARD
                    infos[i]['winner'] = True
                    dones[i] = True
            infos[i]['main_reward'] = \
                infos[i]['win_reward'] + infos[i]['lose_penalty']
            # Draw penalty
            if self._num_steps > self._timestep_limit:
                infos[i]['main_reward'] += self.DRAW_PENALTY
                dones[i] = True
            # Move to opponent(s) and push them out of center
            infos[i]['move_to_opp_reward'] = 0.
            infos[i]['push_opp_reward'] = 0.
            for opp in agent._opponents:
                infos[i]['move_to_opp_reward'] += \
                    self._comp_move_reward(agent, opp.posafter)
                infos[i]['push_opp_reward'] += \
                    self._comp_push_reward(agent, opp.posafter)
            # Stay in center reward (unused)
            # infos[i]['stay_in_center'] = self._comp_stay_in_center_reward(agent)
            # Contact rewards and penalties (unused)
            infos[i]['contact_reward'] = self._comp_contact_reward(agent)
            # Reward shaping
            infos[i]['shaping_reward'] = \
                infos[i]['ctrl_reward'] + \
                infos[i]['push_opp_reward'] + \
                infos[i]['move_to_opp_reward'] + \
                infos[i]['contact_reward']
            # Add up rewards
            rewards[i] = infos[i]['main_reward'] + infos[i]['shaping_reward']

        rewards = tuple(rewards) # normlize
        dones = tuple(dones)
        if np.all(dones):
            if 'winner' in infos[0]:
                infos[0]['score'] = 1.
            elif 'winner' in infos[1]:
                infos[0]['score'] = 0.
            else:
                infos[0]['score'] = 0.5
        infos = tuple(infos)

        return np.array(obs).reshape(self.num_agents, -1), np.array(rewards).reshape(self.num_agents, -1), np.array(dones).reshape(self.num_agents, -1), infos

    def _comp_move_reward(self, agent, target):
        move_vec = (agent.posafter - agent.posbefore) / self.dt
        direction = target - agent.posbefore
        direction /= np.linalg.norm(direction)
        return max(np.sum(move_vec * direction), 0.) * self.MOVE_TO_OPP_COEF

    def _comp_push_reward(self, agent, target):
        dist_to_center = np.linalg.norm(target)
        return - self.PUSH_OUT_COEF * np.exp(-dist_to_center)

    def _comp_stay_in_center_reward(self, agent):
        dist_to_center = np.linalg.norm(agent.posafter)
        return self.STAY_IN_CENTER_COEF * np.exp(-dist_to_center)

    def _comp_contact_reward(self, agent):
        # Penalty for pain
        body_ids = [
            agent.body_name_idx[name]
            for name in ['head', 'torso'] if name in agent.body_name_idx
        ]
        forces = np.clip(agent.get_cfrc_ext(body_ids), -100., 100.)
        pain = agent.COST_COEFS['pain'] * np.sum(np.abs(forces))
        # Reward for attacking opponents
        attack = 0.
        for other in agent._opponents:
            body_ids = [
                other.body_name_idx[name]
                for name in ['head', 'torso'] if name in other.body_name_idx
            ]
            forces = np.clip(other.get_cfrc_ext(body_ids), -100., 100.)
            attack += agent.COST_COEFS['attack'] * np.sum(np.abs(forces))
        return attack - pain

    def _get_obs(self):
        if not self._mujoco_init:
            return np.concatenate([self.data.qpos.flat, self.data.qvel.flat])
        return np.array([agent.get_obs() for agent in self.agents])

    def reset_model(self):
        self._num_steps = 0
        # Randomize agent positions
        r, z = 1.15, 1.25
        delta = (2. * np.pi) / self._n_agents
        phi = self.np_random.uniform(0., 2. * np.pi)
        for i, agent in enumerate(self.agents):
            angle = phi + i * delta
            x, y = r * np.cos(angle), r * np.sin(angle)
            agent.set_xyz((x, y, z))
        # Add noise to all qpos and qvel elements
        pos_noise = self.np_random.uniform(
            size=self.model.nq,
            low=-self._init_pos_noise,
            high=self._init_pos_noise)
        vel_noise = self._init_vel_noise * \
                    self.np_random.randn(self.model.nv)
        qpos = self.data.qpos.ravel() + pos_noise
        qvel = self.data.qvel.ravel() + vel_noise
        self.init_qpos, self.init_qvel = qpos, qvel
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        if self.viewer is not None:
            self.viewer._run_speed = 0.5
            self.viewer.cam.trackbodyid = 0
            # self.viewer.cam.lookat[2] += .8
            self.viewer.cam.elevation = -25
            self.viewer.cam.type = 1
            self.sim.forward()
            self.viewer.cam.distance = self.model.stat.extent * 1.0
        # Make sure that the offscreen context has the same camera setup
        if self.sim._render_context_offscreen is not None:
            self.sim._render_context_offscreen.cam.trackbodyid = 0
            # self.sim._render_context_offscreen.cam.lookat[2] += .8
            self.sim._render_context_offscreen.cam.elevation = -25
            self.sim._render_context_offscreen.cam.type = 1
            self.sim._render_context_offscreen.cam.distance = \
                self.model.stat.extent * 1.0
        self.buffer_size = (1280, 800)

    def render(self, mode='human'):
        return super(SumoEnv, self).render(mode=mode) # just raise an exception