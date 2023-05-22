from gym.envs.registration import register
from envs.slimevolley.slimevolley_wrapper import VolleyballEnv

register(
    id="ToyEnv-v0",
    entry_point="envs.toy:ToyEnv",
    kwargs={'wind':True, 'onehot_state':True},
)

register(
    id='SlimeVolley-v0',
    entry_point='envs:VolleyballEnv'
)


register(
    id='SumoAnts-v0',
    entry_point='envs.robosumo.envs:SumoEnv',
    kwargs={
        'agent_names': ['ant', 'ant'],
        'agent_densities': [13., 13.],
        'tatami_size': 2.0,
        'timestep_limit': 500,
        'reward_shape': 1. 
    },
)