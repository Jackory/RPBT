import numpy as np
from gym import Env, spaces
from gym.utils import seeding
from contextlib import closing
from io import StringIO
import sys

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
DELTA = np.array([
    (-1, 0),
    (0, 1),
    (1, 0),
    (0, -1)
])

class ToyEnv(Env):
    def __init__(self, wind = True, onehot_state=True) -> None:
        self.onehot_state = onehot_state
        self.shape = (4, 4)
        self.nS = np.prod(self.shape)
        self.nA = 4
        # self.observation_space = spaces.Discrete(self.nS)
        self.observation_space = spaces.Box(low=0, high=1, shape=(16,)) if self.onehot_state else spaces.Discrete(self.nS)
        self.action_space = spaces.Discrete(self.nA)

        self.use_wind = wind
        self._wind = np.ones(self.shape)
        self._water = np.zeros(self.shape)
        self._water[3, :] = 1


        self.start = (2, 0)
        self.terminate = (2, 3)
        self.seed()
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def reset(self):
        self.visited = np.zeros(self.shape)
        self.current_step = 0
        self.s = np.array(self.start)
        self.visited[tuple(self.s)] += 1
        state = np.ravel_multi_index(self.s, self.shape)
        if self.onehot_state:
            state = np.eye(self.nS)[state]
        return state
    
    def step(self, action):
        self.current_step += 1
        reward = 0
        done = False

        if self.use_wind and self._wind[tuple(self.s)]:
            if self.np_random.random() >= 0.5:
                action = self.np_random.choice([UP, RIGHT, DOWN, LEFT])
        new_pos1 = self.s + DELTA[action]
        new_pos1 = self._limit_coordinates(new_pos1).astype(int)

        if tuple(new_pos1) == self.terminate:
            reward = 1
            done = True
        elif self._water[tuple(new_pos1)]:
            reward = -1
        if self.current_step >= 25:
            done = True

        self.visited[tuple(new_pos1)] += 1
        self.s = new_pos1
        state = np.ravel_multi_index(self.s, self.shape)
        if self.onehot_state:
            state = np.eye(self.nS)[state]
        return state, reward, done, {"step": self.current_step}
    
    def render(self, mode="human"):
        outfile = StringIO() if mode == "ansi" else sys.stdout

        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            if tuple(self.s) == position:
                output = " x "
            # Print terminal state
            elif position == self.terminate:
                output = " T "
            elif self._water[position]:
                output = " W "
            else:
                output = " o "

            if position[1] == 0:
                output = output.lstrip()
            if position[1] == self.shape[1] - 1:
                output = output.rstrip()
                output += "\n"

            outfile.write(output)
        outfile.write("\n")

        # No need to return anything for human
        if mode != "human":
            with closing(outfile):
                return outfile.getvalue()
        

    def _limit_coordinates(self, coord):
        """
        Prevent the agent from falling out of the grid world
        :param coord:
        :return:
        """
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord


if __name__ == "__main__":
    env = ToyEnv()
    env.seed(0)
    env.action_space.seed(0)
    state_freq = np.zeros(env.shape)
    s = env.reset()
    env.render()
    pos = np.unravel_index(s, env.shape)
    while True:
        a = env.action_space.sample()
        s,r,done,info = env.step(a)    
        env.render()
        pos = np.unravel_index(s, env.shape)
        if done:
            state_freq += env.visited
            print(np.unravel_index(s, env.shape))
            print(info)
            break
    print(state_freq)   