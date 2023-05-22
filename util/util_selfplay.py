from typing import Dict, List, Tuple
import numpy as np
import random
from abc import ABC, abstractstaticmethod

def get_algorithm(algo_name):
    if algo_name == 'sp':
        return SP()
    elif algo_name == 'fsp':
        return FSP()
    elif algo_name == 'pfsp':
        return PFSP()
    else:
        raise NotImplementedError("Unknown algorithm {}".format(algo_name))


class SelfplayAlgorithm(ABC):
    def __init__(self) -> None:
        pass

    def choose_opponents(self, agent_idx: int, agent_elos: dict, num_opponents: int):
        enm_idxs, enm_history_idxs, enm_elos = [], [], []
        num_total = 1
        while True:
            enm_idx = random.choice([k for k in list(agent_elos.keys())])
            # 1) choose the opponent agent from populations.
            enm_idxs.append(enm_idx)
            # 2) choose the history copy from the current agent according to ELO
            enm_history_idx = self._choose_history(agent_elos[enm_idx])
            enm_history_idxs.append(enm_history_idx)
            enm_elos.append(agent_elos[enm_idx][enm_history_idx])
            num_total += 1
            if num_total > num_opponents:
                break
        enms = []
        for agent, itr in zip(enm_idxs, enm_history_idxs):
                enms.append((agent, itr))
        return enms, enm_elos
    
    @abstractstaticmethod
    def _choose_history(self, agents_elo: Dict[str, float]):
        pass

class SP(SelfplayAlgorithm):
    def __init__(self) -> None:
        super().__init__()
    
    def _choose_history(self, agents_elo: Dict[str, float]):
        return list(agents_elo.keys())[-1]


class FSP(SelfplayAlgorithm):
    def __init__(self) -> None:
         super().__init__()
    
    def _choose_history(self, agents_elo: Dict[str, float]):
        return np.random.choice(list(agents_elo.keys()))


class PFSP(SelfplayAlgorithm):
    def __init__(self) -> None:
        super().__init__()
        self.lam = 1.
        self.s = 100.
    
    def _choose_history(self, agents_elo: Dict[str, float]):
        history_elo = np.array(list(agents_elo.values()))
        sample_probs = 1. / (1. + 10. ** (-(history_elo - np.median(history_elo)) / 400.)) * self.s
        """ meta-solver """
        k = float(len(sample_probs) + 1)
        meta_solver_probs = np.exp(self.lam / k * sample_probs) / np.sum(np.exp(self.lam / k * sample_probs))
        opponent_idx = np.random.choice(a=list(agents_elo.keys()), size=1, p=meta_solver_probs).item()
        return opponent_idx


if __name__ == '__main__':
    ranks = {
        0: {0: 1000, 1: 1200, 2: 1300, 3: 1400, 4: 1500, 5: 1800, 6: 2000},
        1: {0: 1000, 1: 1200, 2: 1300, 3: 1400, 4: 1500, 5: 1800, 6: 2000},
        2: {0: 1000, 1: 1200, 2: 1300, 3: 1400, 4: 1500, 5: 1800, 6: 2000},
        3: {0: 1000, 1: 1200, 2: 1300, 3: 1400, 4: 1500, 5: 1800, 6: 2000},
    }

    algo = get_algorithm("pfsp")
    ret = algo.choose_opponents(agent_idx=0, agent_elos=ranks, num_opponents=4)

    print(ret[0], ret[1])
