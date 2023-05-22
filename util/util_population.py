import os
import random
import shutil
import numpy as np
from copy import deepcopy


def population_based_exploit_explore(epoch: int, elos: dict, hypers: dict, run_dir: str, elo_threshold):
    topk = int(np.ceil(0.2 * len(elos)))
    elos_new = deepcopy(elos)
    hypers_new = deepcopy(hypers)
    ranks = {agent_id: elos[agent_id][epoch+1] for agent_id in elos.keys()}
    sorted_ranks_idxs = [k for k in sorted(ranks, key=ranks.__getitem__, reverse=True)]
    topk_idxs = list(sorted_ranks_idxs[:topk])
    for agent_id in elos_new.keys():
        agent_elo = elos[agent_id][epoch+1]
        better_agent_id = random.sample(topk_idxs, 1)[0]
        if len(sorted_ranks_idxs) == 1:
            # population size = 1, no exploit
            break
        if ranks[better_agent_id] - agent_elo < elo_threshold or agent_id in topk_idxs:
            # the agent is already good enough
            continue
        elos_new[agent_id][epoch+1] = elos[better_agent_id][epoch+1]
        os.remove(f"{run_dir}/agent{agent_id}_history{epoch+1}.pt")
        os.remove(f"{run_dir}/agent{agent_id}_latest.pt")
        shutil.copy(f"{run_dir}/agent{better_agent_id}_latest.pt", f"{run_dir}/agent{agent_id}_history{epoch+1}.pt")
        shutil.copy(f"{run_dir}/agent{better_agent_id}_latest.pt", f"{run_dir}/agent{agent_id}_latest.pt")
        for s in hypers[agent_id].keys():
            hyper_ = hypers[agent_id][s]
            new_hyper_ = hypers[better_agent_id][s]
            inherit_prob = np.random.binomial(1, 0.5, 1)[0]
            hyper_tmp = (1. - inherit_prob) * hyper_ + inherit_prob * new_hyper_
            hypers_new[agent_id][s] = np.clip(hyper_tmp + np.random.uniform(-0.2, 0.2), 0., 1.)
    return elos_new, hypers_new
