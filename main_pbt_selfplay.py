import numpy as np
import torch
import ray
import time
import os
import sys
import gym
import random
import logging
import wandb
import setproctitle

from config import get_config
import envs
from pathlib import Path
from ppo.ppo_data_collectors import DataCollectorMix, make_env
from ppo.ppo_trainer import PBTPPOTrainer, PPOTrainer
from ppo.ppo_policy import PPOPolicy
from util.util_population import population_based_exploit_explore
from util.util_selfplay import get_algorithm



'''
Population: N agents

    for each agent, we have M data-collector, 1 trainer:

    for each trainning step t:
        for each agent i:
            1. select K opponent to collect S samples
            2. run_results = data-collector.collect.remote(ego_model, enm_model, hypyer_param)
                2.1 load model
                2.2 collect data via inner environment
                2.3 save data into inner PPOBuffer
                2.4 return PPOBuffer
        3. buffer_list = [[ray.get(run_results) for _ in range(M)] for _ in range(N)]
        for each agent i:
            4. info = trainer.update.remote(ego_model, buffer_list, hyper_param)
                4.1 load model
                4.2 ppo update with buffer data
                4.3 save model to file
                4.4 return trainning info
        5. info = [[ray.get(info) for _ in range(M)] for _ in range(N)]
        6. eval + elo_update
        7. population exploit
'''


def load_enm_params(run_dir: str, enm_idx: tuple):
    agent_id, t = enm_idx
    return torch.load(f"{run_dir}/agent{agent_id}_history{t}.pt", map_location=torch.device('cpu'))

def main(args):
    # init config
    parser = get_config()
    all_args = parser.parse_known_args(args)[0]
    random.seed(all_args.seed)
    np.random.seed(all_args.seed)
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed(all_args.seed)

    tau_hyper = [float(tau) for tau in all_args.tau_list.split(' ')]
    reward_hyper = [float(r) for r in all_args.reward_list.split(' ')]
    setproctitle.setproctitle(str(all_args.env_name)+'@'+ str(all_args.user_name))
    env = make_env(all_args.env_name)

    str_time = time.strftime("%b%d-%H%M%S", time.localtime())
    run_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "runs" / all_args.env_name / all_args.experiment_name
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    if all_args.use_wandb:
        wandb.init(
            project=all_args.env_name, 
            entity=all_args.wandb_name, 
            name=all_args.experiment_name, 
            group=all_args.experiment_name,
            job_type='charts',
            config=all_args,
            dir=str(run_dir))
        s = str(wandb.run.dir).split('/')[:-1]
        s.append('models')
        save_dir= '/'.join(s)
    else:
        run_dir = run_dir / str_time
        save_dir = str(run_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    all_args.save_dir = save_dir
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(save_dir + "_ouput.log"),
            logging.StreamHandler()
        ]
    )

    # init population
    ray.init()
    selfplay_algo = get_algorithm(all_args.selfplay_algorithm)
    population = {}
    population_elos = {}
    population_hypers = {}
    for agent_id in range(all_args.population_size):
        population[agent_id] = PPOPolicy(all_args, env.observation_space, env.action_space)
        if all_args.model_dir is not None:
            population[agent_id].restore_from_path(all_args.model_dir, epoch='latest')
        params = population[agent_id].params()
        torch.save(params, f'{save_dir}/agent{agent_id}_history0.pt')
        torch.save(params, f'{save_dir}/agent{agent_id}_latest.pt')
        population_elos[agent_id] = {0: all_args.init_elo}
        population_hypers[agent_id] = {}
        if all_args.use_risk_sensitive:
            population_hypers[agent_id]['tau'] = tau_hyper[agent_id]
        if all_args.use_reward_hyper:
            population_hypers[agent_id]['reward'] = reward_hyper[agent_id]
    
    M = all_args.num_parallel_each_agent
    N = all_args.population_size
    data_collector_pools = []
    for agent_id in range(N):
        data_collectors = []
        for i in range(M):
            all_args.env_id = agent_id * M + i
            data_collectors.append(DataCollectorMix.remote(all_args))
        data_collector_pools.append(data_collectors)
    ppo_trainers = [PBTPPOTrainer.remote(all_args, env.observation_space, env.action_space) for _ in range(N)]

    logging.info("Init over.")
    num_epochs = int(all_args.num_env_steps // M // all_args.buffer_size)

    for epoch in range(num_epochs):
        cur_steps = epoch * all_args.buffer_size * M
        # data collect
        data_results = []
        for agent_id in range(N):
            enm_idxs, enm_elos = selfplay_algo.choose_opponents(agent_id, population_elos, M)
            ego_model = population[agent_id].params(device='cpu')
            results = []
            for i in range(M):  
                enm_model = load_enm_params(save_dir, enm_idxs[i])
                res = data_collector_pools[agent_id][i].collect_data.remote(
                    ego_params=ego_model, 
                    enm_params=enm_model, 
                    hyper_params=population_hypers[agent_id]
                )
                results.append(res)
            data_results.append(results)
        
        buffers = [[ray.get(data_results[agent_id][i]) for i in range(M)] for agent_id in range(N)]
        # ppo train
        train_results = []
        for agent_id in range(N):
            ego_model = population[agent_id].params(device='cuda')
            res = ppo_trainers[agent_id].train.remote(
                buffer=buffers[agent_id], 
                params=ego_model, 
                hyper_params=population_hypers[agent_id]
            )
            train_results.append(res)
        train_infos = [ray.get(train_results[i]) for i in range(N)]
        for agent_id, (param, info) in enumerate(train_infos):
            population[agent_id].restore_from_params(param)
            torch.save(param, f'{save_dir}/agent{agent_id}_history{epoch+1}.pt')
            torch.save(param, f'{save_dir}/agent{agent_id}_latest.pt')
            train_reward = info["episode_reward"]
            logging.info(f"Epoch {epoch} / {num_epochs}, Agent{agent_id}, train episode reward {train_reward}")
            if all_args.use_wandb and agent_id == 0:
                for k, v in info.items():
                    wandb.log({k: v}, step=cur_steps)

        # evaluate and update elo
        eval_results = []
        enm_infos = {}
        for agent_id in range(N):
            enm_idxs, enm_elos = selfplay_algo.choose_opponents(agent_id, population_elos, M)
            enm_infos[agent_id] = enm_idxs
            ego_model = population[agent_id].params(device='cpu')
            results = []
            for i in range(M):  
                enm_model = load_enm_params(save_dir, enm_idxs[i])
                res = data_collector_pools[agent_id][i].evaluate_data.remote(
                    ego_params=ego_model, 
                    enm_params=enm_model, 
                    hyper_params=population_hypers[agent_id],
                    ego_elo=population_elos[agent_id][epoch],
                    enm_elo=enm_elos[i],
                )
                results.append(res)
            eval_results.append(results)
        
        eval_datas = [[ray.get(eval_results[agent_id][i]) for i in range(M)] for agent_id in range(N)]
        for agent_id in range(N):
            elo_gains, eval_infos = list(zip(*eval_datas[agent_id]))
            population_elos[agent_id][epoch+1] = population_elos[agent_id][epoch]
            for i, (enm_id, enm_t) in enumerate(enm_infos[agent_id]):
                population_elos[agent_id][epoch+1] += elo_gains[i]
                population_elos[enm_id][enm_t] -= elo_gains[i]
            eval_reward = np.mean([info['episode_reward'] for info in eval_infos])
            logging.info(f"Epoch {epoch} / {num_epochs}, Agent{agent_id}, eval episode reward {eval_reward}")
            if all_args.use_wandb and agent_id == 0:
                wandb.log({"eval_episode_reward": eval_reward}, step=cur_steps)

        # exploit and explore
        population_elos, population_hypers = population_based_exploit_explore(epoch, population_elos, population_hypers, save_dir, all_args.exploit_elo_threshold)
        # save checkoutpoint
        checkpoint = {
            "epoch": epoch,
            "population_elos": population_elos,
            "population_hypers": population_hypers
        }
        if all_args.use_wandb:
            for agent_id in range(N):
                wandb.log({f"agent{agent_id}_tau": population_hypers[agent_id]['tau']}, step=cur_steps)
                wandb.log({f"agent{agent_id}_elo": population_elos[agent_id][epoch]}, step=cur_steps)
        torch.save(checkpoint, f"{save_dir}/checkpoint_latest.pt")
        
if __name__ == "__main__":
    main(sys.argv[1:])

    

    
