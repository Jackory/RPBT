exp_name="test"
env_name="SlimeVolley-v0"
seed=2023

CUDA_VISIBLE_DEVICES=0 python main_pbt_selfplay.py --env-name ${env_name} --experiment-name ${exp_name} --seed ${seed} \
--num-env-steps 1e8 --buffer-size 6000 \
-gamma 0.995 \
--cuda \
--population-size 5 --num-parallel-each-agent 16  --exploit-elo-threshold 500 \
--use-risk-sensitive --tau-list "0.1 0.4 0.5 0.6 0.9" \
--use-wandb
