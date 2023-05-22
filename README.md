# RPBT

This is the source code of RPBT, which is proposed in the paper "**Learning Diverse Risk Preferences in Population-based Self-play"**(http://arxiv.org/abs/2305.11476)。This repository provide a single file implementation of **RPPO(Risk-sensitive PPO)** in `toyexample/rppo.py`, and a lightweight, scalable implementation of **RPBT(Population based self-play with RPPO)**. All the experiments were conducted with one AMD EPYC 7702 64-Core Processor and one GeForce RTX 3090 GPU.

## 1. Environment supported

- single-agent setting
  - Toy example in the paper
  - classic gym environment

- multi-agent competitive setting

  - [Slimevolley](https://github.com/hardmaru/slimevolleygym)
  - [Sumoants](https://github.com/openai/robosumo )

  The videos are available at https://sites.google.com/view/rpbt.

## 2. Installation

We use `python 3.8`

```
pip install -r requirements.txt
```



## 3. Usage

### 3.1 toy example

We provide a single file implementation of RPPO for toyexample.

run

```python
python toyexample/rppo.py --env-id ToyEnv-v0 --num-steps 200 --tau 0.2
```

`--tau` is the value of risk level $\tau$ in the paper. If we set `--risk False`, we recover the original PPO.

### 3.2 multi-agent competitive setting

The hyperparameter configs are in `config.py`. We provide 2 training scripts：

For Slimevolley, `bash train_vb.sh`

For Sumoants,  `bash train_sumo.sh`

If not to use PBT, set `--population-size 1 `, we recover the RPPO.

### 4 Acknowledgement

We appreciate the following repos for their valuable code base implementations:

- [CleanRL](https://github.com/vwxyzjn/cleanrl)

- [MAPPO](https://github.com/marlbenchmark/on-policy)





 

