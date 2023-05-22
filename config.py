import argparse
from tokenize import group


def get_config():
    """
    The configuration parser for common hyperparameters of all environment.
    Please reach each `scripts/train/<env>_runner.py` file to find private hyperparameters
    only used in <env>.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser = _get_prepare_config(parser)
    parser = _get_replaybuffer_config(parser)
    parser = _get_network_config(parser)
    parser = _get_recurrent_config(parser)
    parser = _get_optimizer_config(parser)
    parser = _get_ppo_config(parser)
    parser = _get_selfplay_config(parser)
    parser = _get_pbt_config(parser)
    parser = _get_eval_config(parser)
    return parser


def _get_prepare_config(parser: argparse.ArgumentParser):
    """
    Prepare parameters:
        --env-name <str>
            specify the name of environment
        --algorithm-name <str>
            specifiy the algorithm, including `["ppo", "mappo"]`
        --experiment-name <str>
            an identifier to distinguish different experiment.
        --seed <int>
            set seed for numpy and torch
        --cuda
            by default False, will use CPU to train; or else will use GPU;
        --num-env-steps <float>
            number of env steps to train (default: 1e7)
        --model-dir <str>
            by default None. set the path to pretrained model.
        --use-wandb
            [for wandb usage], by default False, if set, will log date to wandb server.
        --user-name <str>
            [for wandb usage], to specify user's name for simply collecting training data.
        --wandb-name <str>
            [for wandb usage], to specify user's name for simply collecting training data.
    """
    group = parser.add_argument_group("Prepare parameters")
    group.add_argument("--env-name", type=str, default='SumoAnts-v0',
                       help="specify the name of environment")
    group.add_argument("--algorithm-name", type=str, default='ppo', choices=["ppo", "mappo"],
                       help="Specifiy the algorithm (default ppo)")
    group.add_argument("--experiment-name", type=str, default="check",
                       help="An identifier to distinguish different experiment.")
    group.add_argument("--seed", type=int, default=1,
                       help="Random seed for numpy/torch")
    group.add_argument("--cuda", action='store_true', default=False,
                       help="By default False, will use CPU to train; or else will use GPU;")
    group.add_argument("--num-env-steps", type=float, default=1e7,
                       help='Number of environment steps to train (default: 1e7)')
    group.add_argument("--model-dir", type=str, default=None,
                       help="By default None. set the path to pretrained model.")
    group.add_argument("--use-wandb", action='store_true', default=False,
                       help="[for wandb usage], by default False, if set, will log date to wandb server.")
    group.add_argument("--user-name", type=str, default='jyh',
                       help="[for wandb usage], to specify user's name for simply collecting training data.")
    group.add_argument("--wandb-name", type=str, default='jyh',
                       help="[for wandb usage], to specify user's name for simply collecting training data.")
    group.add_argument("--capture-video", action='store_true', default=False,
                       help="use wandb to capture video")
    return parser


def _get_replaybuffer_config(parser: argparse.ArgumentParser):
    """
    Replay Buffer parameters:
        --gamma <float>
            discount factor for rewards (default: 0.99)
        --buffer-size <int>
            the maximum storage in the buffer.
        --use-proper-time-limits
            by default, the return value does consider limits of time. If set, compute returns with considering time limits factor.
        --use-gae
            by default, use generalized advantage estimation. If set, do not use gae.
        --gae-lambda <float>
            gae lambda parameter (default: 0.95)
    """
    group = parser.add_argument_group("Replay Buffer parameters")
    group.add_argument("--gamma", type=float, default=0.99,
                       help='discount factor for rewards (default: 0.99)')
    group.add_argument("--buffer-size", type=int, default=200,
                       help="maximum storage in the buffer.")
    group.add_argument("--use-gae", action='store_false', default=True,
                       help='Whether to use generalized advantage estimation')
    group.add_argument("--gae-lambda", type=float, default=0.95,
                       help='gae lambda parameter (default: 0.95)')
    group.add_argument("--use-risk-sensitive", action='store_true', default=False)
    group.add_argument("--use-reward-hyper", action='store_true', default=False)
    group.add_argument("--tau-list", type=str, default="0.1 0.4 0.5 0.6 0.9")
    group.add_argument("--reward-list", type=str, default="0.05 0.2 1.0 5.0 20.0")
    return parser


def _get_network_config(parser: argparse.ArgumentParser):
    """
    Network parameters:
        --hidden-size <str>
            dimension of hidden layers for mlp pre-process networks
        --act-hidden-size <int>
            dimension of hidden layers for actlayer
        --activation-id
            choose 0 to use Tanh, 1 to use ReLU, 2 to use LeakyReLU, 3 to use ELU
        --use-feature-normalization
            by default False, otherwise apply LayerNorm to normalize feature extraction inputs.
        --gain
            by default 0.01, use the gain # of last action layer
    """
    group = parser.add_argument_group("Network parameters")
    group.add_argument("--hidden-size", type=str, default='128 128',
                       help="Dimension of hidden layers for mlp pre-process networks (default '128 128')")
    group.add_argument("--act-hidden-size", type=str, default='128 128',
                       help="Dimension of hidden layers for actlayer (default '128 128')")
    group.add_argument("--activation-id", type=int, default=1,
                       help="Choose 0 to use Tanh, 1 to use ReLU, 2 to use LeakyReLU, 3 to use ELU (default 1)")
    group.add_argument("--use-feature-normalization", action='store_true', default=False,
                       help="Whether to apply LayerNorm to the feature extraction inputs")
    group.add_argument("--gain", type=float, default=0.01,
                       help="The gain # of last action layer")
    group.add_argument("--use-cnn", action='store_true', default=False,
                       help="Whether to use cnn")
    return parser


def _get_recurrent_config(parser: argparse.ArgumentParser):
    """
    Recurrent parameters:
        --use-recurrent-policy
            by default, use Recurrent Policy. If set, do not use.
        --recurrent-hidden-size <int>
            Dimension of hidden layers for recurrent layers (default 128).
        --recurrent-hidden-layers <int>
            The number of recurrent layers (default 1).
        --data-chunk-length <int>
            Time length of chunks used to train a recurrent_policy, default 10.
    """
    group = parser.add_argument_group("Recurrent parameters")
    group.add_argument("--use-recurrent-policy", action='store_false', default=True,
                       help='Whether to use a recurrent policy')
    group.add_argument("--recurrent-hidden-size", type=int, default=128,
                       help="Dimension of hidden layers for recurrent layers (default 128)")
    group.add_argument("--recurrent-hidden-layers", type=int, default=1,
                       help="The number of recurrent layers (default 1)")
    group.add_argument("--data-chunk-length", type=int, default=10,
                       help="Time length of chunks used to train a recurrent_policy (default 10)")
    return parser


def _get_optimizer_config(parser: argparse.ArgumentParser):
    """
    Optimizer parameters:
        --lr <float>
            learning rate parameter (default: 3e-4, fixed).
    """
    group = parser.add_argument_group("Optimizer parameters")
    group.add_argument("--lr", type=float, default=3e-4,
                       help='learning rate (default: 3e-4)')
    return parser


def _get_ppo_config(parser: argparse.ArgumentParser):
    """
    PPO parameters:
        --ppo-epoch <int>
            number of ppo epochs (default: 10)
        --clip-param <float>
            ppo clip parameter (default: 0.2)
        --use-clipped-value-loss
            by default false. If set, clip value loss.
        --num-mini-batch <int>
            number of batches for ppo (default: 1)
        --value-loss-coef <float>
            ppo value loss coefficient (default: 1)
        --entropy-coef <float>
            ppo entropy term coefficient (default: 0.01)
        --use-max-grad-norm
            by default, use max norm of gradients. If set, do not use.
        --max-grad-norm <float>
            max norm of gradients (default: 0.5)
    """
    group = parser.add_argument_group("PPO parameters")
    group.add_argument("--ppo-epoch", type=int, default=4,
                       help='number of ppo epochs (default: 4)')
    group.add_argument("--clip-param", type=float, default=0.2,
                       help='ppo clip parameter (default: 0.2)')
    group.add_argument("--use-clipped-value-loss", action='store_true', default=False,
                       help="By default false. If set, clip value loss.")
    group.add_argument("--num-mini-batch", type=int, default=5,
                       help='number of batches for ppo (default: 5)')
    group.add_argument("--value-loss-coef", type=float, default=1,
                       help='ppo value loss coefficient (default: 1)')
    group.add_argument("--entropy-coef", type=float, default=0.01,
                       help='entropy term coefficient (default: 0.01)')
    group.add_argument("--use-max-grad-norm", action='store_false', default=True,
                       help="By default, use max norm of gradients. If set, do not use.")
    group.add_argument("--max-grad-norm", type=float, default=2,
                       help='max norm of gradients (default: 2)')
    return parser


def _get_selfplay_config(parser: argparse.ArgumentParser):
    """
    Selfplay parameters:
        --use-selfplay
            by default false. If set, use selfplay algorithms.
        --selfplay-algorithm <str>
            specifiy the selfplay algorithm, including `["sp", "fsp"]`
        --n-choose-opponents <int>
            number of different opponents chosen for rollout. (default 1)
        --init-elo <float>
            initial ELO for policy performance. (default 1000.0)
    """
    group = parser.add_argument_group("Selfplay parameters")
    group.add_argument("--use-selfplay", action='store_true', default=False,
                       help="By default false. If set, use selfplay algorithms.")
    group.add_argument("--selfplay-algorithm", type=str, default="fsp", help="selfplay algorithm")
    group.add_argument("--random-side", action='store_true', default=False, help="random agent side when env reset")
    group.add_argument('--init-elo', type=float, default=1000.0,
                       help="initial ELO for policy performance. (default 1000.0)")
    return parser

def _get_pbt_config(parser: argparse.ArgumentParser):
    """
    PBT parameters:
        --population-size
            by default 1
        --num-parallel-each-agent 
            by default 4
    """
    group = parser.add_argument_group("PBT parameters")
    group.add_argument("--population-size", type=int, default=1,
                       help="number of agents in the population")
    group.add_argument("--num-parallel-each-agent", type=int, default=1,
                       help="number of subprocesses for each agent")
    group.add_argument("--exploit-elo-threshold", type=int, default=500,
                       help="")
    return parser

def _get_eval_config(parser: argparse.ArgumentParser):
    """
    Eval parameters:
        --eval-episodes <int>
            number of episodes of a single evaluation.
    """
    group = parser.add_argument_group("Eval parameters")
    group.add_argument("--eval-episodes", type=int, default=1,
                       help="number of episodes of a single evaluation. (default 1)")
    return parser


if __name__ == "__main__":
    parser = get_config()
    all_args = parser.parse_args()
