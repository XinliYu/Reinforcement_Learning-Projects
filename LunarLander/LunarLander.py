from typing import List
from time import time
import torch
import torch.nn as nn
import torch.nn.functional as nnF
from Utilities import *
from QLearner import QLearner, QFun
from FeedForward import FeedForwardTorch
from Utilities import _get_file_path


# implements the lunar lander agent

def _get_dimensions(env, hidden_layer_dimensions):
    state_dim = env.observation_space.shape[0]
    action_count = env.action_space.n
    dimensions = [state_dim] + hidden_layer_dimensions + [action_count]
    return dimensions


def _get_qfun(framework: str, dimensions, learning_rate, model_path=None, use_dropout=False):
    if framework.startswith("pytorch"):
        ffw = FeedForwardTorch(dimensions=dimensions,
                               activations=[(nn.BatchNorm1d if use_dropout else nn.ReLU) for _ in range(len(dimensions) - 2)] + [None],
                               loss_fun=nnF.mse_loss,
                               optimizer=lambda mode_paras: torch.optim.Adam(mode_paras, lr=learning_rate, betas=(0.9, 0.98), eps=1e-8))
        if framework.endswith("_cuda") and torch.cuda.is_available():
            device = torch.device("cuda:0")
            ffw.cuda(device)
            q_fun = QFun(model=ffw,
                         eval_switch=ffw.eval,
                         list_to_tensor=torch.cuda.FloatTensor,
                         nparray_to_tensor=lambda x: torch.from_numpy(x).to(device),
                         argmax=torch.argmax)
        else:
            q_fun = QFun(model=ffw,
                         eval_switch=ffw.eval,
                         list_to_tensor=torch.Tensor,
                         nparray_to_tensor=torch.from_numpy,
                         argmax=torch.argmax)
    else:
        raise ValueError("Framework '{}' is not supported. Use 'pytorch' or 'pytorch_cuda'".format(framework))

    if model_path:
        ffw.load(path=model_path)

    return q_fun


def train_lunar_lander(env, framework="pytorch", hidden_layer_dimensions: List[int] = [128, 64],
                       use_dropout=False,
                       training_episode_count=2000,
                       alpha=1e-4, gamma=0.99,
                       epsilon_start=1.0, epsilon_decay=0.998, epsilon_min=0.0,
                       replay_memory_size=2 ** 16, replay_sample_size=32, training_start_memory_size=64,
                       mean_reward_recency=100, model_saving_folder=join('.', "models")):
    dimensions = _get_dimensions(env, hidden_layer_dimensions)
    q_fun = _get_qfun(framework, dimensions, alpha, use_dropout=use_dropout)

    lunar_lander = QLearner(env=env,
                            q_fun=q_fun,
                            epsilon_decay=epsilon_decay,
                            epsilon_min=epsilon_min,
                            gamma=gamma)

    mean_reward, logs = lunar_lander.train(episode_count=training_episode_count,
                                           epsilon_start=epsilon_start,
                                           replay_memory=replay_memory_size,
                                           replay_sample_size=replay_sample_size,
                                           training_start_memory_size=training_start_memory_size,
                                           mean_reward_recency=mean_reward_recency)

    file_path_no_exname, _ = _get_file_path(folder_path=model_saving_folder,
                                            framework=framework,
                                            extension_name='.mod',
                                            hidden_layer_dimensions=hidden_layer_dimensions,
                                            mean_reward=mean_reward,
                                            alpha=alpha,
                                            gamma=gamma,
                                            epsilon_start=epsilon_start,
                                            epsilon_decay=epsilon_decay,
                                            epsilon_min=epsilon_min,
                                            replay_memory_size=replay_memory_size,
                                            replay_sample_size=replay_sample_size,
                                            use_dropout=use_dropout,
                                            timestamp=int(time()))

    q_fun.model.save(file_path_no_exname + '.mod')
    logs.to_csv(file_path_no_exname + '.csv', index=False)


def test_lunar_lander(env, framework="pytorch", repeat=10,
                      hidden_layer_dimensions: List[int] = [128, 64],
                      test_episode_count=100,
                      alpha=1e-4, gamma=0.99,
                      epsilon_start=1.0, epsilon_decay=0.998, epsilon_min=0.0,
                      replay_memory_size=2 ** 16, replay_sample_size=32,
                      model_saving_folder=join('.', "models"), model_file=None, render=False):
    if model_file:
        files = [join(model_saving_folder, model_file)]
    else:
        _, files = _get_file_path(folder_path=model_saving_folder,
                                  framework=framework,
                                  extension_name='.mod',
                                  hidden_layer_dimensions=hidden_layer_dimensions,
                                  mean_reward=None,
                                  alpha=alpha,
                                  gamma=gamma,
                                  epsilon_start=epsilon_start,
                                  epsilon_decay=epsilon_decay,
                                  epsilon_min=epsilon_min,
                                  replay_memory_size=replay_memory_size,
                                  replay_sample_size=replay_sample_size,
                                  use_dropout=False,
                                  timestamp=None)
    if files:
        dimensions = _get_dimensions(env, hidden_layer_dimensions)
        logs = True
        for file in files:
            mean_rewards = [None] * repeat
            start_episode_idx = 0
            print("==========test model '{}'==========".format(basename(file)))
            for repeat_idx in range(repeat):
                q_fun = _get_qfun(framework, dimensions, alpha, file)

                lunar_lander = QLearner(env=env,
                                        q_fun=q_fun,
                                        epsilon_decay=epsilon_decay,
                                        epsilon_min=epsilon_min,
                                        gamma=gamma)

                mean_reward, logs = lunar_lander.test(start_episode_idx=start_episode_idx,
                                                      episode_count=test_episode_count,
                                                      continued_learning=False,
                                                      logging=logs,
                                                      render=render)
                start_episode_idx += test_episode_count
                mean_rewards[repeat_idx] = mean_reward

            logs.to_csv(get_path_without_extension(file) + '_test{}_tmr{}.csv'.format(int(time()), int(sum(mean_rewards) / repeat)), index=False)

