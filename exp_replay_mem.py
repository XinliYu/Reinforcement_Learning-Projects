import gym
from exp_setup import *
from LunarLander import *

env = gym.make('LunarLander-v2')
attempt_per_trial = 1
for mem_size in exp_replay_memory_sizes:
    for i in range(hyper_parameter_tuning_repeat):
        train_lunar_lander(env=env,
                           framework=default_framework,
                           hidden_layer_dimensions=default_network_size,
                           use_dropout=False,
                           training_episode_count=default_train_episode_count,
                           alpha=default_alpha,
                           gamma=default_gamma,
                           epsilon_start=default_epsilon_start,
                           epsilon_decay=default_epsilon_decay,
                           epsilon_min=default_epsilon_min,
                           replay_memory_size=mem_size,
                           replay_sample_size=default_replay_sample_size,
                           training_start_memory_size=default_replay_sample_size + default_replay_sample_size,
                           mean_reward_recency=100,
                           model_saving_folder=memory_size_exp_dir)

for sample_size in (128,):
    for i in range(hyper_parameter_tuning_repeat):
        train_lunar_lander(env=env,
                           framework=default_framework,
                           hidden_layer_dimensions=default_network_size,
                           use_dropout=False,
                           training_episode_count=2000,
                           alpha=default_alpha,
                           gamma=default_gamma,
                           epsilon_start=default_epsilon_start,
                           epsilon_decay=default_epsilon_decay,
                           epsilon_min=default_epsilon_min,
                           replay_sample_size=sample_size,
                           mean_reward_recency=100,
                           model_saving_folder=sample_size_exp_dir)

