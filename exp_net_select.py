import gym
from LunarLander import *
from exp_setup import *

env = gym.make('LunarLander-v2')

# training for network selection
for network_size in exp_network_sizes:
    for i in range(network_selection_training_repeat):
        train_lunar_lander(env=env,
                           framework=default_framework,
                           hidden_layer_dimensions=network_size,
                           use_dropout=False,
                           training_episode_count=default_train_episode_count + 500,
                           alpha=default_alpha,
                           gamma=default_gamma,
                           epsilon_start=default_epsilon_start,
                           epsilon_decay=default_epsilon_decay,
                           epsilon_min=default_epsilon_min,
                           replay_sample_size=default_replay_sample_size,
                           mean_reward_recency=100,
                           model_saving_folder=netsize_exp_dir)

# training the default network, but disable the experience replay
for i in range(network_selection_training_repeat):
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
                       replay_memory_size=default_replay_sample_size,  # this disables experience replay
                       replay_sample_size=default_replay_sample_size,
                       training_start_memory_size=default_replay_sample_size,
                       mean_reward_recency=100,
                       model_saving_folder=netsize_exp_dir)

# test for network selection
for network_size in exp_network_sizes:
    test_lunar_lander(env=env,
                      framework=default_framework,
                      repeat=default_test_repeat,
                      hidden_layer_dimensions=network_size,
                      test_episode_count=default_test_episode_count,
                      alpha=default_alpha,
                      gamma=default_gamma,
                      epsilon_decay=default_epsilon_decay,
                      epsilon_min=default_epsilon_min,
                      replay_memory_size=default_replay_memory_size,
                      replay_sample_size=default_replay_sample_size,
                      model_saving_folder=netsize_exp_dir)

# test default network trained without experience replay
test_lunar_lander(env=env,
                  framework=default_framework,
                  hidden_layer_dimensions=default_network_size,
                  test_episode_count=default_test_episode_count,
                  alpha=default_alpha,
                  gamma=default_gamma,
                  epsilon_start=default_epsilon_start,
                  epsilon_decay=default_epsilon_decay,
                  epsilon_min=default_epsilon_min,
                  replay_memory_size=default_replay_sample_size,
                  replay_sample_size=default_replay_sample_size,
                  model_saving_folder=netsize_exp_dir)
