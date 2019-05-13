import gym
from LunarLander import *
from exp_setup import *

env = gym.make('LunarLander-v2')

# region Trainings
for epsilon_start in exp_epsilon_starts:
    for i in range(hyper_parameter_tuning_repeat):
        train_lunar_lander(env=env,
                           framework=default_framework,
                           hidden_layer_dimensions=default_network_size,
                           use_dropout=False,
                           training_episode_count=default_train_episode_count,
                           alpha=default_alpha,
                           gamma=default_gamma,
                           epsilon_start=epsilon_start,
                           epsilon_decay=default_epsilon_decay,
                           epsilon_min=default_epsilon_min,
                           replay_sample_size=default_replay_sample_size,
                           mean_reward_recency=default_mean_reward_recency,
                           model_saving_folder=epsilon_start_exp_dir)

for epsilon_decay in exp_epsilon_decays:
    for i in range(hyper_parameter_tuning_repeat):
        train_lunar_lander(env=env,
                           framework=default_framework,
                           hidden_layer_dimensions=[256, 128],
                           use_dropout=False,
                           training_episode_count=2000,
                           alpha=1e-4,
                           gamma=0.99,
                           epsilon_start=1.0,
                           epsilon_decay=epsilon_decay,
                           epsilon_min=0.0,
                           replay_sample_size=32,
                           mean_reward_recency=100,
                           model_saving_folder=epsilon_decay_exp_dir)

for epsilon_min in exp_epsilon_mins:
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
                           epsilon_min=epsilon_min,
                           replay_sample_size=default_replay_sample_size,
                           mean_reward_recency=default_mean_reward_recency,
                           model_saving_folder=epsilon_min_exp_dir)

for alpha in exp_alphas:
    for i in range(hyper_parameter_tuning_repeat):
        train_lunar_lander(env=env,
                           framework=default_framework,
                           hidden_layer_dimensions=default_network_size,
                           use_dropout=False,
                           training_episode_count=default_train_episode_count,
                           alpha=alpha,
                           gamma=default_gamma,
                           epsilon_start=default_epsilon_start,
                           epsilon_decay=default_epsilon_decay,
                           epsilon_min=default_epsilon_min,
                           replay_sample_size=default_replay_sample_size,
                           mean_reward_recency=default_mean_reward_recency,
                           model_saving_folder=alpha_exp_dir)

for gamma in exp_gammas:
    for i in range(hyper_parameter_tuning_repeat):
        train_lunar_lander(env=env,
                           framework=default_framework,
                           hidden_layer_dimensions=default_network_size,
                           use_dropout=False,
                           training_episode_count=default_train_episode_count,
                           alpha=default_alpha,
                           gamma=gamma,
                           epsilon_start=default_epsilon_start,
                           epsilon_decay=default_epsilon_decay,
                           epsilon_min=default_epsilon_min,
                           replay_memory_size=default_replay_memory_size,
                           replay_sample_size=default_replay_sample_size,
                           mean_reward_recency=default_mean_reward_recency,
                           model_saving_folder=gamma_exp_dir)

# endregion


# region Tests
for epsilon_start in exp_epsilon_starts:
    test_lunar_lander(env=env,
                      framework=default_framework,
                      hidden_layer_dimensions=default_network_size,
                      test_episode_count=default_test_episode_count,
                      alpha=default_alpha,
                      gamma=default_gamma,
                      epsilon_start=epsilon_start,
                      epsilon_decay=default_epsilon_decay,
                      epsilon_min=default_epsilon_min,
                      replay_memory_size=default_replay_memory_size,
                      replay_sample_size=default_replay_sample_size,
                      model_saving_folder=epsilon_start_exp_dir)

for epsilon_decay in exp_epsilon_decays:
    test_lunar_lander(env=env,
                      framework=default_framework,
                      hidden_layer_dimensions=default_network_size,
                      test_episode_count=default_test_episode_count,
                      alpha=default_alpha,
                      gamma=default_gamma,
                      epsilon_start=default_epsilon_start,
                      epsilon_decay=epsilon_decay,
                      epsilon_min=default_epsilon_min,
                      replay_memory_size=default_replay_memory_size,
                      replay_sample_size=default_replay_sample_size,
                      model_saving_folder=epsilon_decay_exp_dir)

for epsilon_min in exp_epsilon_mins:
    test_lunar_lander(env=env,
                      framework=default_framework,
                      hidden_layer_dimensions=default_network_size,
                      test_episode_count=default_test_episode_count,
                      alpha=default_alpha,
                      gamma=default_gamma,
                      epsilon_start=default_epsilon_start,
                      epsilon_decay=default_epsilon_decay,
                      epsilon_min=epsilon_min,
                      replay_memory_size=default_replay_memory_size,
                      replay_sample_size=default_replay_sample_size,
                      model_saving_folder=epsilon_min_exp_dir)

for alpha in exp_alphas:
    test_lunar_lander(env=env,
                      framework=default_framework,
                      hidden_layer_dimensions=default_network_size,
                      test_episode_count=default_test_episode_count,
                      alpha=alpha,
                      gamma=default_gamma,
                      epsilon_start=default_epsilon_start,
                      epsilon_decay=default_epsilon_decay,
                      epsilon_min=default_epsilon_min,
                      replay_memory_size=default_replay_memory_size,
                      replay_sample_size=default_replay_sample_size,
                      model_saving_folder=alpha_exp_dir)

for gamma in exp_gammas:
    test_lunar_lander(env=env,
                      framework=default_framework,
                      hidden_layer_dimensions=default_network_size,
                      test_episode_count=default_test_episode_count,
                      alpha=default_alpha,
                      gamma=gamma,
                      epsilon_start=default_epsilon_start,
                      epsilon_decay=default_epsilon_decay,
                      epsilon_min=default_epsilon_min,
                      replay_memory_size=default_replay_memory_size,
                      replay_sample_size=default_replay_sample_size,
                      model_saving_folder=gamma_exp_dir)
# endregion

