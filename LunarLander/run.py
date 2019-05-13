import gym
from LunarLander import *
from exp_setup import *

env = gym.make('LunarLander-v2')
test_lunar_lander(env=env, model_saving_folder=exp_root_dir, model_file='best.mod', render=True,
                  framework=default_framework,
                  hidden_layer_dimensions=default_network_size,
                  test_episode_count=default_test_episode_count,
                  alpha=default_alpha,
                  gamma=default_gamma,
                  epsilon_start=default_epsilon_start,
                  epsilon_decay=default_epsilon_decay,
                  epsilon_min=default_epsilon_min,
                  replay_memory_size=default_replay_memory_size,
                  replay_sample_size=default_replay_sample_size)
