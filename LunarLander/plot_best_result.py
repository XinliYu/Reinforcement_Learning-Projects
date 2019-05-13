from exp_setup import *
import matplotlib.pyplot as plt
import numpy as np
from Utilities import _get_file_path, read_mean_rewards_for_plot


def plot_best_result(files, result_name):
    train_mr, test_mr, train_re_trend, test_re_trend, train_mr_trend, _ = read_mean_rewards_for_plot(files, 1, default_mean_reward_recency, default_test_episode_count)
    train_re_trend = np.array(train_re_trend).T
    test_re_trend = np.array(test_re_trend).T[0:default_test_episode_count, :]
    train_mr_trend = np.array(train_mr_trend).T
    train_mr_trend[0:default_mean_reward_recency] = float('-inf')

    cmaps = plt.cm.get_cmap('tab10')
    fig = plt.figure(1, figsize=(10, 3))

    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(train_re_trend, color=cmaps(7), label='training reward', linewidth=0.5)
    ax1.plot(train_mr_trend, color=cmaps(3), label='training mean reward', linewidth=2)
    ax1.set_ylim([-100, max(300, np.max(train_re_trend) + 10, np.max(test_re_trend) + 10)])
    ax1.axhline(y=200, linestyle='dashed', color='black', linewidth=0.5)
    ax1.legend(loc=4, fontsize=12)
    ax1_2 = ax1.twiny()
    ax1_2.plot(test_re_trend, color=cmaps(0), label='test\nreward', linewidth=2)
    ax1_2.legend(loc=2, fontsize=12)
    ax1.set_xlabel('Training Episode', fontsize=14)
    ax1_2.tick_params(axis='x', colors=cmaps(0))
    ax1_2.set_xlabel('Test Episode', color=cmaps(0), fontsize=14)
    plt.subplots_adjust(hspace=0.3)
    fig.savefig(join(plot_dir, '{}.png'.format(result_name)), bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    best_network_size = default_network_size
    _, files = _get_file_path(folder_path=netsize_exp_dir,
                              framework=default_framework,
                              extension_name='.csv',
                              hidden_layer_dimensions=best_network_size,
                              mean_reward=None,
                              alpha=default_alpha,
                              gamma=default_gamma,
                              epsilon_start=default_epsilon_start,
                              epsilon_decay=default_epsilon_decay,
                              epsilon_min=default_epsilon_min,
                              replay_memory_size=default_replay_memory_size,
                              replay_sample_size=default_replay_sample_size,
                              use_dropout=False,
                              timestamp=None)
    plot_best_result(files, result_name='best_result')

    _, files = _get_file_path(folder_path=netsize_exp_dir,
                              framework=default_framework,
                              extension_name='.csv',
                              hidden_layer_dimensions=best_network_size,
                              mean_reward=None,
                              alpha=default_alpha,
                              gamma=default_gamma,
                              epsilon_start=default_epsilon_start,
                              epsilon_decay=default_epsilon_decay,
                              epsilon_min=default_epsilon_min,
                              replay_memory_size=default_replay_sample_size,
                              replay_sample_size=default_replay_sample_size,
                              use_dropout=False,
                              timestamp=None)
    plot_best_result(files, result_name='no_experience_replay')

