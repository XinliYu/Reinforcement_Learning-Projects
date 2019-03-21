import numpy as np
from Utilities import *
import matplotlib.pyplot as plt
from exp_setup import *
from Utilities import _get_file_path


def plot_hyper_parameter_tune(get_files, paras, para_name, para_label):
    train_mean_rewards = []
    test_mean_rewards = []
    train_mean_reward_trends = []
    test_reward_trends = []
    for para in paras:
        _, files = get_files(para)

        train_mr, test_mr, train_re_trend, test_re_trend, train_mr_trend, test_mr_trend = \
            read_mean_rewards_for_plot(files, 1, default_mean_reward_recency, default_test_episode_count)

        train_mean_rewards.append(train_mr[0])
        test_mean_rewards.append(test_mr[0])
        train_mean_reward_trends.append(train_mr_trend[0])
        test_reward_trends.append(test_re_trend[0])

    train_mean_reward_trends = np.array(train_mean_reward_trends).T
    train_mean_reward_trends[0:default_mean_reward_recency] = float('-inf')

    para_labels = [str(paras[i]) for i in range(len(paras))]

    fig = plt.figure(1, figsize=(10, 3))
    ax1 = fig.add_subplot(1, 1, 1)
    for i in range(len(paras)):
        ax1.plot(train_mean_reward_trends[:, i], label=para_labels[i], linewidth=2)
    ax1.axhline(y=200, linestyle='dashed', color='black', linewidth=0.5)
    ax1.set_xlim([default_mean_reward_recency - 1, default_train_episode_count])
    ax1.set_ylim([max(-200, np.min(train_mean_reward_trends) - 10), max(300, np.max(train_mean_reward_trends) + 10)])
    ax1.set_xlabel('Training Episode', fontsize=default_label_font_size)
    ax1.set_ylabel('100-Episode Mean Reward', fontsize=default_label_font_size)
    ax1.set_title('(a) 100-episode mean rewards over time for {}s'.format(para_label), fontsize=default_title_font_size)
    ax1.tick_params(labelsize=default_tick_font_size)
    ax1.legend(fontsize=default_tick_font_size)
    fig.savefig(join(plot_dir, '{}_1.png'.format(para_name)), bbox_inches='tight')
    plt.close()

    fig = plt.figure(1, figsize=(10, 3))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(train_mean_rewards, label="train mean rewards", marker='o')
    ax1.plot(test_mean_rewards, label="test mean rewards", marker='o')
    ax1.axhline(y=200, linestyle='dashed', color='black', linewidth=0.5)
    ax1.set_ylim([max(-200, np.min(train_mean_rewards) - 10, np.min(test_mean_rewards) - 10), max(300, np.max(train_mean_rewards) + 10, np.max(test_mean_rewards) + 10)])
    ax1.set_xlabel(para_label, fontsize=default_label_font_size - 1)
    ax1.set_ylabel('Mean Reward', fontsize=default_label_font_size - 3)
    ax1.set_xticklabels([para_labels[0]] + para_labels)  # ! A bug of matplotlib
    ax1.set_title('(b) Training & test mean rewards for {}s'.format(para_label), fontsize=default_title_font_size - 3)
    ax1.tick_params(labelsize=default_tick_font_size)
    ax1.legend(fontsize=default_tick_font_size - 1)
    ax2 = fig.add_subplot(1, 2, 2)
    bp = ax2.boxplot(test_reward_trends)
    set_default_box_plot_color(bp)
    ax2.set_ylim([max(-200, np.min(test_reward_trends) - 10), max(300, np.max(test_reward_trends) + 10)])
    ax2.set_xlabel(para_label, fontsize=default_label_font_size - 1)
    ax2.set_ylabel('Test Reward', fontsize=default_label_font_size - 3)
    ax2.set_title('(c) Test mean reward box plot for {}s'.format(para_label), fontsize=default_title_font_size - 3)
    ax2.set_xticklabels(para_labels)
    ax2.tick_params(labelsize=default_tick_font_size)
    plt.subplots_adjust(wspace=0.22)
    fig.savefig(join(plot_dir, '{}_2.png'.format(para_name)), bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    plot_hyper_parameter_tune(lambda gamma: _get_file_path(folder_path=netsize_exp_dir if gamma == default_gamma else gamma_exp_dir,
                                                           framework=default_framework,
                                                           extension_name='.csv',
                                                           hidden_layer_dimensions=default_network_size,
                                                           mean_reward=None,
                                                           alpha=default_alpha,
                                                           gamma=gamma,
                                                           epsilon_start=default_epsilon_start,
                                                           epsilon_decay=default_epsilon_decay,
                                                           epsilon_min=default_epsilon_min,
                                                           replay_memory_size=default_replay_memory_size,
                                                           replay_sample_size=default_replay_sample_size,
                                                           use_dropout=False,
                                                           timestamp=None),
                              paras=exp_gammas, para_name='gamma', para_label=r'$\gamma$')

    plot_hyper_parameter_tune(lambda alpha: _get_file_path(folder_path=netsize_exp_dir if alpha == default_alpha else alpha_exp_dir,
                                                           framework=default_framework,
                                                           extension_name='.csv',
                                                           hidden_layer_dimensions=default_network_size,
                                                           mean_reward=None,
                                                           alpha=alpha,
                                                           gamma=default_gamma,
                                                           epsilon_start=default_epsilon_start,
                                                           epsilon_decay=default_epsilon_decay,
                                                           epsilon_min=default_epsilon_min,
                                                           replay_memory_size=default_replay_memory_size,
                                                           replay_sample_size=default_replay_sample_size,
                                                           use_dropout=False,
                                                           timestamp=None),
                              paras=exp_alphas, para_name='alpha', para_label=r'$\alpha$')

    plot_hyper_parameter_tune(lambda epd: _get_file_path(folder_path=netsize_exp_dir if epd == default_epsilon_decay else epsilon_decay_exp_dir,
                                                         framework=default_framework,
                                                         extension_name='.csv',
                                                         hidden_layer_dimensions=default_network_size,
                                                         mean_reward=None,
                                                         alpha=default_alpha,
                                                         gamma=default_gamma,
                                                         epsilon_start=default_epsilon_start,
                                                         epsilon_decay=epd,
                                                         epsilon_min=default_epsilon_min,
                                                         replay_memory_size=default_replay_memory_size,
                                                         replay_sample_size=default_replay_sample_size,
                                                         use_dropout=False,
                                                         timestamp=None),
                              paras=exp_epsilon_decays, para_name='epsilon_decay', para_label=r'$\epsilon_{\operatorname{decay}}$')

    plot_hyper_parameter_tune(lambda epm: _get_file_path(folder_path=netsize_exp_dir if epm == default_epsilon_min else epsilon_min_exp_dir,
                                                         framework=default_framework,
                                                         extension_name='.csv',
                                                         hidden_layer_dimensions=default_network_size,
                                                         mean_reward=None,
                                                         alpha=default_alpha,
                                                         gamma=default_gamma,
                                                         epsilon_start=default_epsilon_start,
                                                         epsilon_decay=default_epsilon_decay,
                                                         epsilon_min=epm,
                                                         replay_memory_size=default_replay_memory_size,
                                                         replay_sample_size=default_replay_sample_size,
                                                         use_dropout=False,
                                                         timestamp=None),
                              paras=exp_epsilon_mins, para_name='epsilon_min', para_label=r'$\epsilon_{\operatorname{min}}$')

    plot_hyper_parameter_tune(lambda eps: _get_file_path(folder_path=netsize_exp_dir if eps == default_epsilon_start else epsilon_start_exp_dir,
                                                         framework=default_framework,
                                                         extension_name='.csv',
                                                         hidden_layer_dimensions=default_network_size,
                                                         mean_reward=None,
                                                         alpha=default_alpha,
                                                         gamma=default_gamma,
                                                         epsilon_start=eps,
                                                         epsilon_decay=default_epsilon_decay,
                                                         epsilon_min=default_epsilon_min,
                                                         replay_memory_size=default_replay_memory_size,
                                                         replay_sample_size=default_replay_sample_size,
                                                         use_dropout=False,
                                                         timestamp=None),
                              paras=exp_epsilon_starts, para_name='epsilon_start', para_label=r'$\epsilon_{\operatorname{start}}$')

