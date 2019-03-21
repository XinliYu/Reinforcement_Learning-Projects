from Utilities import *
import matplotlib.pyplot as plt
from exp_setup import *
from LunarLander import _get_file_path

x_tick_labels = ['/\n'.join([str(i) for i in network_size]) for network_size in exp_network_sizes]
train_mean_reward_list = []
test_mean_reward_list = []

for network_size in exp_network_sizes:  # [128, 64, 32]
    _, files = _get_file_path(folder_path=netsize_exp_dir,
                              framework=default_framework,
                              extension_name='.csv',
                              hidden_layer_dimensions=network_size,
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
    train_mr, test_mr, _, _, _, _ = read_mean_rewards_for_plot(files, 6, default_mean_reward_recency, default_test_episode_count)
    train_mean_reward_list.append(train_mr)
    test_mean_reward_list.append(test_mr)

_, files = _get_file_path(folder_path=netsize_exp_dir,
                          framework=default_framework,
                          extension_name='.csv',
                          hidden_layer_dimensions=default_network_size,
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
train_mr, test_mr, _, _, _, _ = read_mean_rewards_for_plot(files, 6, default_mean_reward_recency, default_test_episode_count)
train_mean_reward_list.insert(7, train_mr)
test_mean_reward_list.insert(7, test_mr)
x_tick_labels.insert(7, '256/\n128\n($\\mathbf{no rep}$)')
fig = plt.figure(1, figsize=(10, 8))

ax1 = fig.add_subplot(2, 1, 1)
bp = ax1.boxplot(train_mean_reward_list)
set_default_box_plot_color(bp)
ax1.set_xlabel('Network Size', fontsize=default_label_font_size)
ax1.set_ylabel('Mean Rewards of last 100 episodes', fontsize=default_label_font_size - 3)
ax1.set_title('(a) Final Mean Rewards for Training', fontsize=default_title_font_size)
ax1.set_ylim([-100, max(300, max([max(l) for l in train_mean_reward_list]) + 10)])
ax1.tick_params(labelsize=default_tick_font_size - 1)

ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
bp = ax2.boxplot(test_mean_reward_list)
set_default_box_plot_color(bp)
ax2.set_xlabel('Network Size', fontsize=default_label_font_size)
ax2.set_ylabel('Mean Rewards of all 1000 episodes', fontsize=default_label_font_size - 3)
ax2.set_title('(b) Final Mean Rewards for Test', fontsize=default_title_font_size)
ax2.set_xticklabels(x_tick_labels)
ax2.set_ylim([-100, max(300, max([max(l) for l in test_mean_reward_list]) + 10)])
ax2.tick_params(labelsize=default_tick_font_size - 1)
plt.subplots_adjust(hspace=0.45)

fig.savefig(join(plot_dir, 'netsize.png'), bbox_inches='tight')
plt.close()

