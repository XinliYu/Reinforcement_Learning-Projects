from os.path import join

default_framework = 'pytorch'
default_network_size = [256, 128]
default_gamma = 0.99
default_epsilon_start = 1.0
default_epsilon_min = 0.0
default_epsilon_decay = 0.998
default_alpha = 1e-4
default_replay_memory_size = 2 ** 16
default_replay_sample_size = 32
default_train_episode_count = 2000
default_test_episode_count = 100
default_mean_reward_recency = 100
default_test_repeat = 10
network_selection_training_repeat = 6
hyper_parameter_tuning_repeat = 2

default_line_color = (31 / 255, 119 / 255, 180 / 255)
default_highlight_color = (214 / 255, 39 / 255, 40 / 255)

exp_network_sizes = ([32, 16], [64, 32], [96, 48], [128, 64], [128, 64, 32], [192, 96], [256, 128], [256, 128, 64], [512, 256], [512, 256, 128])
exp_epsilon_starts = (0.01, 0.2, 0.5, 0.7, 0.9, 1.0)
exp_epsilon_mins = (0, 0.01, 0.03, 0.05, 0.1, 0.2)
exp_epsilon_decays = (0.9, 0.95, 0.97, 0.99, 0.994, 0.998)
exp_alphas = (1e-2, 1e-3, 5 * 1e-4, 1e-4, 0.5 * 1e-4, 1e-5)
exp_gammas = (0.87, 0.93, 0.96, 0.99, 0.995, 0.999)
exp_replay_memory_sizes = (100, 1000, 5000, 10000, 30000, 2 ** 16)

plot_dir = join('.', 'plots')
exp_root_dir = join('.', 'models')
netsize_exp_dir = join(exp_root_dir, 'netsize')
gamma_exp_dir = join(exp_root_dir, 'gamma')
epsilon_start_exp_dir = join(exp_root_dir, 'epsilon_start')
epsilon_min_exp_dir = join(exp_root_dir, 'epsilon_min')
epsilon_decay_exp_dir = join(exp_root_dir, 'epsilon_decay')
alpha_exp_dir = join(exp_root_dir, 'alpha')
memory_size_exp_dir = join(exp_root_dir, 'memory_size')
sample_size_exp_dir = join(exp_root_dir, 'sample_size')

default_label_font_size = 14
default_title_font_size = 16
default_tick_font_size = 12


def set_default_box_plot_color(box_plot):
    for component in ['boxes', 'whiskers', 'caps', 'fliers']:
        for obj in box_plot[component]:
            obj.set(color=default_line_color, linewidth=2)
    for median in box_plot['medians']:
        median.set(color=default_highlight_color, linewidth=2)

