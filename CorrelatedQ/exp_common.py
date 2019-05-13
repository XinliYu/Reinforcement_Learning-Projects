from os import path
from glob import glob
from time import time
from numpy import array, mean, arange, sum as npsum
from matplotlib import pyplot as plt
from cvxopt import solvers
from sys import platform
from soccer_env import SoccerGame
from itertools import product
import pickle
import matplotlib

if platform == "linux" or platform == "linux2":
    matplotlib.use('Agg')

result_folder = path.join('.', 'results')


def get_files_by_pattern(dir_path: str, pattern: str, fullpath: bool = True):
    if path.isdir(dir_path):
        if fullpath:
            return [f for f in glob(path.join(dir_path, pattern)) if path.isfile(f)]
        else:
            return [path.basename(f) for f in glob(path.join(dir_path, pattern)) if path.isfile(f)]


def read_stats_from_cvs(csv_file):
    with open(csv_file, 'r') as f:
        stats = []
        for line in f:
            splits = line.split(',')
            row = []
            for split in splits:
                try:
                    val = int(split)
                except:
                    val = float(split)
                row.append(val)
            stats.append(row)
        return stats


def output_results(exp_name: str, stats_target_state_and_action, stats_target_state, stats_player1, stats_player2, gamma, alpha, alpha_decay, alpha_min, epsilon, epsilon_decay, epsilon_min, rand_init_Q, normalized_Q_update):
    file_pattern = '{exp_name}_g{gamma}_a{alpha}_ad{alpha_decay}_am{alpha_min}_e{epsilon}_ed{epsilon_decay}_em{epsilon_min}_{rand_init_Q}_{normalized_Q_update}_{time}' \
        .format(exp_name=exp_name, gamma=int(gamma * 100),
                alpha=int(alpha * 100), alpha_decay=int(alpha_decay * 100000), alpha_min=int(alpha_min * 10000),
                epsilon=int(epsilon * 100), epsilon_decay=int(epsilon_decay * 100000), epsilon_min=int(epsilon_min * 1000),
                rand_init_Q=rand_init_Q, normalized_Q_update=normalized_Q_update,
                time=str(int(time())))
    file_pattern = path.join(result_folder, file_pattern)
    with open(file_pattern + '_sa.csv', 'w') as file:
        for stat_row in stats_target_state_and_action:
            file.write(','.join([str(cell) for cell in stat_row]) + '\n')
        file.flush()
    with open(file_pattern + '_s.csv', 'w') as file:
        for stat_row in stats_target_state:
            file.write(','.join([str(cell) for cell in stat_row]) + '\n')
        file.flush()
    with open(file_pattern + '_p1.csv', 'w') as file:
        for stat_row in stats_player1:
            file.write(','.join([str(cell) for cell in stat_row]) + '\n')
        file.flush()
    with open(file_pattern + '_p2.csv', 'w') as file:
        for stat_row in stats_player2:
            file.write(','.join([str(cell) for cell in stat_row]) + '\n')
        file.flush()

    rows = len(stats_target_state_and_action)
    interval = 1 if rows <= 1500 else rows // 1500
    data_idxes = range(0, rows, interval)
    plot_q_diff(stats_target_state_and_action, data_idxes, file_pattern)
    if exp_name.startswith('ce-q'):
        plot_policy_evolution_correlated(stats_target_state, file_pattern)
    else:
        plot_policy_evolution_uncorrelated(stats_target_state, file_pattern)


def plot_q_diff(stats, data_idxes, save_file):
    stats = array(stats)
    x_vals = stats[:, 0] if data_idxes is None else stats[data_idxes, 0]
    y_vals = stats[:, 9] if data_idxes is None else stats[data_idxes, 9]
    plt.plot(x_vals, y_vals, linewidth=0.7)
    plt.xlabel('Simulation Iteration')
    plt.ylabel('Q-Value Difference')
    plt.ylim(0, 0.5)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.savefig(save_file + '.png', dpi=600)
    plt.close()


def plot_policy_evolution_uncorrelated(stats, save_path):
    stats = array(stats)
    stat_len = len(stats)
    policy_history_player_1 = []
    policy_history_player_2 = []
    stat_interval = 1000

    stat_idxes = list(range(stat_len - stat_interval, -1, -stat_interval))
    for i in stat_idxes:
        policy_history_player_1.append(mean(stats[i:i + stat_interval, -10:-5], axis=0))
        policy_history_player_2.append(mean(stats[i:i + stat_interval, -5:], axis=0))

    action_labels = ['N', 'S', 'W', 'E', 'O']

    plt.figure(figsize=(5, 12))
    ax1 = plt.subplot(121)
    ax1.pcolor(policy_history_player_1)
    ax1.set_xticks(arange(len(action_labels)) + 0.5)
    ax1.set_yticks(arange(len(stat_idxes)))
    ax1.set_xticklabels(action_labels, fontsize=7)
    ax1.set_yticklabels(stat_idxes)

    ax2 = plt.subplot(122)
    ax2.pcolor(policy_history_player_2)
    ax2.set_xticks(arange(len(action_labels)) + 0.5)
    ax2.set_yticks([])
    ax2.set_yticklabels([])
    ax2.set_xticklabels(action_labels, fontsize=7)

    plt.savefig(save_path + '_all.png', dpi=600)
    plt.close()

    plt.figure(1)
    ax1 = plt.subplot(211)
    final_dist = policy_history_player_1[0].reshape((1, 5))
    ax1.imshow(final_dist)
    ax1.set_xticks(arange(len(action_labels)))
    ax1.set_xticklabels(action_labels)
    ax1.set_xlabel('Player 1 (Row Player)', fontsize=11)
    ax1.set_yticks([])
    ax1.set_yticklabels([])
    for j in range(5):
        ax1.text(j, 0, '{:.2f}'.format(final_dist[0, j]), ha="center", va="center", color='w' if final_dist[0, j] < 0.2 else 'k', fontsize=14)

    ax2 = plt.subplot(212)
    final_dist = policy_history_player_2[0].reshape((1, 5))
    ax2.imshow(final_dist)
    ax2.set_xticks(arange(len(action_labels)))
    ax2.set_xticklabels(action_labels)
    ax2.set_xlabel('Player 2 (Column Player)', fontsize=11)
    ax2.set_yticks([])
    ax2.set_yticklabels([])
    for j in range(5):
        ax2.text(j, 0, '{:.2f}'.format(final_dist[0, j]), ha="center", va="center", color='w' if final_dist[0, j] < 0.2 else 'k', fontsize=14)

    plt.savefig(save_path + '_final.png', dpi=600)
    plt.close()


def plot_policy_evolution_correlated(stats, save_path):
    stats = array(stats)
    stat_len = len(stats)
    policy_history = []
    stat_interval = 1000

    stat_idxes = list(range(stat_len - stat_interval, -1, -stat_interval))
    for i in stat_idxes:
        policy_history.append(mean(stats[i:i + stat_interval, -25:], axis=0))

    action_labels = ['N', 'S', 'W', 'E', 'O']
    product_action_labels = [l1 + l2 for l1, l2, in product(action_labels, action_labels)]
    plt.figure(figsize=(5, 12))
    ax1 = plt.subplot(111)
    c = ax1.pcolor(policy_history)
    ax1.set_xticks(arange(len(product_action_labels)) + 0.5)
    ax1.set_yticks(arange(len(stat_idxes)))
    ax1.set_xticklabels(product_action_labels, fontsize=7)
    ax1.set_yticklabels(stat_idxes)
    plt.colorbar(c, ax=ax1, orientation='horizontal')
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    plt.tight_layout()
    plt.savefig(save_path + '_all.png', dpi=500)
    plt.close()

    plt.figure(figsize=(6, 6))
    final_policy = policy_history[0].reshape((5, 5))
    ax1 = plt.subplot(111)
    ax1.imshow(final_policy)
    ax1.set_xticks(arange(len(action_labels)))
    ax1.set_yticks(arange(len(action_labels)))
    ax1.set_xticklabels(action_labels)
    ax1.set_yticklabels(action_labels)
    ax1.set_xlabel('Player 2 (Column Player)', fontsize=11)
    ax1.set_ylabel('Player 1 (Row Player)', fontsize=11)
    for i in range(5):
        for j in range(5):
            ax1.text(j, i, '{:.2f}'.format(final_policy[i, j]), ha="center", va="center", color='w' if final_policy[i, j] < 0.2 else 'k', fontsize=14)
    plt.savefig(save_path + '_final2d.png', dpi=600)
    plt.close()

    ax2 = plt.subplot(211)
    final_dist = npsum(final_policy, axis=1).reshape((1, 5))
    ax2.imshow(final_dist)
    ax2.set_xticks(arange(len(action_labels)))
    ax2.set_xticklabels(action_labels)
    ax2.set_xlabel('Player 1 (Row Player)', fontsize=11)
    ax2.set_yticks([])
    ax2.set_yticklabels([])
    for j in range(5):
        ax2.text(j, 0, '{:.2f}'.format(final_dist[0, j]), ha="center", va="center", color='w' if final_dist[0, j] < 0.2 else 'k', fontsize=14)

    ax3 = plt.subplot(212)
    final_dist = npsum(final_policy, axis=0).reshape((1, 5))
    ax3.imshow(final_dist)
    ax3.set_xticks(arange(len(action_labels)))
    ax3.set_xticklabels(action_labels)
    ax3.set_xlabel('Player 2 (Column Player)', fontsize=11)
    ax3.set_yticks([])
    ax3.set_yticklabels([])
    for j in range(5):
        ax3.text(j, 0, '{:.2f}'.format(final_dist[0, j]), ha="center", va="center", color='w' if final_dist[0, j] < 0.2 else 'k', fontsize=14)

    plt.savefig(save_path + '_final.png', dpi=600)
    plt.close()


def get_pre_build_seq(env):
    seq_file_path = path.join('.', 'prebuilt_seq.dat')
    if path.exists(seq_file_path):
        with open(seq_file_path, 'rb') as f:
            prebuilt_seq, who_goes_first = pickle.load(f)
    else:
        prebuilt_seq, who_goes_first = env.action_seq(1000000)
        with open(seq_file_path, 'wb') as f:
            pickle.dump((prebuilt_seq, who_goes_first), f)
            f.flush()
    return prebuilt_seq, who_goes_first


def get_initialization(fixed_initialization):
    if fixed_initialization:
        init_pos_a = (0, 3)
        init_pos_b = (0, 1)
        init_ball_owner = 1
    else:
        init_pos_a = None
        init_pos_b = None
        init_ball_owner = None
    return init_pos_a, init_pos_b, init_ball_owner


def exp_common(model_class, exp_name, max_iter=1000000, test_max_iter=10000,
               test_mode=False, fixed_initialization=True, use_prebuilt_seq=True, gamma=0.6,
               alpha=0.2, alpha_decay=0.999995, alpha_min=0.001,
               epsilon=1, epsilon_decay=1, epsilon_min=1,
               progress_msg_interval=5000, test_progress_msg_interval=1000,
               rand_init_Q=False, normalized_Q_update=False):
    solvers.options['show_progress'] = False
    init_pos_a, init_pos_b, init_ball_owner = get_initialization(fixed_initialization)
    max_iter = test_max_iter if test_mode else max_iter
    report_interval = test_progress_msg_interval if test_mode else progress_msg_interval
    env = SoccerGame(init_pos_a=init_pos_a, init_pos_b=init_pos_b, init_ball_owner=init_ball_owner)
    players = model_class(state_count=env.state_count, action_count=env.action_count,
                          gamma=gamma, alpha=alpha, alpha_decay=alpha_decay, min_alpha=alpha_min,
                          epsilon=epsilon, epsilon_decay=epsilon_decay, min_epsilon=epsilon_min,
                          rand_init_Q=rand_init_Q, normalized_Q_update=normalized_Q_update)

    curr_state = env.state()
    state_to_monitor = 71
    stats_target_state_and_action, stats_target_state, stats_player1, stats_player2 = [], [], [], []
    train_start_time = time()
    max_q1_diff, max_q2_diff = 0, 0

    if use_prebuilt_seq:
        prebuilt_seq, who_goes_first = get_pre_build_seq(env)

    for i in range(max_iter):
        if use_prebuilt_seq:
            act1, act2 = prebuilt_seq[i]
            first = who_goes_first[i]
        else:
            act1, act2 = players.choose_action(curr_state)
            first = None
        reward, first = env.step(act1, act2, first)
        next_state = env.state()
        stat1, stat2 = players.step_all_players(curr_state, next_state, (act1, act2), (reward, -reward))
        prev_q1, post_q1, prev_v1, post_v1, alpha1, epsilon1 = stat1
        prev_q2, post_q2, prev_v2, post_v2, alpha2, epsilon2 = stat2
        # pre_q2, post_q2, prev_v2, post_v2, alpha2, epsilon2 = players.step(1, curr_state, next_state, (act2, act1), -reward)
        p1_policy = list(players.get_policy(0, curr_state))
        p2_policy = list(players.get_policy(1, curr_state))
        stats_player1.append([curr_state, next_state, act1, act2, first, reward, prev_q1, post_q1, prev_v1, post_v1, alpha1, epsilon1] + p1_policy)
        stats_player2.append([curr_state, next_state, act1, act2, first, -reward, prev_q2, post_q2, prev_v2, post_v2, alpha2, epsilon2] + p2_policy)
        if curr_state == state_to_monitor:
            q1_diff = abs(post_q1 - prev_q1)
            q2_diff = abs(post_q2 - prev_q2)
            max_q1_diff = max(max_q1_diff, q1_diff)
            max_q2_diff = max(max_q2_diff, q2_diff)
            rcd = [i, curr_state, next_state, act1, act2, first, reward, prev_q1, post_q1, q1_diff, prev_q2, post_q2, q2_diff] + p1_policy + p2_policy
            stats_target_state.append(rcd)
            if act1 == 1 and act2 == 4:
                stats_target_state_and_action.append(rcd)
            q1_diff = abs(post_q1 - prev_q1)
            q2_diff = abs(post_q2 - prev_q2)
            max_q1_diff = max(max_q1_diff, q1_diff)
            max_q2_diff = max(max_q2_diff, q2_diff)

        if reward != 0:
            env.reset()
            curr_state = env.state()
        else:
            curr_state = next_state
        if (i != 0 and i % report_interval == 0) or i == max_iter - 1:
            stat_state_hit = len(stats_target_state_and_action)
            print("Episode:{}, Stat State Hit:{}, Max Q1 DIFF: {}, Max Q2 DIFF: {}, Time Per Loop: {:.4f}".format(
                i, stat_state_hit, max_q1_diff, max_q2_diff, (time() - train_start_time) / i))
            max_q1_diff = 0
            max_q2_diff = 0

    output_results(exp_name=exp_name, stats_target_state_and_action=stats_target_state_and_action,
                   stats_target_state=stats_target_state,
                   stats_player1=stats_player1,
                   stats_player2=stats_player2,
                   gamma=gamma,
                   alpha=alpha, alpha_decay=alpha_decay, alpha_min=alpha_min,
                   epsilon=epsilon, epsilon_decay=epsilon_decay, epsilon_min=epsilon_min,
                   rand_init_Q=rand_init_Q, normalized_Q_update=normalized_Q_update)
