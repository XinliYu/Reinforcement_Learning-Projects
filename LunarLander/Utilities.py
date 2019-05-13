from os.path import isfile, isdir, join, basename, splitext
from glob import glob
import pandas as pd
import re


def get_files_by_pattern(dir_path: str, pattern: str, full_path: bool = True):
    if isdir(dir_path):
        if full_path:
            return [f for f in glob(join(dir_path, pattern)) if isfile(f)]
        else:
            return [basename(f) for f in glob(join(dir_path, pattern)) if isfile(f)]


def get_path_without_extension(path: str):
    return splitext(path)[0]


def _get_file_path(folder_path, extension_name, framework, hidden_layer_dimensions, mean_reward, alpha, gamma, epsilon_start, epsilon_decay, epsilon_min, replay_memory_size, replay_sample_size, use_dropout, timestamp):
    file_pattern_no_exname = "LL_{framework}_size{hidden_layer_dimensions}_mr{mean_reward}_lr{learning_rate}_ga{gamma}_epd{epsilon_decay}{epsilon_start}_epm{epsilon_min}_re{replay_sample_size}{replay_memory_size}{use_dropout}_t{timestamp}".format(
        hidden_layer_dimensions='-'.join([str(d) for d in hidden_layer_dimensions]),
        framework=framework,
        mean_reward=int(mean_reward) if mean_reward is not None else '*',
        learning_rate=int(alpha * 1e5),
        gamma=int(gamma * 1e3),
        epsilon_start='' if epsilon_start == 1 else '_eps_' + str(int(epsilon_start * 1e3)),
        epsilon_decay=int(epsilon_decay * 1e3),
        epsilon_min=int(epsilon_min * 1e3),
        replay_sample_size=replay_sample_size,
        replay_memory_size='' if replay_memory_size == int(2 ** 16) else '_mem_' + str(replay_memory_size),
        use_dropout='_dropout' if use_dropout else '',
        timestamp=timestamp if timestamp is not None else '*')
    files = get_files_by_pattern(folder_path, file_pattern_no_exname + extension_name)
    return join(folder_path, file_pattern_no_exname), files if files else None


def _read_train_csv_mrs(file):
    data = pd.read_csv(file)
    return list(data['Total Reward']), list(data['Mean Reward'])


def _read_test_csv_mrs(file):
    data = pd.read_csv(file)
    return list(data['Total Reward']), list(data['Mean Reward'])


def read_mean_rewards_for_plot(files, top, mean_reward_recency, test_episode_count):
    mr_files = []
    for file in files:
        if '_test' in file:
            match = re.findall(r'_tmr(\-?[0-9]+)\.', file)
            mr = int(match[0])
            mr_files.append((-mr, file))

    mr_files = sorted(mr_files)
    top = min(top, len(mr_files))
    train_mr = [None] * top
    test_mr = [None] * top
    train_re_trends = [None] * top
    test_re_trends = [None] * top
    train_mr_trends = [None] * top
    test_mr_trends = [None] * top
    for i in range(top):
        test_mr[i] = -mr_files[i][0]
        test_file = mr_files[i][1]
        splits = splitext(test_file)
        train_file = splits[0][0:test_file.index('_test')] + splits[1]
        match = re.findall(r'_mr(\-?[0-9]+)_', train_file)
        train_mr[i] = int(match[0])
        train_re_trends[i], train_mr_trends[i] = _read_train_csv_mrs(train_file)
        test_re_trends[i], test_mr_trends[i] = _read_test_csv_mrs(test_file)
    return train_mr, test_mr, train_re_trends, test_re_trends, train_mr_trends, test_mr_trends
