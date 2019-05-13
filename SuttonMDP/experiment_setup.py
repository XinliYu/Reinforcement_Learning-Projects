from td_lambda import generate_state_sequence

use_cached_files = False  # global switch for using cached files to generate the figures
plot_label_size = 13  # axis label font size

ground_truth = [1 / 6, 1 / 3, 1 / 2, 2 / 3, 5 / 6]
num_train_sets = 100
num_episodes_per_training_set = 10

max_seq_len = 13
soft_min_seq_len = 4
training_sets = None


def generate_training_set():
    global training_sets
    training_sets = [
        [
            generate_state_sequence(state_count=len(ground_truth) + 2,
                                    max_len=max_seq_len,
                                    soft_min_len=soft_min_seq_len) for _ in range(num_episodes_per_training_set)
        ]
        for _ in range(num_train_sets)
    ]


generate_training_set()
